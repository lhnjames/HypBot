"""HypBot: sector-aware hyperbolic graph learning for social bot detection.

Each account is encoded in two channels (raw features and GraphSAGE
neighborhood features). Every channel maps its input to hyperbolic polar
coordinates (radius r, direction u), predicts an angular-resolution scale
gamma(u), and summarizes angular concentration with sector prototypes.
The two five-dimensional readouts are fused only at the classifier.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class PolarEncoder(nn.Module):
    """Project features to a radius and a unit direction."""

    def __init__(self, d_in, d_proj, d_hidden, dropout):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hidden, d_proj))
        with torch.no_grad():
            nn.init.orthogonal_(self.proj[0].weight)
            nn.init.orthogonal_(self.proj[-1].weight)

    def forward(self, x):
        z = self.proj(x)
        r = z.norm(dim=-1)
        u = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        return r, u


class LocalWarpNet(nn.Module):
    """Positive angular-resolution scale gamma(u) for every direction."""

    def __init__(self, d_proj, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_proj, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.normal_(self.net[-1].weight, std=0.05)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, u):
        return F.softplus(self.net(u)).squeeze(-1) + 0.1


class SectorPrototypes(nn.Module):
    """Soft sector membership, membership entropy, and maximum alignment.

    q_ik is proportional to g_k * exp(tau_k * gamma(u_i) * phi_ik) over the
    active sectors, where phi_ik is the cosine between u_i and prototype k.
    Gates g_k are only used during automatic selection of K.
    """

    def __init__(self, K, d_proj, tau_init=5.0, gated=False):
        super().__init__()
        self.prototypes = nn.Parameter(F.normalize(torch.randn(K, d_proj), dim=-1))
        self.log_tau = nn.Parameter(torch.full((K,), math.log(tau_init)))
        self.gated = gated
        if gated:
            self.gate_logit = nn.Parameter(torch.full((K,), 3.0))
        self.register_buffer("active", torch.ones(K, dtype=torch.bool))

    def gates(self):
        return torch.sigmoid(self.gate_logit)

    @property
    def k_eff(self):
        return int(self.active.sum())

    def cosines(self, u):
        return u @ F.normalize(self.prototypes, dim=-1).T

    def forward(self, u, gamma):
        tau = self.log_tau.exp().clamp(1.0, 50.0)
        phi = self.cosines(u)
        logits = gamma.unsqueeze(-1) * tau.unsqueeze(0) * phi
        if self.gated:
            logits = logits + torch.log(self.gates() + 1e-8)
        logits = logits.masked_fill(~self.active, float("-inf"))
        logits = logits - logits.max(dim=-1, keepdim=True).values
        q = F.softmax(logits, dim=-1)
        H = -(q * (q + 1e-12).log()).sum(-1)
        A = phi.masked_fill(~self.active, float("-inf")).max(dim=-1).values
        return H, A, q


class HypBot(nn.Module):
    """Dual-channel HypBot detector."""

    def __init__(self, in_dim, d_proj=64, d_hidden=128, K=2, dropout=0.25,
                 graph_width=None, sparse_adj=False, gated=False, gate_lambda=1e-3):
        super().__init__()
        graph_dim = min(in_dim, 128) if graph_width is None else graph_width
        aggr = "add" if sparse_adj else "mean"   # a sparse adjacency already holds 1/deg
        self.sage1 = SAGEConv(in_dim, graph_dim, aggr=aggr)
        self.sage1_act = nn.Sequential(nn.LayerNorm(graph_dim), nn.GELU(), nn.Dropout(dropout))
        self.sage2 = SAGEConv(graph_dim, graph_dim, aggr=aggr)
        self.sage2_act = nn.Sequential(nn.LayerNorm(graph_dim), nn.GELU(), nn.Dropout(dropout))

        self.enc_node = PolarEncoder(in_dim, d_proj, d_hidden, dropout)
        self.warp_node = LocalWarpNet(d_proj)
        self.sectors_node = SectorPrototypes(K, d_proj, gated=gated)
        self.enc_graph = PolarEncoder(graph_dim, d_proj, d_hidden, dropout)
        self.warp_graph = LocalWarpNet(d_proj)
        self.sectors_graph = SectorPrototypes(K, d_proj, gated=gated)

        self.K, self.gated, self.gate_lambda = K, gated, gate_lambda
        self.head = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1))

    def channels(self):
        return ((self.enc_node, self.warp_node, self.sectors_node),
                (self.enc_graph, self.warp_graph, self.sectors_graph))

    def aggregate(self, x, adj):
        h = self.sage1_act(self.sage1(x, adj))
        return self.sage2_act(self.sage2(h, adj))

    def _readout(self, enc, warp, sectors, x):
        r, u = enc(x)
        g = warp(u)
        H, A, _ = sectors(u, g)
        k = max(sectors.k_eff, 2) if self.gated else self.K
        r_n = r / (r.detach().std() + 1e-6)
        H_n = H / (math.log(k) + 1e-6)
        return torch.stack([r_n, H_n, A, r_n * A, g * A], dim=-1), H

    def forward_features(self, x, x_agg):
        f_node, H_node = self._readout(*self.channels()[0], x)
        f_graph, _ = self._readout(*self.channels()[1], x_agg)
        return self.head(torch.cat([f_node, f_graph], dim=-1)).squeeze(-1), H_node

    def forward(self, x, adj):
        return self.forward_features(x, self.aggregate(x, adj))[0]

    @torch.no_grad()
    def predict_prob(self, x, adj):
        self.eval()
        return torch.sigmoid(self.forward(x, adj))

    def gate_penalty(self):
        if not self.gated or self.gate_lambda <= 0:
            return None
        g = self.sectors_node.gates().sum() + self.sectors_graph.gates().sum()
        return self.gate_lambda * g / self.K

    def param_groups(self, lr, weight_decay, gate_lr_mult=30.0):
        if not self.gated:
            return [dict(params=list(self.parameters()), lr=lr, weight_decay=weight_decay)]
        gates = [self.sectors_node.gate_logit, self.sectors_graph.gate_logit]
        ids = {id(p) for p in gates}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [dict(params=rest, lr=lr, weight_decay=weight_decay),
                dict(params=gates, lr=lr * gate_lr_mult, weight_decay=0.0)]


# ---------------------------------------------------------------------------
# Automatic selection of the number of sectors
# ---------------------------------------------------------------------------

def truncation_level(n, alpha=1.0, eps=0.01):
    """Ishwaran-James truncation level T = ceil(1 + alpha * log(4n / eps))."""
    return math.ceil(1 + alpha * math.log(4 * n / eps))


@torch.no_grad()
def sector_scores(model, x, adj, idx):
    """Soft mass m_k and alignment occupancy o_k of every sector on idx."""
    model.eval()
    idx = idx.to(x.device)
    inputs = (x, model.aggregate(x, adj))
    out = []
    for (enc, warp, sec), inp in zip(model.channels(), inputs):
        r, u = enc(inp[idx])
        _, _, q = sec(u, warp(u))
        phi = sec.cosines(u).masked_fill(~sec.active, float("-inf"))
        occ = torch.bincount(phi.argmax(-1), minlength=model.K).float() / len(idx)
        out.append((q.mean(0), occ))
    return out


@torch.no_grad()
def eliminate_sectors(model, x, adj, val_idx, y_val, z=1.0):
    """Remove sectors in increasing order of m_k + o_k under the one-SE rule."""
    model.eval()
    va = val_idx.to(x.device)
    y = y_val.to(x.device).float()

    def val_losses():
        p = torch.sigmoid(model(x, adj))[va].clamp(1e-6, 1 - 1e-6)
        return -(y * p.log() + (1 - y) * (1 - p).log())

    l0 = val_losses()
    tol = l0.mean().item() + z * (l0.std() / math.sqrt(len(l0))).item()
    secs = (model.sectors_node, model.sectors_graph)
    order = sorted((float(m[k] + o[k]), c, k)
                   for c, (m, o) in enumerate(sector_scores(model, x, adj, val_idx))
                   for k in range(model.K) if bool(secs[c].active[k]))
    for _, c, k in order:
        if secs[c].k_eff <= 1:
            continue
        secs[c].active[k] = False
        if val_losses().mean().item() > tol:
            secs[c].active[k] = True
    return secs[0].k_eff, secs[1].k_eff
