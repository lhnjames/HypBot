"""Train and evaluate HypBot on Fox8-23, BotSim-24, or MGTAB.

Examples:
    python train.py --dataset fox8   --data_dir data/fox8_23
    python train.py --dataset botsim --data_dir data/botsim_24
    python train.py --dataset mgtab  --data_dir data/mgtab
    python train.py --dataset fox8   --data_dir data/fox8_23 --auto_k
"""

import argparse
import math
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from hypbot.data import knn_graph, load_mgtab, load_tabular, mean_adjacency
from hypbot.model import HypBot, eliminate_sectors

# Dataset-specific settings used in the paper.
PRESETS = {
    "fox8": dict(model=dict(d_proj=64, d_hidden=128, dropout=0.25),
                 train=dict(lr=3e-4, epochs=120, patience=15, alpha=0.75, gamma=2.0,
                            ent=0.03, warmup=20, batch=128)),
    "botsim": dict(model=dict(d_proj=32, d_hidden=64, dropout=0.30),
                   train=dict(lr=3e-4, epochs=120, patience=15, alpha=0.80, gamma=2.0,
                              ent=0.03, warmup=20, batch=256)),
    "mgtab": dict(model=dict(d_proj=64, d_hidden=256, dropout=0.30),
                  train=dict(lr=2e-4, epochs=80, patience=15, alpha=0.85, gamma=0.5,
                             ent=0.0, warmup=12, batch=512)),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def focal_loss(logit, y, alpha, gamma):
    y = y.float()
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    a = alpha * y + (1 - alpha) * (1 - y)
    pt = p * y + (1 - p) * (1 - y)
    return (-a * (1 - pt) ** gamma * torch.log(pt)).mean()


def fit(model, x, y, adj, tr, va, cfg, weight_decay=1e-4):
    """Minibatch training with early stopping on the validation split."""
    opt = torch.optim.AdamW(model.param_groups(cfg["lr"], weight_decay),
                            lr=cfg["lr"], weight_decay=weight_decay)
    y_np, va_np = y.cpu().numpy(), va.numpy()
    best, best_state, wait = -float("inf"), None, 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        lam = cfg["ent"] * max(0.0, 1.0 - (ep - 1) / cfg["warmup"])
        perm = tr[torch.randperm(len(tr))]
        for s in range(0, len(perm), cfg["batch"]):
            b = perm[s:s + cfg["batch"]].to(x.device)
            x_agg = model.aggregate(x, adj)[b]
            logit, H = model.forward_features(x[b], x_agg)
            loss = focal_loss(logit, y[b], cfg["alpha"], cfg["gamma"])
            if lam > 0:   # sharpen sector membership of bots early in training
                bot = (y[b] == 1).float()
                loss = loss + lam * (bot * H / (math.log(model.K) + 1e-6)).sum() / (bot.sum() + 1e-6)
            pen = model.gate_penalty()
            if pen is not None:
                loss = loss + pen
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        score = roc_auc_score(y_np[va_np], model.predict_prob(x, adj)[va.to(x.device)].cpu().numpy())
        if score > best:
            best, wait = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model


def select_threshold(prob, y):
    """Decision threshold maximizing validation macro-F1."""
    cands = np.unique(np.r_[0.5, prob])
    scores = np.array([f1_score(y, prob >= t, average="macro", zero_division=0) for t in cands])
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(cands[best[np.argmin(abs(cands[best] - 0.5))]])


def metrics(prob, y, thr):
    pred = (prob >= thr).astype(int)
    return dict(acc=accuracy_score(y, pred),
                f1=f1_score(y, pred, average="macro"),
                pre=precision_score(y, pred, average="macro", zero_division=0),
                rec=recall_score(y, pred, average="macro", zero_division=0))


def run_seed(args, seed, device):
    set_seed(seed)
    preset = PRESETS[args.dataset]
    if args.dataset == "mgtab":
        x, edge_index, labels, split = load_mgtab(args.data_dir, seed, args.degree_features)
        adj = mean_adjacency(edge_index, x.shape[0]).to(device)
        extra = dict(graph_width=x.shape[1], sparse_adj=True)
    else:
        x, labels, split = load_tabular(args.data_dir, seed)
        adj = knn_graph(x.to(device), k=args.knn_k)
        extra = {}
    x, y = x.to(device), torch.as_tensor(labels, device=device)
    tr, va = torch.as_tensor(split["train"]), torch.as_tensor(split["val"])
    build = lambda K, gated: HypBot(x.shape[1], K=K, gated=gated, **preset["model"], **extra).to(device)

    if args.auto_k:
        # Start from the truncation level T; double it while a channel keeps > T/2.
        T = args.truncation
        while True:
            set_seed(seed)
            model = fit(build(T, True), x, y, adj, tr, va, preset["train"])
            k_node, k_graph = eliminate_sectors(model, x, adj, va, y[va.to(device)])
            print(f"  seed {seed}: T={T} -> kept sectors node={k_node}, graph={k_graph}")
            if max(k_node, k_graph) <= T // 2 or T >= 256:
                break
            T *= 2
    else:
        model = fit(build(args.num_sectors, False), x, y, adj, tr, va, preset["train"])

    prob = model.predict_prob(x, adj).cpu().numpy()
    thr = select_threshold(prob[split["val"]], labels[split["val"]])
    return metrics(prob[split["test"]], labels[split["test"]], thr)


def main():
    p = argparse.ArgumentParser(description="HypBot social bot detection")
    p.add_argument("--dataset", choices=sorted(PRESETS), required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num_sectors", type=int, default=2)
    p.add_argument("--auto_k", action="store_true", help="select the number of sectors automatically")
    p.add_argument("--truncation", type=int, default=16,
                   help="initial truncation level; hypbot.model.truncation_level gives the Ishwaran-James bound")
    p.add_argument("--knn_k", type=int, default=10, help="k of the cosine k-NN graph (Fox8-23, BotSim-24)")
    p.add_argument("--degree_features", action="store_true", help="MGTAB: append per-relation log in/out-degree")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in args.seeds:
        m = run_seed(args, seed, device)
        rows.append(m)
        print(f"seed {seed}: " + "  ".join(f"{k.upper()}={v * 100:.2f}" for k, v in m.items()))
    print("mean +- std: " + "  ".join(
        f"{k.upper()}={np.mean([r[k] for r in rows]) * 100:.2f}+-{np.std([r[k] for r in rows]) * 100:.2f}"
        for k in rows[0]))


if __name__ == "__main__":
    main()
