# HypBot: Sector-Aware Hyperbolic Graph Learning for Social Bot Detection

HypBot encodes each account in hyperbolic polar coordinates with a learnable
angular-resolution scale γ(u), summarizes how concentrated the account is
around a behavioral mode with sector prototypes, and keeps account-level and
neighborhood evidence in two channels that are fused only at the classifier.
The number of sectors can be selected automatically from the data.

## Requirements

```bash
pip install -r requirements.txt
```

## Data

The datasets are not redistributed here; download them from their original sources.

| Dataset | Source | Expected directory |
|---|---|---|
| Fox8-23 | [Zenodo 8035290](https://zenodo.org/records/8035290) | `data/fox8_23/` |
| BotSim-24 | [BotSim repository](https://github.com/QQQQQQBY/BotSim) | `data/botsim_24/` |
| MGTAB | [MGTAB repository](https://github.com/GraphDetec/MGTAB) | `data/mgtab/` (`edge_index.pt`, `features.pt`, `labels_bot.pt`, `edge_type.pt`) |

Fox8-23 and BotSim-24 are converted to account-level features:

```bash
python preprocess/fox8.py --ndjson <fox8_23_dataset.ndjson.gz> --out_dir data/fox8_23
python preprocess/botsim.py --csv <Users.csv> --json <user_post_comment.json> --out_dir data/botsim_24
```

## Usage

```bash
python train.py --dataset fox8   --data_dir data/fox8_23
python train.py --dataset botsim --data_dir data/botsim_24
python train.py --dataset mgtab  --data_dir data/mgtab
```

Options:

- `--auto_k` selects the number of sectors automatically: training starts from
  a truncation level `T` (default 16, from the Ishwaran–James bound), sectors
  are removed under the one-standard-error rule on validation data, and `T` is
  doubled if a channel keeps more than half of it.
- `--num_sectors K` sets the number of sectors (default 2).
- `--seeds 0 1 2` sets the random splits and initializations.
- `--degree_features` (MGTAB) appends per-relation log in/out-degrees.

For Fox8-23 and BotSim-24, the graph channel uses a cosine k-NN graph built
from account features (`--knn_k`, default 10); MGTAB uses its social graph.
Each seed draws a stratified split (70/15/15 for Fox8-23 and BotSim-24,
70/20/10 for MGTAB). Checkpoints and the decision threshold are selected on
the validation split, and the script reports test ACC, macro-F1, macro-PRE,
and macro-REC.

## License

MIT
