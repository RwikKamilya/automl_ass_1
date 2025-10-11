# test_successive_halving_viz_topn.py
"""
Visualize Successive Halving on a *subset* of anchors: take the TOP-N (largest) anchors from the CSV.
This keeps the run compact and plots clean.

Run:
python test_successive_halving_viz_topn.py \
  --config_space_file lcdb_config_space_knn.json \
  --performances_csv config-performances/config_performances_dataset-6.csv \
  --max_anchor_size 16000 \
  --top-n-anchors 7 \
  --eta 2 \
  --log-level INFO \
  --log-every 1 \
  --max-evals 1023
"""

from __future__ import annotations
import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ConfigSpace as CS

from surrogate_model import SurrogateModel


# -------------------- Logging --------------------
def setup_logger(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("SH")


# -------------------- Keying (robust) --------------------
def _normalize_value(v):
    import numpy as _np
    if isinstance(v, (_np.integer,)):
        return int(v)
    if isinstance(v, (_np.floating,)):
        return float(v)
    if isinstance(v, (_np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return tuple(_normalize_value(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((str(k), _normalize_value(val)) for k, val in v.items()))
    return v

def cfg_key(cfg: Dict) -> str:
    norm_items = tuple(sorted((str(k), _normalize_value(v)) for k, v in cfg.items()))
    return repr(norm_items)


# -------------------- Schedule helpers --------------------
def total_calls_for(n0: int, anchors: List[int], eta: int) -> int:
    n, calls = n0, 0
    for _ in anchors:
        calls += n
        n = max(1, int(np.ceil(n / eta)))
    return calls

def choose_n0(anchors: List[int], eta: int, n0: int | None, max_evals: int | None) -> int:
    L = len(anchors)
    n0_eff = max(1, eta ** (L - 1)) if (n0 is None or n0 <= 0) else int(n0)
    if max_evals is None:
        return n0_eff
    # Shrink until the schedule fits
    while n0_eff > 1 and total_calls_for(n0_eff, anchors, eta) > max_evals:
        n0_eff = max(1, int(np.ceil(n0_eff / eta)))
    return n0_eff

def compute_schedule(anchors: List[int], eta: int, n0: int) -> Tuple[List[int], int, int]:
    pops, total_evals, cum_resource = [], 0, 0
    n = n0
    for a in anchors:
        pops.append(n)
        total_evals += n
        cum_resource += n * int(a)
        n = max(1, int(np.ceil(n / eta)))
    return pops, total_evals, cum_resource


# -------------------- SH runner (with tracking + logging) --------------------
def run_sh_with_tracking(
    cs: CS.ConfigurationSpace,
    surrogate_model: SurrogateModel,
    anchors: List[int],
    eta: int,
    seed: int,
    n0: int,
    log_every: int,
    logger: logging.Logger,
) -> Tuple[List[int], List[int], List[List[str]], List[List[float]], List[float]]:
    rng = np.random.default_rng(seed)
    anchors = list(sorted(int(a) for a in anchors))
    L = len(anchors)

    pops, total_evals, cum_resource = compute_schedule(anchors, eta, n0)
    logger.info(
        "[SH] start | L=%d | eta=%d | anchors=%s | n0=%d | total_evals=%d | cum_resource=%d",
        L, eta, anchors, n0, total_evals, cum_resource
    )
    if total_evals > 100_000:
        logger.warning(
            "[SH] very large schedule detected (total_evals=%d). "
            "Consider reducing --top-n-anchors or using --max-evals.",
            total_evals
        )

    # Sample initial population
    seen, configs = set(), []
    while len(configs) < n0:
        cfg = dict(cs.sample_configuration())
        k = cfg_key(cfg)
        if k not in seen:
            seen.add(k)
            configs.append(cfg)

    best_so_far = float("inf")
    best_trace: List[float] = []
    survivors_keys: List[List[str]] = []
    scores_per_rung: List[List[float]] = []
    eval_ctr = 0

    for rung_idx, (anchor, pop_target) in enumerate(zip(anchors, pops), start=1):
        logger.info("[SH] rung %d/%d | anchor=%s | pop=%d | progress=%d/%d",
                    rung_idx, L, anchor, len(configs), eval_ctr, total_evals)

        # 1) Evaluate all configs at this rung
        scored: List[Tuple[Dict, float, str]] = []
        for cfg in configs:
            perf = float(surrogate_model.predict({**cfg, "anchor_size": anchor}))
            best_so_far = min(best_so_far, perf)  # lower is better
            best_trace.append(best_so_far)
            scored.append((cfg, perf, cfg_key(cfg)))

            eval_ctr += 1
            if eval_ctr <= 3 or eval_ctr == total_evals or eval_ctr % max(1, log_every) == 0:
                logger.info("[SH] eval %d/%d | anchor=%s | rung=%d/%d | best=%.6f",
                            eval_ctr, total_evals, anchor, rung_idx, L, best_so_far)

        # 2) Select top fraction 1/eta
        scored.sort(key=lambda t: t[1])  # ascending = better
        k = max(1, int(np.ceil(len(scored) / eta)))
        kept = scored[:k]
        configs = [cfg for (cfg, _, _) in kept]
        survivors_keys.append([key for (_, _, key) in kept])
        scores_per_rung.append([perf for (_, perf, _) in kept])

        logger.info("[SH] select | anchor=%s | kept=%d/%d (eta=%d) | best=%.6f",
                    anchor, len(configs), len(scored), eta, best_so_far)

    logger.info("[SH] done  | total_evals=%d | best=%.6f", eval_ctr, best_so_far)

    # For the bar chart use the planned pops (cleaner than recomputing from survivors)
    pop_sizes = pops
    return anchors, pop_sizes, survivors_keys, scores_per_rung, best_trace


# -------------------- Plotting --------------------
def plot_population_per_rung(anchors: List[int], pop_sizes: List[int], savepath: str | None = None):
    plt.figure(figsize=(6, 3.5))
    x = np.arange(len(anchors))
    plt.bar(x, pop_sizes)
    plt.xticks(x, anchors, rotation=0)
    plt.xlabel("Anchor (resource)")
    plt.ylabel("# Candidates")
    plt.title("Successive Halving: population size per rung")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

def plot_best_trace(best_trace: List[float], savepath: str | None = None):
    plt.figure(figsize=(6, 3.5))
    plt.plot(np.arange(1, len(best_trace) + 1), best_trace, drawstyle="steps-post")
    plt.xlabel("# Oracle calls (evaluations)")
    plt.ylabel("Best-so-far (lower is better)")
    plt.title("Best-so-far vs evaluations (SH)")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

def plot_survival_spaghetti(
    anchors: List[int],
    survivors_keys: List[List[str]],
    scores_per_rung: List[List[float]],
    savepath: str | None = None,
):
    # Compute per-rung ranks (1 = best)
    ranks_per_rung: List[Dict[str, int]] = []
    for keys, scores in zip(survivors_keys, scores_per_rung):
        order = np.argsort(scores)  # ascending
        ranks = {keys[i]: int(j + 1) for j, i in enumerate(order)}
        ranks_per_rung.append(ranks)

    all_keys = set(k for keys in survivors_keys for k in keys)

    plt.figure(figsize=(7.5, 4))
    for k in all_keys:
        xs, ys = [], []
        for r, ranks in enumerate(ranks_per_rung):
            if k in ranks:
                xs.append(r)
                ys.append(ranks[k])
        if len(xs) >= 2:
            plt.plot(xs, ys, alpha=0.6)
        elif len(xs) == 1:
            plt.plot(xs, ys, marker="o", alpha=0.6)

    plt.xticks(ticks=np.arange(len(anchors)), labels=[str(a) for a in anchors])
    plt.gca().invert_yaxis()
    plt.xlabel("Rung (anchor)")
    plt.ylabel("Rank at rung (1=best)")
    plt.title("Successive Halving: survival across rungs")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_space_file", type=str, required=True)
    ap.add_argument("--performances_csv", type=str, required=True)
    ap.add_argument("--max_anchor_size", type=int, default=16000)
    ap.add_argument("--top-n-anchors", type=int, default=7,
                    help="Use only the largest N anchors (good for compact viz).")
    ap.add_argument("--eta", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n0", type=int, default=0, help="Initial population; 0 = auto (eta^(L-1))")
    ap.add_argument("--max-evals", type=int, default=0,
                    help="Optional hard cap on total evaluations; 0 = no cap")
    ap.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log-every", type=int, default=1, help="Log each Nth evaluation (>=1)")
    args = ap.parse_args()

    logger = setup_logger(args.log_level)

    # Load ConfigSpace with new API, fallback if needed
    try:
        cs = CS.ConfigurationSpace().from_json(args.config_space_file)
    except Exception:
        from ConfigSpace.read_and_write import json as cs_json
        with open(args.config_space_file, "r") as fh:
            cs = cs_json.read(fh.read())

    # Load data and fit surrogate on ALL anchors
    df = pd.read_csv(args.performances_csv)
    surrogate = SurrogateModel(cs)
    surrogate.fit(df)  # prints quick diagnostics (R^2/MSE)

    # Build anchors and trim to top-N (largest N) under max_anchor_size
    anchors_all = sorted({int(a) for a in df["anchor_size"].unique().tolist()})
    anchors_all = [a for a in anchors_all if a <= args.max_anchor_size]
    if not anchors_all:
        raise ValueError("No anchors <= max_anchor_size found in the CSV.")

    N = min(args.top_n_anchors, len(anchors_all))
    anchors = anchors_all[-N:]  # take largest N
    logger.info("[SH] using top-N anchors: N=%d | anchors=%s", N, anchors)

    # Choose n0 (auto or provided), optionally shrink to fit max_evals
    cap = (None if args.max_evals <= 0 else args.max_evals)
    n0_eff = choose_n0(anchors, args.eta, n0=(None if args.n0 == 0 else args.n0), max_evals=cap)
    logger.info("[SH] chosen n0=%d (eta=%d, L=%d)", n0_eff, args.eta, len(anchors))

    # Run SH
    anchors, pop_sizes, survivors_keys, scores_per_rung, best_trace = run_sh_with_tracking(
        cs=cs,
        surrogate_model=surrogate,
        anchors=anchors,
        eta=args.eta,
        seed=args.seed,
        n0=n0_eff,
        log_every=max(1, args.log_every),
        logger=logger,
    )

    # Plots
    plot_population_per_rung(anchors, pop_sizes)
    plot_best_trace(best_trace)
    plot_survival_spaghetti(anchors, survivors_keys, scores_per_rung)


if __name__ == "__main__":
    main()
