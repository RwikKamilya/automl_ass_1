# tests/test_successive_halving_viz.py
"""
Visualize Successive Halving on the LCDB dataset using the external surrogate.

Figures produced:
  1) Bar chart: population size per rung (shows halving)
  2) Line chart: best-so-far score vs evaluation calls
  3) "Spaghetti" survival plot: each line is a surviving config across rungs
"""

from __future__ import annotations
import argparse
import json
from typing import Dict, List, Tuple

import ConfigSpace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ConfigSpace.read_and_write import json as cs_json

from surrogate_model import SurrogateModel
from successive_halving import SuccessiveHalving


def cfg_key(cfg: Dict) -> str:
    """Stable identifier for a sampled configuration."""
    # Sort items for determinism; JSON ensures compact canonical-ish representation
    return json.dumps(sorted(cfg.items()), separators=(",", ":"))


def run_sh_with_tracking(
    sh: SuccessiveHalving,
    surrogate_model: SurrogateModel,
    keep_fraction: float | None = None,
) -> Tuple[List[int], List[int], List[List[str]], List[List[float]], List[float]]:
    """
    A thin wrapper around SH that *tracks* survivors per rung for visualization.

    Returns:
        anchors           : list of rung anchors
        pop_sizes         : list of population sizes per rung
        survivors_keys    : list (per rung) of config keys kept after selection
        scores_per_rung   : list (per rung) of scores (aligned with survivors order)
        best_trace        : best-so-far after each oracle call (for a step plot)
    """
    eta = sh.eta
    anchors = list(sh.anchors)
    rng = sh.rng

    # Choose initial population size n0 (classic: eta^(L-1))
    L = len(anchors)
    n0 = max(1, eta ** (L - 1))

    # Sample unique configurations
    # (We use the helper from the class if present; else sample via cs directly)
    if hasattr(sh, "_unique_sample"):
        configs = sh._unique_sample(n0)  # pylint: disable=protected-access
    else:
        seen, configs = set(), []
        while len(configs) < n0:
            cfg = dict(sh.cs.sample_configuration())
            k = cfg_key(cfg)
            if k not in seen:
                seen.add(k)
                configs.append(cfg)

    best_so_far = float("inf")
    best_trace: List[float] = []

    survivors_keys: List[List[str]] = []
    scores_per_rung: List[List[float]] = []
    pop_sizes: List[int] = []

    for rung_idx, anchor in enumerate(anchors):
        # 1) Evaluate all current configs at this rung's resource
        scored: List[Tuple[Dict, float, str]] = []
        for cfg in configs:
            perf = float(surrogate_model.predict({**cfg, "anchor_size": anchor}))
            best_so_far = min(best_so_far, perf)  # lower is better (invert if your score is higher=better)
            best_trace.append(best_so_far)
            scored.append((cfg, perf, cfg_key(cfg)))

        # 2) Select top 1/eta (or a custom keep_fraction if provided)
        scored.sort(key=lambda t: t[1])  # ascending = better
        if keep_fraction is None:
            k = max(1, int(np.ceil(len(scored) / eta)))
        else:
            k = max(1, int(np.ceil(len(scored) * float(keep_fraction))))

        kept = scored[:k]
        configs = [cfg for (cfg, _, _) in kept]
        survivors_keys.append([key for (_, _, key) in kept])
        scores_per_rung.append([perf for (_, perf, _) in kept])
        pop_sizes.append(len(configs))

        print(f"[SH rung {rung_idx+1}/{L}] anchor={anchor} kept={len(configs)} best={best_so_far:.6f}")

    return anchors, pop_sizes, survivors_keys, scores_per_rung, best_trace


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
    # step-like curve: best-so-far after each evaluation
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
    """
    Spaghetti survival plot:
      - x-axis: rung index (or anchor label)
      - y-axis: rank of each survivor at that rung (1 = best)
      Each line = the same configuration tracked across rungs it survives.
    """
    # For each rung: compute ranks by score (1 = best)
    ranks_per_rung: List[Dict[str, int]] = []
    for keys, scores in zip(survivors_keys, scores_per_rung):
        order = np.argsort(scores)  # ascending
        ranks = {keys[i]: int(j + 1) for j, i in enumerate(order)}
        ranks_per_rung.append(ranks)

    # Collect all keys that appear in any rung
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
            # show singletons too
            plt.plot(xs, ys, marker="o", alpha=0.6)

    plt.xticks(ticks=np.arange(len(anchors)), labels=[str(a) for a in anchors])
    plt.gca().invert_yaxis()  # rank 1 at the top
    plt.xlabel("Anchor (resource)")
    plt.ylabel("Rank at rung (1=best)")
    plt.title("Successive Halving: survival across rungs")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_space_file", type=str, required=True,
                    help="Path to lcdb_config_space_knn.json")
    ap.add_argument("--performances_csv", type=str, required=True,
                    help="Path to config-performances CSV (e.g., dataset-6)")
    ap.add_argument("--max_anchor_size", type=int, default=16000)
    ap.add_argument("--eta", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load ConfigSpace
    # with open(args.config_space_file, "r") as fh:
    #     cs = cs_json.read(fh)

    cs = ConfigSpace.ConfigurationSpace().from_json(args.config_space_file)

    # Load LCDB dataframe and fit surrogate on ALL anchors
    df = pd.read_csv(args.performances_csv)
    surrogate = SurrogateModel(cs)
    surrogate.fit(df)  # prints R^2/MSE too

    # Anchors present (filtered to <= max_anchor_size)
    anchors_all = sorted({int(a) for a in df["anchor_size"].unique().tolist()})
    anchors = [a for a in anchors_all if a <= args.max_anchor_size]
    if not anchors:
        raise ValueError("No anchors <= max_anchor_size found in the CSV.")

    # Run SH with tracking
    sh = SuccessiveHalving(cs, anchors=anchors, eta=args.eta, rng=np.random.default_rng(args.seed))
    anchors, pop_sizes, survivors_keys, scores_per_rung, best_trace = run_sh_with_tracking(sh, surrogate)

    # --- Visualizations ---
    plot_population_per_rung(anchors, pop_sizes, savepath=None)
    plot_best_trace(best_trace, savepath=None)
    plot_survival_spaghetti(anchors, survivors_keys, scores_per_rung, savepath=None)


if __name__ == "__main__":
    main()
