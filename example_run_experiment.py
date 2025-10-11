import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=50)

    return parser.parse_args()


def run(args):
    import logging
    from smbo import SequentialModelBasedOptimization as SMBO

    # ---------- logging ----------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("experiment")

    # ---------- load inputs ----------
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    hp_names = list(config_space.keys())

    log.info(f"ConfigSpace JSON: {args.config_space_file}")
    log.info(f"Perf CSV: {args.configurations_performance_file}")
    log.info(f"Params: max_anchor_size={args.max_anchor_size} | num_iters={args.num_iterations}")

    # External surrogate (oracle) to score configs
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    # Helpers
    def perf_of(cfg_dict: dict) -> float:
        cfg = dict(cfg_dict)
        cfg["anchor_size"] = args.max_anchor_size
        return float(surrogate_model.predict(cfg))

    def only_hparams(row: pd.Series) -> dict:
        return {h: row[h] for h in hp_names if h in row.index}

    # For plotting
    results = {
        "RS_full": [float("inf")],
        "BO_full": [float("inf")],
        "RS_small": [float("inf")],
        "BO_small": [float("inf")],
    }

    # Convergence tracking (stopwatch-style; we still run the full budget but we log first-hit)
    def iter_to_converge(curve: list[float], tol: float = 1e-6, patience: int = 5) -> int | None:
        """Return the first iteration index where no improvement > tol for 'patience' steps (None if never)."""
        best = curve[0]
        since = 0
        for i, v in enumerate(curve[1:], start=1):
            if v < best - tol:
                best = v
                since = 0
            else:
                since += 1
                if since >= patience:
                    return i
        return None

    # =========================
    # Phase A: Full-init BO (pedagogical demo: quick convergence vs RS)
    # =========================
    smbo_full = SMBO(config_space)
    rs_full = RandomSearch(config_space)

    # Build capital_phi from ALL known configs (deduped by hyperparameters)
    # If your CSV contains multiple anchor sizes, we dedupe by HPs and query the oracle at the requested anchor.
    df_hp = df.loc[:, [c for c in df.columns if c in hp_names]].drop_duplicates(ignore_index=True)
    capital_phi = []
    for i, row in df_hp.iterrows():
        cfg = only_hparams(row)
        try:
            p = perf_of(cfg)
        except Exception as e:
            log.warning(f"[Full-init] Skipping row {i} due to oracle error: {e}")
            continue
        capital_phi.append((cfg, p))

    log.info(f"[Full-init] Using {len(capital_phi)} known (config, perf) pairs to initialize BO.")
    if len(capital_phi) == 0:
        log.error("[Full-init] No valid pairs found; aborting Phase A.")
    else:
        smbo_full.initialize(capital_phi)
        best_bo = min(p for _, p in capital_phi)
        results["BO_full"].append(best_bo)
        log.info(f"[Full-init] BO incumbent after init: {best_bo:.6f}")

        # Run very few epochs; BO should already be at/near optimum
        epochs_full = max(3, min(20, args.num_iterations // 10))
        log.info(f"[Full-init] Running {epochs_full} comparison epochs (RS vs BO).")

        best_rs = results["RS_full"][-1]
        for it in range(1, epochs_full + 1):
            # RS step
            cfg_rs = dict(rs_full.select_configuration())
            pr = perf_of(cfg_rs)
            best_rs = min(best_rs, pr)
            results["RS_full"].append(best_rs)
            rs_full.update_runs((cfg_rs, pr))
            log.info("[Full-init|RS ] iter=%3d | perf=%.6f | best=%.6f", it, pr, best_rs)

            # BO step
            cfg_bo = dict(smbo_full.select_configuration(n_configurations=512))
            pb = perf_of(cfg_bo)
            best_bo = min(best_bo, pb)
            results["BO_full"].append(best_bo)
            smbo_full.update_runs((cfg_bo, pb))
            log.info("[Full-init|BO ] iter=%3d | perf=%.6f | best=%.6f", it, pb, best_bo)

        tcv_rs = iter_to_converge(results["RS_full"])
        tcv_bo = iter_to_converge(results["BO_full"])
        log.info("[Full-init] First converge epoch: RS=%s | BO=%s", tcv_rs, tcv_bo)

    # =========================
    # Phase B: Small-init BO (realistic expensive-eval setting)
    # =========================
    smbo_small = SMBO(config_space)
    rs_small = RandomSearch(config_space)

    bo_n_init = 5  # small random warm-up (good default)
    n_candidates = 512  # robust EI screening default

    # Warm-up
    init_phi = []
    best_bo_small = float("inf")
    for i in range(1, bo_n_init + 1):
        cfg = dict(config_space.sample_configuration())
        p = perf_of(cfg)
        init_phi.append((cfg, p))
        best_bo_small = min(best_bo_small, p)
        results["BO_small"].append(best_bo_small)
        log.info("[Small-init|BO warmup] i=%3d | perf=%.6f | best=%.6f", i, p, best_bo_small)

    smbo_small.initialize(init_phi)
    log.info(f"[Small-init] BO seeded with {len(init_phi)} random points.")

    best_rs_small = results["RS_small"][-1]
    for it in range(1, args.num_iterations + 1):
        # RS step
        cfg_rs = dict(rs_small.select_configuration())
        pr = perf_of(cfg_rs)
        best_rs_small = min(best_rs_small, pr)
        results["RS_small"].append(best_rs_small)
        rs_small.update_runs((cfg_rs, pr))
        log.info("[Small-init|RS ] iter=%3d | perf=%.6f | best=%.6f", it, pr, best_rs_small)

        # BO step
        cfg_bo = dict(smbo_small.select_configuration(n_configurations=n_candidates))
        pb = perf_of(cfg_bo)
        best_bo_small = min(best_bo_small, pb)
        results["BO_small"].append(best_bo_small)
        smbo_small.update_runs((cfg_bo, pb))
        log.info("[Small-init|BO ] iter=%3d | perf=%.6f | best=%.6f", it, pb, best_bo_small)

    tcv_rs_s = iter_to_converge(results["RS_small"])
    tcv_bo_s = iter_to_converge(results["BO_small"])
    log.info("[Small-init] First converge epoch: RS=%s | BO=%s", tcv_rs_s, tcv_bo_s)

    # ---------- Plot ----------
    plt.plot(range(len(results["RS_full"])), results["RS_full"], label="RS (Full-init)")
    plt.plot(range(len(results["BO_full"])), results["BO_full"], label="BO (Full-init)")
    plt.plot(range(len(results["RS_small"])), results["RS_small"], label="RS (Small-init)")
    plt.plot(range(len(results["BO_small"])), results["BO_small"], label="BO (Small-init)")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best score so far")
    plt.title(f"RS vs BO — Full-init vs Small-init (anchor_size={args.max_anchor_size})")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(parse_args())
