import argparse
import os
import random
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ConfigSpace as CS
from scipy.stats import spearmanr

from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization
from successive_halving import SuccessiveHalving

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Skipping features without any observed values", category=UserWarning)


class RunExperiment:
    def __init__(self, config_space_file, performance_paths, n_init=5, n_evaluations=200, successive_halving_eta=2,
                 successive_halving_n0=None, expected_improvement_pool=512, seed=42,
                 output_directory="report_artifacts", show_plots=False):
        self.config_space_file = config_space_file
        self.performance_paths = performance_paths
        self.n_init = int(n_init)
        self.n_evaluations = int(n_evaluations)
        self.successive_halving_eta = int(successive_halving_eta)
        self.successive_halving_n0 = successive_halving_n0
        self.expected_improvement_pool = int(expected_improvement_pool)
        self.seed = int(seed)
        self.output_directory = output_directory
        self.show_plots = bool(show_plots)
        self.set_all_seeds(self.seed)
        self.random_generator = np.random.default_rng(self.seed)
        self.config_space = CS.ConfigurationSpace.from_json(self.config_space_file)
        self.config_space.seed(self.seed)

        os.makedirs(self.output_directory, exist_ok=True)

    @staticmethod
    def set_all_seeds(seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def run(self):
        csvs = []
        for pattern in self.performance_paths:
            csvs.extend(glob.glob(pattern, recursive=True))
        csv_files = sorted(set(csvs))

        if not csv_files:
            raise FileNotFoundError(f"No CSVs found for: {self.performance_paths}")

        summary_rows = []
        spearman_map = {}

        for csv_path in csv_files:
            dataset_label = os.path.splitext(os.path.basename(csv_path))[0]
            print(f"\n=== Dataset: {dataset_label} ===")
            data_frame = pd.read_csv(csv_path)

            surrogate_model = SurrogateModel(self.config_space, seed=self.seed)
            surrogate_model.fit(data_frame)

            anchors_sorted = sorted(data_frame["anchor_size"].unique().tolist())
            final_anchor = anchors_sorted[-1]

            rho, parity_df = self.compute_spearman_at_final(data_frame, surrogate_model, final_anchor)
            spearman_map[dataset_label] = rho
            print(f"[External surrogate] anchors={anchors_sorted[:4]}..{anchors_sorted[-4:]}, "
                  f"final_anchor={final_anchor}, Spearman@final≈{rho:.3f}")
            self.plot_parity_scatter(dataset_label, parity_df)

            n_evaluations = self.n_evaluations
            sh_n0 = self.successive_halving_n0 if self.successive_halving_n0 is not None \
                else int(n_evaluations * (self.successive_halving_eta - 1) / self.successive_halving_eta)

            rs_xs, rs_ys, rs_best_cfg = self.run_random_search(
                self.config_space, surrogate_model, final_anchor, n_evaluations
            )

            ei_xs, ei_ys, ei_best_cfg = self.run_expected_improvement(
                self.config_space, surrogate_model, final_anchor,
                self.n_init, n_evaluations, self.expected_improvement_pool
            )

            sh_xs, sh_ys, sh_best_cfg = self.run_successive_halving_final_metric(
                self.config_space, surrogate_model, anchors_sorted, sh_n0, self.successive_halving_eta,
                n_evaluations
            )

            curves = {
                "Random Search": (rs_xs, rs_ys),
                "SMBO (Expected Improvement)": (ei_xs, ei_ys),
                "Successive Halving": (sh_xs, sh_ys),
            }
            note = f"Budget={n_evaluations}, SH n0≈{sh_n0}, η={self.successive_halving_eta}"

            # Plot log_y
            self.plot_convergence(dataset_label, curves, note=note, logy=False)

            rs_area, rs_t95, rs_best = self.digest(rs_xs, rs_ys, n_evaluations)
            ei_area, ei_t95, ei_best = self.digest(ei_xs, ei_ys, n_evaluations)
            sh_area, sh_t95, sh_best = self.digest(sh_xs, sh_ys, n_evaluations)

            table_df = pd.DataFrame([
                {"method": "Random Search", "AUBC (lower=better)": rs_area,
                 "iters@95%": rs_t95, "best_final_score": rs_best,
                 "best_config": self.truncate_config(rs_best_cfg)},
                {"method": "SMBO (Expected Improvement)", "AUBC (lower=better)": ei_area,
                 "iters@95%": ei_t95, "best_final_score": ei_best,
                 "best_config": self.truncate_config(ei_best_cfg)},
                {"method": "Successive Halving", "AUBC (lower=better)": sh_area,
                 "iters@95%": sh_t95, "best_final_score": sh_best,
                 "best_config": self.truncate_config(sh_best_cfg)},
            ])
            table_path = os.path.join(self.output_directory, f"summary_{dataset_label}.csv")
            table_df.to_csv(table_path, index=False)
            print(f"Saved: {table_path}")

            for row in table_df.to_dict(orient="records"):
                summary_rows.append({
                    "dataset": dataset_label,
                    "method": row["method"],
                    "AUBC": row["AUBC (lower=better)"],
                    "iters@95%": row["iters@95%"],
                    "best_final_score": row["best_final_score"]
                })

        self.plot_spearman_bar(spearman_map)

        summary_all = pd.DataFrame(summary_rows)
        summary_all_path = os.path.join(self.output_directory, "summary_all_datasets.csv")
        summary_all.to_csv(summary_all_path, index=False)
        print(f"Saved: {summary_all_path}")

    def digest(self, xs, ys, n_evaluations):
        area = self.area_under_best_so_far(xs, ys, n_evaluations)
        iters95 = self.iterations_to_fraction_of_best(xs, ys, 0.95)
        best = ys[-1] if ys else float("inf")
        return area, iters95, best

    def sample_unique_configurations(self, config_space: CS.ConfigurationSpace, n: int):
        configurations, seen = [], set()
        while len(configurations) < n:
            cfg = dict(config_space.sample_configuration())
            key = str(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                configurations.append(cfg)
        return configurations

    @staticmethod
    def area_under_best_so_far(xs, ys, budget):
        if len(xs) == 0:
            return float("inf")
        dense = np.full(budget, np.nan, dtype=float)
        last_best = np.inf
        for x, y in zip(xs, ys):
            last_best = min(last_best, y)
            idx = min(x, budget) - 1
            dense[idx] = last_best
        for i in range(budget):
            if np.isnan(dense[i]):
                dense[i] = dense[i - 1] if i > 0 else last_best
        return float(np.trapezoid(dense, dx=1.0))

    @staticmethod
    def iterations_to_fraction_of_best(xs, ys, fraction=0.95):
        if not ys:
            return 0
        best = ys[-1]
        target = best * fraction
        for x, y in zip(xs, ys):
            if y <= target:
                return x
        return xs[-1]

    @staticmethod
    def truncate_config(configuration, max_items=6):
        items = list(configuration.items())
        return dict(items[:max_items])

    @staticmethod
    def compute_spearman_at_final(data_frame, surrogate_model, final_anchor):
        df_final = data_frame[data_frame["anchor_size"] == final_anchor].copy()
        if len(df_final) < 3:
            return float("nan"), pd.DataFrame(columns=["y_true", "y_pred"])
        y_true = df_final["score"].to_numpy()
        x_dicts = df_final.drop(columns=["score"]).to_dict(orient="records")
        y_pred = np.array([surrogate_model.predict(rec) for rec in x_dicts], dtype=float)
        rho = spearmanr(y_true, y_pred)[0]
        return float(rho), pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    def run_random_search(self, config_space, surrogate_model,
                          final_anchor, n_evaluations):
        xs, ys = [], []
        best_value = float("inf")
        best_configuration = {}
        for i in range(1, n_evaluations + 1):
            configuration = self.sample_unique_configurations(config_space, 1)[0]
            value = surrogate_model.predict({**configuration, "anchor_size": final_anchor})
            if value < best_value:
                best_value = value
                best_configuration = configuration
            xs.append(i)
            ys.append(best_value)
        return xs, ys, best_configuration

    def run_expected_improvement(self, config_space, surrogate_model,
                                 final_anchor, n_init, n_evaluations,
                                 pool_size):
        init_configurations = self.sample_unique_configurations(config_space, n_init)
        initial_runs = []
        best_value = float("inf")
        best_configuration = {}
        xs, ys = [], []

        for i, configuration in enumerate(init_configurations, 1):
            performance = surrogate_model.predict({**configuration, "anchor_size": final_anchor})
            initial_runs.append((configuration, performance))
            if performance < best_value:
                best_value, best_configuration = performance, configuration
            xs.append(i)
            ys.append(best_value)

        smbo = SequentialModelBasedOptimization(config_space, seed=self.seed)
        smbo.initialize(initial_runs)

        for t in range(n_evaluations - n_init):
            theta_new = smbo.select_configuration(n_configurations=pool_size)
            configuration = dict(theta_new)
            performance = surrogate_model.predict({**configuration, "anchor_size": final_anchor})
            smbo.update_runs((configuration, performance))
            if performance < best_value:
                best_value, best_configuration = performance, configuration
            xs.append(n_init + t + 1)
            ys.append(best_value)

        return xs, ys, best_configuration

    # def run_successive_halving_final_metric(self, config_space, surrogate_model, anchors_sorted, n0,
    #                                         eta, max_evaluations):
    #     final_anchor = anchors_sorted[-1]
    #
    #     while n0 > 1 and self.total_calls(n0, anchors_sorted, eta) > max_evaluations:
    #         n0 -= 1
    #
    #     survivors = self.sample_unique_configurations(config_space, n0)
    #     xs, ys = [], []
    #     best_final_value = float("inf")
    #     best_configuration = {}
    #     calls = 0
    #
    #     for anchor in anchors_sorted:
    #         scored = []
    #         for configuration in survivors:
    #             current_value = surrogate_model.predict({**configuration, "anchor_size": anchor})
    #             calls += 1
    #             final_value = surrogate_model.predict({**configuration, "anchor_size": final_anchor})
    #             if final_value < best_final_value:
    #                 best_final_value, best_configuration = final_value, configuration
    #             xs.append(calls)
    #             ys.append(best_final_value)
    #             scored.append((configuration, current_value))
    #
    #         scored.sort(key=lambda kv: (kv[1], tuple(sorted(kv[0].items()))))
    #         k = max(1, int(np.ceil(len(survivors) / eta)))
    #         survivors = [cfg for cfg, _ in scored[:k]]
    #
    #         if calls >= max_evaluations:
    #             break
    #
    #     return xs, ys, best_configuration

    def run_successive_halving_final_metric(self, config_space, surrogate_model, anchors_sorted, n0,
                                            eta, max_evaluations):
        sh = SuccessiveHalving(config_space=config_space,
                               anchors=anchors_sorted,
                               eta=eta)
        L = len(sh.anchors)
        while n0 > 1 and sh.total_calls(n0, L) > max_evaluations:
            n0 -= 1
        survivors = sh.unique_sample(n0)
        final_anchor = sh.anchors[-1]
        xs, ys = [], []
        best_final_value = float("inf")
        best_configuration = {}
        calls = 0

        for anchor in sh.anchors:
            scored = []
            for configuration in survivors:
                current_value = float(surrogate_model.predict({**configuration, "anchor_size": anchor}))
                calls += 1
                final_value = float(surrogate_model.predict({**configuration, "anchor_size": final_anchor}))
                if final_value < best_final_value:
                    best_final_value, best_configuration = final_value, configuration
                xs.append(calls)
                ys.append(best_final_value)
                scored.append((configuration, current_value))

            # Keep top 1/eta by current-anchor performance (lower is better)
            # Deterministic tie-break with config content.
            scored.sort(key=lambda kv: (kv[1], tuple(sorted(kv[0].items()))))
            k = max(1, int(np.ceil(len(survivors) / sh.eta)))
            survivors = [cfg for cfg, _ in scored[:k]]

            if calls >= max_evaluations:
                break

        return xs, ys, best_configuration

    def total_calls(self, n0_guess, anchors_sorted, eta):
        n, calls = n0_guess, 0
        for _ in anchors_sorted:
            calls += n
            n = max(1, int(np.ceil(n / eta)))
        return calls

    def plot_parity_scatter(self, label, data_frame_parity):
        plt.figure(figsize=(4.8, 4.8))
        x_vals, y_vals = data_frame_parity["y_true"].values, data_frame_parity["y_pred"].values
        limits = [min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max())]
        plt.plot(limits, limits, linestyle="--", linewidth=1)
        plt.scatter(x_vals, y_vals, s=12)
        plt.xlabel("True score @ final anchor")
        plt.ylabel("Predicted score @ final anchor")
        plt.title(f"Parity — {label}")
        plt.grid(True, alpha=0.3)
        out = os.path.join(self.output_directory, f"parity_{label}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        if self.show_plots:
            plt.show()
        plt.close()
        print(f"Saved: {out}")

    def plot_convergence(self, label, curves, note="", logy=False):
        plt.figure(figsize=(7.5, 5.0))
        for name, (xs, ys) in curves.items():
            if logy:
                plt.semilogy(xs, ys, linewidth=2, label=name)
            else:
                plt.plot(xs, ys, linewidth=2, label=name)
        plt.xlabel("# surrogate evaluations")
        plt.ylabel("Best predicted score @ final anchor (lower is better)")
        title = f"Convergence — {label}"
        if note: title += f"\n{note}"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = os.path.join(self.output_directory, f"convergence_{label}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        if self.show_plots: plt.show()
        plt.close()
        print(f"Saved: {out}")

    def plot_spearman_bar(self, rho_by_dataset):
        if not rho_by_dataset:
            return
        labels = list(rho_by_dataset.keys())
        values = [rho_by_dataset[k] if rho_by_dataset[k] == rho_by_dataset[k] else 0.0 for k in labels]
        plt.figure(figsize=(max(6.0, 0.4 * len(labels)), 4.0))
        plt.bar(labels, values)
        plt.ylabel("Spearman ρ @ final anchor")
        plt.title("External surrogate quality across datasets")
        plt.ylim(0.0, 1.05)
        plt.xticks(rotation=30, ha="right")
        out = os.path.join(self.output_directory, "surrogate_quality_bar.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        if self.show_plots:
            plt.show()
        plt.close()
        print(f"Saved: {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_space_file", type=str, default="lcdb_config_space_knn.json")
    ap.add_argument("--performances", nargs="+",
                    default=["config-performances/config_performances_dataset-*.csv"],
                    help="One or more CSV paths or glob patterns.")
    ap.add_argument("--n_init", type=int, default=5, help="Initial random design size for EI.")
    ap.add_argument("--n_evals", type=int, default=100, help="Total surrogate evals for RS/EI (and cap for SH).")
    ap.add_argument("--sh_eta", type=int, default=2, help="Halving rate for SH.")
    ap.add_argument("--sh_n0", type=int, default=32,
                    help="Initial pool for SH. If not set, uses ~ n_evals*(eta-1)/eta to match budget.")
    ap.add_argument("--ei_pool", type=int, default=512, help="Candidate pool for EI at each step.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="report_artifacts")
    ap.add_argument("--show", action="store_true", help="Show plots interactively.")
    return ap.parse_args()


def main():
    args = parse_args()
    runner = RunExperiment(
        config_space_file=args.config_space_file,
        performance_paths=args.performances,
        n_init=args.n_init,
        n_evaluations=args.n_evals,
        successive_halving_eta=args.sh_eta,
        successive_halving_n0=args.sh_n0,
        expected_improvement_pool=args.ei_pool,
        seed=args.seed,
        output_directory=args.outdir,
        show_plots=args.show,
    )
    runner.run()


if __name__ == "__main__":
    main()
