# successive_halving.py
from __future__ import annotations
from typing import List, Dict
import numpy as np

class SuccessiveHalving:
    """
    Pure 'simulation' of SH using the external surrogate as the oracle.
    Minimization (lower is better).
    """

    def __init__(self, config_space, anchors: List[int], eta: int = 2, rng: np.random.Generator | None = None):
        self.cs = config_space
        self.anchors = sorted(int(a) for a in anchors)
        self.eta = int(eta)
        self.rng = rng or np.random.default_rng(0)

    def _unique_sample(self, n: int) -> List[Dict]:
        # Light uniqueness guard on dicts (hash by str) to reduce duplicates
        seen, out = set(), []
        while len(out) < n:
            cfg = dict(self.cs.sample_configuration())
            key = str(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                out.append(cfg)
        return out

    def run(self, surrogate_model, n0: int | None = None, max_evals: int | None = None, log=None) -> List[float]:
        """
        Returns a 'best-so-far' curve counted per oracle call (so it lines up with RS/SMBO by iterations).
        If max_evals is given, we back-solve the largest n0 feasible under the SH schedule.
        """
        L = len(self.anchors)

        def total_calls(n0_guess: int) -> int:
            n = n0_guess
            calls = 0
            for _ in range(L):
                calls += n
                n = max(1, int(np.ceil(n / self.eta)))
            return calls

        if n0 is None:
            # Pick a healthy n0: eta^(L-1) is the classical choice
            n0 = max(1, self.eta ** (L - 1))

        if max_evals is not None and total_calls(n0) > max_evals:
            # Shrink n0 until schedule fits under the budget
            while n0 > 1 and total_calls(n0) > max_evals:
                n0 -= 1

        # --- SH core ---
        configs = self._unique_sample(n0)

        best_so_far = float("inf")
        trace: List[float] = []

        for r_idx, anchor in enumerate(self.anchors):
            # 1) Evaluate current set at this anchor
            scored = []
            for cfg in configs:
                perf = float(surrogate_model.predict({**cfg, "anchor_size": anchor}))
                best_so_far = min(best_so_far, perf)
                trace.append(best_so_far)  # one entry per oracle call
                scored.append((cfg, perf))

            # 2) Keep the top 1/eta to advance (ceil to avoid dropping to 0)
            scored.sort(key=lambda kv: kv[1])  # lower = better
            k = max(1, int(np.ceil(len(scored) / self.eta)))
            configs = [cfg for cfg, _ in scored[:k]]

            if log is not None:
                log.info("[SH ] rung=%d/%d | anchor=%s | kept=%d | best=%.6f",
                         r_idx + 1, L, anchor, len(configs), best_so_far)

        return trace
