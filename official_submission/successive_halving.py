from __future__ import annotations
import numpy as np


class SuccessiveHalving:

    def __init__(self, config_space, anchors, eta = 2):
        self.cs = config_space
        self.anchors = sorted(int(a) for a in anchors)
        self.eta = int(eta)

    def unique_sample(self, n):
        seen, out = set(), []
        while len(out) < n:
            config = dict(self.cs.sample_configuration())
            key = str(sorted(config.items()))
            if key not in seen:
                seen.add(key)
                out.append(config)
        return out

    def total_calls(self, n0, L):
        n = n0
        calls = 0
        for i in range(L):
            calls += n
            n = max(1, int(np.ceil(n / self.eta)))
        return calls

    def run(self, surrogate_model, n0 = None, max_evals = None):
        L = len(self.anchors)

        if n0 is None:
            n0 = max(1, self.eta ** (L - 1))

        if max_evals is not None and self.total_calls(n0, L) > max_evals:
            while n0 > 1 and self.total_calls(n0, L) > max_evals:
                n0 -= 1

        configs = self.unique_sample(n0)

        best_so_far = float("inf")
        trace= []

        for r_idx, anchor in enumerate(self.anchors):
            scored = []
            for config in configs:
                perf = float(surrogate_model.predict({**config, "anchor_size": anchor}))
                best_so_far = min(best_so_far, perf)
                trace.append(best_so_far)  # one entry per oracle call
                scored.append((config, perf))

            scored.sort(key=lambda kv: kv[1])
            k = max(1, int(np.ceil(len(scored) / self.eta)))
            configs = [cfg for cfg, _ in scored[:k]]
        return trace
