import numpy as np
from typing import List, Dict, Union

class StateBehaviorClassifier:
    def __init__(
        self,
        ratio_threshold: float = 0.8,
        gap_threshold: float = 0.05
    ):
        """
        ratio_threshold : minimum balance ratio between I_plus and I_minus
        gap_threshold   : maximum E_plus – E_minus for delocalization
        """
        self.ratio_threshold = ratio_threshold
        self.gap_threshold   = gap_threshold

    def classify_state(
        self,
        recs: List[Dict[str, float]]
    ) -> Dict[str, Union[str, float]]:
        # extract arrays
        dE = np.array([r["dE"] for r in recs], dtype=float)
        ov = np.array([r["ov"] for r in recs], dtype=float)
        total = ov.sum()

        # no overlap case → treat as zero shift
        if total <= 0:
            return {"mode": "shift", "mean_shift": 0.0}

        # overall weighted mean shift
        mean_shift = (ov * dE).sum() / total

        # branch weights
        w_plus  = ov[dE > 0].sum()
        w_minus = ov[dE < 0].sum()
        w_total = w_plus + w_minus

        # normalized overlap-weighted intensities
        if w_total > 0:
            I_plus  = w_plus  / w_total
            I_minus = w_minus / w_total
        else:
            I_plus, I_minus = 0.0, 0.0

        # branch average shifts
        E_plus = (ov[dE > 0] * dE[dE > 0]).sum() / w_plus if w_plus else 0.0
        E_minus = (ov[dE < 0] * dE[dE < 0]).sum() / w_minus if w_minus else 0.0

        # 1) pure shift if only one branch carries weight
        if I_plus == 0.0 or I_minus == 0.0:
            return {"mode": "shift", "mean_shift": mean_shift}

        # 2) delocalized if intensities are balanced and gap is small
        balance = min(I_plus, I_minus) / max(I_plus, I_minus)
        gap = E_plus - E_minus
        if balance >= self.ratio_threshold and gap <= self.gap_threshold:
            return {
                "mode": "delocalized",
                "deloc_factor": balance,
                "E_low":       E_minus,
                "E_high":      E_plus
            }

        # 3) split otherwise
        return {
            "mode":    "split",
            "E_plus":  E_plus,
            "I_plus":  I_plus,
            "E_minus": E_minus,
            "I_minus": I_minus
        }

    def classify_all(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]]
    ) -> Dict[int, Dict[str, Union[str, float]]]:
        results = {}
        for full_idx, comps in by_full.items():
            recs = comps.get("simple", [])
            results[full_idx] = self.classify_state(recs)
        return results