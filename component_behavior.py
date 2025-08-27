import numpy as np
from typing import List, Dict, Union

class StateBehaviorClassifier:
    """
    Classifies a component band's behavior across full-state matches into
    either 'shift' or 'split', and reports the total overlap weight.
    """

    def __init__(self):
        pass

    def classify_state(
        self,
        recs: List[Dict[str, float]]
    ) -> Dict[str, Union[str, float]]:
        """
        Parameters
        ----------
        recs : list of dicts
            Each dict must have keys:
              - "dE" : float  (energy shift = E_full - E_comp)
              - "ov" : float  (overlap weight)

        Returns
        -------
        result : dict
          - mode : "shift" or "split"
          - total_ov   : sum of overlap weights across all matches
          - If mode == "shift":
                mean_shift : weighted average shift (float)
          - If mode == "split":
                E_plus     : weighted average of positive shifts
                I_plus     : fraction of total overlap in positive shifts
                E_minus    : weighted average of negative shifts
                I_minus    : fraction of total overlap in negative shifts
        """
        # extract arrays
        dE       = np.array([r["dE"] for r in recs], dtype=float)
        ov       = np.array([r["ov"] for r in recs], dtype=float)
        total_ov = float(ov.sum())

        # no overlap => treat as zero shift
        if total_ov <= 0:
            return {
                "mode":       "shift",
                "mean_shift": 0.0,
                "total_ov":   0.0
            }

        # overall weighted mean shift
        mean_shift = (ov * dE).sum() / total_ov

        # branch overlap sums
        w_plus  = float(ov[dE > 0].sum())
        w_minus = float(ov[dE < 0].sum())
        w_total = w_plus + w_minus

        # normalized intensities (sum to 1 if w_total > 0)
        I_plus  = w_plus  / w_total if w_total > 0 else 0.0
        I_minus = w_minus / w_total if w_total > 0 else 0.0

        # branch-average shifts
        E_plus  = (ov[dE > 0] * dE[dE > 0]).sum()  / w_plus  if w_plus  > 0 else 0.0
        E_minus = (ov[dE < 0] * dE[dE < 0]).sum()  / w_minus if w_minus > 0 else 0.0

        # Pure shift if all weight in one branch
        if I_plus == 0.0 or I_minus == 0.0:
            return {
                "mode":       "shift",
                "mean_shift": mean_shift,
                "total_ov":   total_ov
            }

        # Otherwise it's a split
        return {
            "mode":       "split",
            "E_plus":     E_plus,
            "I_plus":     I_plus,
            "E_minus":    E_minus,
            "I_minus":    I_minus,
            "total_ov":   total_ov
        }

    def classify_all(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]]
    ) -> Dict[int, Dict[str, Union[str, float]]]:
        """
        Classify every component across all full-state indices.

        Parameters
        ----------
        by_full : dict
            Mapping full_idx -> {"simple": [...], "metal": [...]}
            Each list is passed to classify_state().

        Returns
        -------
        results : dict
            Mapping full_idx -> classification dict
        """
        results: Dict[int, Dict[str, Union[str, float]]] = {}
        for full_idx, comps in by_full.items():
            recs = comps.get("simple", [])
            results[full_idx] = self.classify_state(recs)
        return results
