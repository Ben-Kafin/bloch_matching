import os
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np


class StateBehaviorClassifier:
    """
    Classifies a component band's behavior into shift vs split,
    reports E+/I+, E0/I0, E-/I-, mean_shift and variance.
    Bonding output first clamps any zero-crossing shifts.
    """

    def __init__(self):
        pass

    def classify_state(
        self,
        recs: List[Dict[str, float]]
    ) -> Dict[str, Union[str, float]]:
        """
        Classify a set of (dE, ov) records into up/zero/down branches,
        compute weighted mean shift & variance, and decide shift vs split
        based on exactly one nonzero branch.
        """

        dE       = np.array([r["dE"] for r in recs], dtype=float)
        ov       = np.array([r["ov"] for r in recs], dtype=float)
        total_ov = float(ov.sum())

        # no overlap → zero everything
        if total_ov <= 0:
            return {
                "mode":       "shift",
                "total_ov":   0.0,
                "mean_shift": 0.0,
                "variance":   0.0,
                "E_plus":     0.0, "I_plus":  0.0,
                "E_zero":     0.0, "I_zero":  0.0,
                "E_minus":    0.0, "I_minus": 0.0
            }

        # full-population weighted mean & variance
        mean_shift = float((ov * dE).sum() / total_ov)
        variance   = float(((ov * (dE - mean_shift)**2).sum()) / total_ov)

        # branch weights
        w_plus   = float(ov[dE >  0].sum())
        w_zero   = float(ov[dE == 0].sum())
        w_minus  = float(ov[dE <  0].sum())
        w_total  = w_plus + w_zero + w_minus

        # normalized intensities
        I_plus   = w_plus   / w_total if w_total > 0 else 0.0
        I_zero   = w_zero   / w_total if w_total > 0 else 0.0
        I_minus  = w_minus  / w_total if w_total > 0 else 0.0

        # branch-average shifts
        E_plus   = float((ov[dE >  0] * dE[dE >  0]).sum() / w_plus)   if w_plus  > 0 else 0.0
        E_zero   = 0.0
        E_minus  = float((ov[dE <  0] * dE[dE <  0]).sum() / w_minus)  if w_minus > 0 else 0.0

        # decide pure‐shift vs split
        branches_nonzero = sum(b > 0 for b in (w_plus, w_zero, w_minus))
        if branches_nonzero == 1:
            # exactly one branch: pure shift
            if w_plus > 0:
                out = {"E_plus": mean_shift, "I_plus": 1.0,
                       "E_zero": 0.0,       "I_zero": 0.0,
                       "E_minus": 0.0,      "I_minus": 0.0}
            elif w_minus > 0:
                out = {"E_plus": 0.0,       "I_plus": 0.0,
                       "E_zero": 0.0,       "I_zero": 0.0,
                       "E_minus": mean_shift, "I_minus": 1.0}
            else:
                # only zero‐shift has weight
                out = {"E_plus": 0.0,       "I_plus": 0.0,
                       "E_zero": 0.0,       "I_zero": 1.0,
                       "E_minus": 0.0,      "I_minus": 0.0}

            return {
                "mode":       "shift",
                "total_ov":   total_ov,
                "mean_shift": mean_shift,
                "variance":   variance,
                **out
            }

        # true split among multiple branches
        return {
            "mode":       "split",
            "total_ov":   total_ov,
            "mean_shift": mean_shift,
            "variance":   variance,
            "E_plus":     E_plus,   "I_plus":  I_plus,
            "E_zero":     E_zero,   "I_zero":  I_zero,
            "E_minus":    E_minus,  "I_minus": I_minus
        }

    def _make_bonding_records(
        self,
        recs: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Clamp any shift crossing zero so no electron sits above Fermi (0):
          - downward crossing (E>0→E+dE<0): dE_bond = E+dE
          - upward   crossing (E<0→E+dE>0): dE_bond = -E
          - else: keep original dE
        """
        bonded: List[Dict[str, float]] = []
        for r in recs:
            E_comp = r["E"]
            dE     = r["dE"]
            E_full = E_comp + dE

            if   E_comp >  0 and E_full <  0:
                dE_bond = E_full
            elif E_comp <  0 and E_full >  0:
                dE_bond = -E_comp
            else:
                dE_bond = dE

            bonded.append({"dE": dE_bond, "ov": r["ov"]})
        return bonded

    def classify_state_bonding(
        self,
        recs: List[Dict[str, float]]
    ) -> Dict[str, Union[str, float]]:
        """
        Apply zero-cross clamp, then reuse classify_state logic.
        """
        bond_recs = self._make_bonding_records(recs)
        return self.classify_state(bond_recs)

    def write_component_summaries(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]],
        simple_path: str,
        metal_path: str
    ) -> None:
        """
        Write four fixed-width tables:
          1) simple_path             (normal behavior)
          2) metal_path              (normal behavior)
          3) bonding_simple_<name>   (zero-cross truncated)
          4) bonding_metal_<name>    (zero-cross truncated)

        Columns:
          comp_idx  band_E    total_ov
          E_plus    I_plus    E_zero    I_zero    E_minus    I_minus
          mean_shift    variance
        """
        # 1) group by component index
        simple_groups = defaultdict(list)
        metal_groups  = defaultdict(list)
        for comps in by_full.values():
            for rec in comps.get("simple", []):
                simple_groups[rec["comp_idx"]].append(rec)
            for rec in comps.get("metal", []):
                metal_groups[rec["comp_idx"]].append(rec)

        # 2) inner writer
        def _write(
            groups: Dict[int, List[Dict[str, float]]],
            filename: str,
            bonding: bool
        ):
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w") as f:
                hdr_fmt = (
                    "{:>8s}  {:>9s}  {:>10s}  "
                    "{:>8s}  {:>8s}  {:>8s}  {:>8s}  "
                    "{:>8s}  {:>8s}  {:>12s}  {:>10s}\n"
                )
                row_fmt = (
                    "{comp_idx:8d}  {band_E:9.3f}  {total_ov:10.5f}  "
                    "{E_plus:8.3f}  {I_plus:8.3f}  "
                    "{E_zero:8.3f}  {I_zero:8.3f}  "
                    "{E_minus:8.3f}  {I_minus:8.3f}  "
                    "{mean_shift:12.5f}  {variance:10.5f}\n"
                )

                # header
                f.write(hdr_fmt.format(
                    "comp_idx", "band_E", "total_ov",
                    "E_plus",   "I_plus", "E_zero", "I_zero",
                    "E_minus",  "I_minus", "mean_shift", "variance"
                ))

                # rows
                for comp_idx in sorted(groups):
                    recs    = groups[comp_idx]
                    band_E  = float(recs[0]["E"])
                    total_ov = float(sum(r["ov"] for r in recs))
                    info    = (
                        self.classify_state_bonding(recs)
                        if bonding else
                        self.classify_state(recs)
                    )

                    f.write(row_fmt.format(
                        comp_idx    = comp_idx,
                        band_E      = band_E,
                        total_ov    = total_ov,
                        E_plus      = info["E_plus"],
                        I_plus      = info["I_plus"],
                        E_zero      = info["E_zero"],
                        I_zero      = info["I_zero"],
                        E_minus     = info["E_minus"],
                        I_minus     = info["I_minus"],
                        mean_shift  = info["mean_shift"],
                        variance    = info["variance"]
                    ))

        # normal behavior
        _write(simple_groups, simple_path, bonding=False)
        _write(metal_groups,  metal_path,  bonding=False)
        print(f"Written component behavior files:\n  {simple_path}\n  {metal_path}")

        # bonding behavior
        dir_s, base_s = os.path.dirname(simple_path), os.path.basename(simple_path)
        dir_m, base_m = os.path.dirname(metal_path),  os.path.basename(metal_path)
        bonding_simple = os.path.join(dir_s or ".", f"bonding_{base_s}")
        bonding_metal  = os.path.join(dir_m or ".", f"bonding_{base_m}")

        _write(simple_groups, bonding_simple, bonding=True)
        _write(metal_groups,  bonding_metal,  bonding=True)
        print(f"Written bonding behavior files:\n  {bonding_simple}\n  {bonding_metal}")
