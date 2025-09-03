# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 09:34:48 2025

@author: Benjamin Kafin
"""

import os
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np


class StateBehaviorClassifier:
    """
    Classifies a component band's behavior across full-state matches into
    either 'shift' or 'split', and reports the total overlap weight,
    the weighted mean shift, and the weighted variance of the shifts.
    """

    def __init__(self):
        # No thresholds needed: we only distinguish shift vs. split
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
          - mode       : "shift" or "split"
          - total_ov   : sum of overlap weights across all matches
          - mean_shift : weighted average shift
          - variance   : weighted population variance of shifts
          - If mode == "split":
                E_plus  : weighted average of positive shifts
                I_plus  : fraction of total overlap in positive shifts
                E_minus : weighted average of negative shifts
                I_minus : fraction of total overlap in negative shifts
        """
        dE       = np.array([r["dE"] for r in recs], dtype=float)
        ov       = np.array([r["ov"] for r in recs], dtype=float)
        total_ov = float(ov.sum())

        # no overlap → zero shift, zero variance
        if total_ov <= 0:
            return {
                "mode":       "shift",
                "mean_shift": 0.0,
                "variance":   0.0,
                "total_ov":   0.0
            }

        # weighted mean shift
        mean_shift = float((ov * dE).sum() / total_ov)

        # weighted variance
        var = float(((ov * (dE - mean_shift)**2).sum()) / total_ov)

        # branch overlap sums
        w_plus  = float(ov[dE > 0].sum())
        w_minus = float(ov[dE < 0].sum())
        w_total = w_plus + w_minus

        I_plus  = w_plus  / w_total if w_total > 0 else 0.0
        I_minus = w_minus / w_total if w_total > 0 else 0.0

        E_plus  = float((ov[dE > 0] * dE[dE > 0]).sum()  / w_plus)  if w_plus  > 0 else 0.0
        E_minus = float((ov[dE < 0] * dE[dE < 0]).sum()  / w_minus) if w_minus > 0 else 0.0

        # pure shift if only one branch carries weight
        if I_plus == 0.0 or I_minus == 0.0:
            return {
                "mode":       "shift",
                "mean_shift": mean_shift,
                "variance":   var,
                "total_ov":   total_ov
            }

        # split otherwise
        return {
            "mode":       "split",
            "mean_shift": mean_shift,
            "variance":   var,
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
        """
        results: Dict[int, Dict[str, Union[str, float]]] = {}
        for full_idx, comps in by_full.items():
            recs = comps.get("simple", [])
            results[full_idx] = self.classify_state(recs)
        return results

    def write_component_summaries(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]],
        simple_path: str,
        metal_path: str
    ) -> None:
        """
        Write two summary tables (simple_path & metal_path) with columns:
          comp_idx, band_E, total_ov,
          E_plus, I_plus, E_minus, I_minus,
          mean_shift, variance

        Then automatically write the zero-cross "bonding" versions
        as bonding_<simple_path> and bonding_<metal_path>.
        """
        simple_groups = defaultdict(list)
        metal_groups  = defaultdict(list)
        for comps in by_full.values():
            for rec in comps.get("simple", []):
                simple_groups[rec["comp_idx"]].append(rec)
            for rec in comps.get("metal", []):
                metal_groups[rec["comp_idx"]].append(rec)

        def _write(groups: Dict[int, List[Dict[str, float]]], filename: str):
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w") as f:
                f.write(
                    "# comp_idx  band_E    total_ov    "
                    "E_plus    I_plus    E_minus    I_minus    "
                    "mean_shift    variance\n"
                )
                for comp_idx in sorted(groups):
                    recs     = groups[comp_idx]
                    band_E   = float(recs[0]["E"])
                    dE_arr   = np.array([r["dE"] for r in recs], dtype=float)
                    ov_arr   = np.array([r["ov"] for r in recs], dtype=float)
                    total_ov = float(ov_arr.sum())

                    # run the classifier
                    info = self.classify_state(
                        [{"dE": d, "ov": w} for d, w in zip(dE_arr, ov_arr)]
                    )

                    mean_shift = float(info.get("mean_shift", 0.0))
                    variance   = float(info.get("variance",    0.0))

                    # assign E_plus/I_plus and E_minus/I_minus
                    if info["mode"] == "shift":
                        # pure shift → put shift in the correct branch
                        if mean_shift >= 0:
                            E_plus, I_plus  = mean_shift, 1.0
                            E_minus, I_minus = 0.0,       0.0
                        else:
                            E_plus, I_plus  = 0.0,       0.0
                            E_minus, I_minus = mean_shift, 1.0
                    else:
                        E_plus  = float(info.get("E_plus",  0.0))
                        I_plus  = float(info.get("I_plus",  0.0))
                        E_minus = float(info.get("E_minus", 0.0))
                        I_minus = float(info.get("I_minus", 0.0))

                    # write the row
                    f.write(
                        f"{comp_idx:8d}  "
                        f"{band_E:9.3f}  "
                        f"{total_ov:10.5f}  "
                        f"{E_plus:9.3f}  "
                        f"{I_plus:8.3f}  "
                        f"{E_minus:9.3f}  "
                        f"{I_minus:8.3f}  "
                        f"{mean_shift:12.5f}  "
                        f"{variance:10.5f}\n"
                    )

        # Write main summaries
        _write(simple_groups, simple_path)
        _write(metal_groups,  metal_path)
        print(f"Written component behavior files:\n  {simple_path}\n  {metal_path}")

        # Now auto-write the zero-cross "bonding" versions
        dir_s, base_s = os.path.dirname(simple_path), os.path.basename(simple_path)
        dir_m, base_m = os.path.dirname(metal_path),  os.path.basename(metal_path)
        bonding_simple = os.path.join(dir_s or ".", f"bonding_{base_s}")
        bonding_metal  = os.path.join(dir_m or ".", f"bonding_{base_m}")

        self.write_component_bonding_summaries(
            by_full,
            bonding_simple,
            bonding_metal
        )
        print(f"Written bonding behavior files:\n  {bonding_simple}\n  {bonding_metal}")

    def _make_bonding_records(
        self,
        recs: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        For each record with fields 'E', 'dE', 'ov', compute a truncated
        'bonding' shift that never crosses zero:
          - downward crossing (E>0→E+dE<0): dE_bond = E+dE
          - upward crossing   (E<0→E+dE>0): dE_bond = -E
          - else: keep original dE
        Returns new rec-list of {'dE': dE_bond, 'ov': ov}.
        """
        bonded = []
        for r in recs:
            E_comp = r["E"]
            dE     = r["dE"]
            E_full = E_comp + dE

            if (E_comp > 0) and (E_full < 0):
                dE_bond = E_full
            elif (E_comp < 0) and (E_full > 0):
                dE_bond = -E_comp
            else:
                dE_bond = dE

            bonded.append({"dE": dE_bond, "ov": r["ov"]})
        return bonded

    def write_component_bonding_summaries(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]],
        simple_path: str,
        metal_path: str
    ) -> None:
        """
        Mirror write_component_summaries but first truncate each dE via
        _make_bonding_records, then classify & write the same columns.
        """
        simple_groups = defaultdict(list)
        metal_groups  = defaultdict(list)
        for comps in by_full.values():
            for r in comps.get("simple", []):
                simple_groups[r["comp_idx"]].append(r)
            for r in comps.get("metal", []):
                metal_groups[r["comp_idx"]].append(r)

        def _write(groups: Dict[int, List[Dict[str, float]]], filename: str):
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w") as f:
                f.write(
                    "# comp_idx  band_E    total_ov    "
                    "E_plus    I_plus    E_minus    I_minus    "
                    "mean_shift    variance\n"
                )
                for comp_idx in sorted(groups):
                    recs      = groups[comp_idx]
                    band_E    = float(recs[0]["E"])
                    bond_recs = self._make_bonding_records(recs)
                    dE_arr    = np.array([b["dE"] for b in bond_recs], dtype=float)
                    ov_arr    = np.array([b["ov"] for b in bond_recs], dtype=float)
                    total_ov  = float(ov_arr.sum())

                    # classify on truncated shifts
                    info = self.classify_state(bond_recs)

                    mean_shift = float(info.get("mean_shift", 0.0))
                    variance   = float(info.get("variance",    0.0))

                    if info["mode"] == "shift":
                        if mean_shift >= 0:
                            E_plus, I_plus  = mean_shift, 1.0
                            E_minus, I_minus = 0.0,       0.0
                        else:
                            E_plus, I_plus  = 0.0,       0.0
                            E_minus, I_minus = mean_shift, 1.0
                    else:
                        E_plus  = float(info.get("E_plus",  0.0))
                        I_plus  = float(info.get("I_plus",  0.0))
                        E_minus = float(info.get("E_minus", 0.0))
                        I_minus = float(info.get("I_minus", 0.0))

                    f.write(
                        f"{comp_idx:8d}  "
                        f"{band_E:9.3f}  "
                        f"{total_ov:10.5f}  "
                        f"{E_plus:9.3f}  "
                        f"{I_plus:8.3f}  "
                        f"{E_minus:9.3f}  "
                        f"{I_minus:8.3f}  "
                        f"{mean_shift:12.5f}  "
                        f"{variance:10.5f}\n"
                    )

        _write(simple_groups, simple_path)
        _write(metal_groups,  metal_path)