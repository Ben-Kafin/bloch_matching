# -*- coding: utf-8 -*-
"""
component_bonding_behavior.py — classifier updated for multi-component (multiple molecules)

Changes:
- classify_state unchanged.
- classify_all now returns a mapping: component_label -> comp_idx -> classification dict.
- write_component_summaries now accepts `by_full` in the multi-component shape
  and writes one summary file per component label (named by component_label + suffix)
  in the same directory as the provided base paths. It still writes the legacy
  simple_path and metal_path files when those labels are present.
- write_component_bonding_summaries updated likewise.

Usage:
  classifier.write_component_summaries(by_full, simple_path, metal_path)
Will create files:
  <dir_of_simple_path>/<component_label>_behavior.txt   (for each component_label)
  <dir_of_simple_path>/bonding_<component_label>_behavior.txt
and, if 'simple'/'metal' labels are present, also writes the exact simple_path/metal_path.
"""
from __future__ import annotations
import os
from collections import defaultdict
from typing import List, Dict, Union, Tuple, Any

import numpy as np


class StateBehaviorClassifier:
    def __init__(self):
        pass

    def classify_state(
        self,
        recs: List[Dict[str, float]]
    ) -> Dict[str, Union[str, float]]:
        dE = np.array([r["dE"] for r in recs], dtype=float)
        ov = np.array([r["ov"] for r in recs], dtype=float)
        total_ov = float(ov.sum())

        if total_ov <= 0:
            return {
                "mode": "shift",
                "mean_shift": 0.0,
                "variance": 0.0,
                "total_ov": 0.0
            }

        mean_shift = float((ov * dE).sum() / total_ov)
        var = float(((ov * (dE - mean_shift) ** 2).sum()) / total_ov)

        w_plus = float(ov[dE > 0].sum())
        w_minus = float(ov[dE < 0].sum())
        w_total = w_plus + w_minus

        I_plus = w_plus / w_total if w_total > 0 else 0.0
        I_minus = w_minus / w_total if w_total > 0 else 0.0

        E_plus = float((ov[dE > 0] * dE[dE > 0]).sum() / w_plus) if w_plus > 0 else 0.0
        E_minus = float((ov[dE < 0] * dE[dE < 0]).sum() / w_minus) if w_minus > 0 else 0.0

        if I_plus == 0.0 or I_minus == 0.0:
            return {
                "mode": "shift",
                "mean_shift": mean_shift,
                "variance": var,
                "total_ov": total_ov
            }

        return {
            "mode": "split",
            "mean_shift": mean_shift,
            "variance": var,
            "E_plus": E_plus,
            "I_plus": I_plus,
            "E_minus": E_minus,
            "I_minus": I_minus,
            "total_ov": total_ov
        }

    def classify_all(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]]
    ) -> Dict[str, Dict[int, Dict[str, Union[str, float]]]]:
        """
        Classify every component across all full-state indices.

        Returns:
          { component_label: { comp_idx: classification_dict, ... }, ... }
        """
        # Build groups per component label -> comp_idx -> recs
        comp_groups: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
        for full_idx, comps in by_full.items():
            for comp_label, recs in comps.items():
                for r in recs:
                    comp_groups[comp_label][int(r["comp_idx"])].append(r)

        results: Dict[str, Dict[int, Dict[str, Union[str, float]]]] = {}
        for comp_label, idx_map in comp_groups.items():
            results[comp_label] = {}
            for idx, recs in idx_map.items():
                # classify using dE/ov form expected by classify_state
                # note recs are dicts with keys E,dE,ov,w_span
                results[comp_label][idx] = self.classify_state(recs)
        return results

    def write_component_summaries(
        self,
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]],
        simple_path: str,
        metal_path: str
    ) -> None:
        """
        Write per-component summary files for every component label found in by_full.

        For each component_label, writes:
          <dir_of_simple_path>/<component_label>_behavior.txt

        Additionally, if 'simple' or 'metal' labels exist, the function will
        also write exactly the files simple_path and metal_path (legacy names).
        The bonding (zero-cross) summaries are also written as bonding_<basename>.
        """
        # Build comp -> comp_idx -> list[recs]
        comp_groups: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
        for comps in by_full.values():
            for comp_label, recs in comps.items():
                for r in recs:
                    comp_groups[comp_label][int(r["comp_idx"])].append(r)

        # Determine output directory base
        simple_dir = os.path.dirname(simple_path) or "."
        metal_dir = os.path.dirname(metal_path) or simple_dir

        def _write(groups: Dict[int, List[Dict[str, float]]], filename: str):
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w") as f:
                f.write(
                    "# comp_idx  band_E    total_ov    "
                    "E_plus    I_plus    E_minus    I_minus    "
                    "mean_shift    variance\n"
                )
                for comp_idx in sorted(groups):
                    recs = groups[comp_idx]
                    band_E = float(recs[0]["E"])
                    dE_arr = np.array([r["dE"] for r in recs], dtype=float)
                    ov_arr = np.array([r["ov"] for r in recs], dtype=float)
                    total_ov = float(ov_arr.sum())

                    info = self.classify_state([{"dE": d, "ov": w} for d, w in zip(dE_arr, ov_arr)])

                    mean_shift = float(info.get("mean_shift", 0.0))
                    variance = float(info.get("variance", 0.0))

                    if info["mode"] == "shift":
                        if mean_shift >= 0:
                            E_plus, I_plus = mean_shift, 1.0
                            E_minus, I_minus = 0.0, 0.0
                        else:
                            E_plus, I_plus = 0.0, 0.0
                            E_minus, I_minus = mean_shift, 1.0
                    else:
                        E_plus = float(info.get("E_plus", 0.0))
                        I_plus = float(info.get("I_plus", 0.0))
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

        # Write one file per component label
        for comp_label, groups in comp_groups.items():
            # sanitize label for filename (keep human readable)
            safe_label = str(comp_label).strip().replace(os.sep, "_")
            out_filename = os.path.join(simple_dir, f"{safe_label}_behavior.txt")
            _write(groups, out_filename)

            # write bonding variant
            bonding_filename = os.path.join(simple_dir, f"bonding_{safe_label}_behavior.txt")
            self.write_component_bonding_summaries({k: {comp_label: v} for k, v in by_full.items()}, bonding_filename, bonding_filename)
            # note: above call wraps into expected shape and writes both simple/metal variants inside

        # Additionally write legacy files if requested and present
        if "simple" in comp_groups:
            try:
                _write(comp_groups["simple"], simple_path)
            except Exception:
                pass
        if "metal" in comp_groups:
            try:
                _write(comp_groups["metal"], metal_path)
            except Exception:
                pass

    def _make_bonding_records(
        self,
        recs: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        bonded = []
        for r in recs:
            E_comp = r["E"]
            dE = r["dE"]
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
        Variant of write_component_summaries that truncates dE via _make_bonding_records,
        then writes the same columns. For multi-component use, this will write one file
        per component label named <dir>/<component_label> (bonding_...).
        """
        # Build comp -> comp_idx -> recs
        comp_groups: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
        for comps in by_full.values():
            for comp_label, recs in comps.items():
                for r in recs:
                    comp_groups[comp_label][int(r["comp_idx"])].append(r)

        # same writer as before but using truncated recs
        def _write(groups: Dict[int, List[Dict[str, float]]], filename: str):
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w") as f:
                f.write(
                    "# comp_idx  band_E    total_ov    "
                    "E_plus    I_plus    E_minus    I_minus    "
                    "mean_shift    variance\n"
                )
                for comp_idx in sorted(groups):
                    recs = groups[comp_idx]
                    band_E = float(recs[0]["E"])
                    bond_recs = self._make_bonding_records(recs)
                    dE_arr = np.array([b["dE"] for b in bond_recs], dtype=float)
                    ov_arr = np.array([b["ov"] for b in bond_recs], dtype=float)
                    total_ov = float(ov_arr.sum())

                    info = self.classify_state(bond_recs)

                    mean_shift = float(info.get("mean_shift", 0.0))
                    variance = float(info.get("variance", 0.0))

                    if info["mode"] == "shift":
                        if mean_shift >= 0:
                            E_plus, I_plus = mean_shift, 1.0
                            E_minus, I_minus = 0.0, 0.0
                        else:
                            E_plus, I_plus = 0.0, 0.0
                            E_minus, I_minus = mean_shift, 1.0
                    else:
                        E_plus = float(info.get("E_plus", 0.0))
                        I_plus = float(info.get("I_plus", 0.0))
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

        # write per-component bonding summaries
        for comp_label, groups in comp_groups.items():
            safe_label = str(comp_label).strip().replace(os.sep, "_")
            bonding_filename = os.path.join(os.path.dirname(simple_path) or ".", f"bonding_{safe_label}_behavior.txt")
            _write(groups, bonding_filename)

        # also write legacy bonding files if 'simple'/'metal' present
        if "simple" in comp_groups:
            try:
                _write(comp_groups["simple"], os.path.join(os.path.dirname(simple_path) or ".", f"bonding_{os.path.basename(simple_path)}"))
            except Exception:
                pass
        if "metal" in comp_groups:
            try:
                _write(comp_groups["metal"], os.path.join(os.path.dirname(metal_path) or ".", f"bonding_{os.path.basename(metal_path)}"))
            except Exception:
                pass