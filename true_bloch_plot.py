from __future__ import annotations
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False

# import the classifier you implemented
#from component_behavior import StateBehaviorClassifier
from component_bonding_behavior import StateBehaviorClassifier


def _read_rect_txt_delimited(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            blocks = [b.strip() for b in line.split("|")]
            if len(blocks) != 4:
                continue

            full_fields = blocks[0].split()
            s_fields    = blocks[1].split()
            m_fields    = blocks[2].split()

            try:
                full_idx = int(full_fields[0]);    E_full = float(full_fields[1])
                s_idx    = int(s_fields[0]);       s_E    = float(s_fields[1])
                s_dE     = float(s_fields[2]);     s_ov   = float(s_fields[3])
                s_ws     = float(s_fields[4])
                m_idx    = int(m_fields[0]);       m_E    = float(m_fields[1])
                m_dE     = float(m_fields[2]);     m_ov   = float(m_fields[3])
                m_ws     = float(m_fields[4])
                residual = float(blocks[3])
            except Exception:
                continue

            rows.append(dict(
                full_idx = full_idx,
                E_full   = E_full,
                simple   = dict(
                    idx     = s_idx,
                    E       = s_E,
                    dE      = s_dE,
                    ov_best = s_ov,
                    w_span  = s_ws
                ),
                metal    = dict(
                    idx     = m_idx,
                    E       = m_E,
                    dE      = m_dE,
                    ov_best = m_ov,
                    w_span  = m_ws
                ),
                residual = residual,
            ))

    if not rows:
        raise ValueError(f"No valid rows parsed from {path}.")
    return rows


def _read_ov_all(path: str) -> Tuple[
    Dict[int, Dict[str, List[Dict[str, float]]]],
    List[Tuple[int, float]],
    List[Tuple[int, float]]
]:
    by_full: Dict[int, Dict[str, List[Dict[str, float]]]] = defaultdict(
        lambda: {"simple": [], "metal": []}
    )
    simple_idx_E: Dict[int, float] = {}
    metal_idx_E: Dict[int, float]  = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            comp = parts[0].lower()
            try:
                full_idx = int(parts[1])
                comp_idx = int(parts[2])
                E_comp   = float(parts[3])
                dE_comp  = float(parts[4])
                ov        = float(parts[5])
                w_span    = float(parts[6])
            except Exception:
                continue

            record = dict(
                comp_idx = comp_idx,
                E        = E_comp,
                dE       = dE_comp,
                ov       = ov,
                w_span   = w_span
            )
            if comp == "simple":
                by_full[full_idx]["simple"].append(record)
                simple_idx_E.setdefault(comp_idx, E_comp)
            elif comp == "metal":
                by_full[full_idx]["metal"].append(record)
                metal_idx_E.setdefault(comp_idx, E_comp)

    simple_pairs = list(simple_idx_E.items())
    metal_pairs  = list(metal_idx_E.items())
    return by_full, simple_pairs, metal_pairs


@dataclass
class PlotConfig:
    cmap_name_simple: str                       = "coolwarm"
    cmap_name_metal: str                        = "vanimo_r"
    center_simple: Optional[int]                = None
    center_metal: Optional[int]                 = None
    power_simple_neg: float                     = 0.36
    power_simple_pos: float                     = 0.75
    power_metal_neg: float                      = 0.075
    power_metal_pos: float                      = 0.075
    min_simple_wspan: float                     = 0.01
    figsize: Tuple[float, float]                = (8.0, 8.0)
    lw_stick: float                             = 2.0
    xlabel: str                                 = "Energy (eV)"
    ylabel: str                                 = "Normalized"
    title_metal: str                            = "Metal"
    title_simple: str                           = "Simple"
    title_full: str                             = "Full"
    show_fermi_line: bool                       = True
    fermi_line_style: str                       = ":"
    fermi_line_color: str                       = "k"
    annotate_on_hover: bool                     = True
    interactive: bool                           = True
    show_colorbar: bool                         = False
    pick_primary: Any                           = True
    energy_range: Optional[Tuple[float, float]] = None
    align_full_to_metal_min_band: Optional[int] = None





class RectAEPAWColorPlotter:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.cfg = config or PlotConfig()

    def _get_cmap(self, name: str):
        try:
            return plt.get_cmap(name)
        except Exception:
            return plt.get_cmap("coolwarm")

    def _build_colors_rank_pivot(
        self,
        pairs: List[Tuple[int, float]],
        cmap_name: str,
        center_idx: Optional[int],
        power_neg: float = 1.0,
        power_pos: float = 1.0,
        mode: str = "power",
        log_base: float = 9.0
    ) -> Dict[int, Tuple[float, float, float, float]]:
        if not pairs:
            return {}
        ordered = sorted(pairs, key=lambda t: t[1])
        idxs    = [idx for idx, _ in ordered]
        n       = len(ordered)

        if center_idx is not None and center_idx in idxs:
            pivot = idxs.index(center_idx)
        else:
            pivot = n // 2

        neg_count = max(pivot, 1)
        pos_count = max(n - pivot - 1, 1)
        cmap      = self._get_cmap(cmap_name)

        def warp(r: float) -> float:
            a = abs(r)
            if mode == "power":
                exp = power_neg if r < 0 else power_pos
                return a ** exp
            return np.log1p(log_base * a) / np.log1p(log_base)

        colors: Dict[int, Tuple[float, float, float, float]] = {}
        for i, (idx, _) in enumerate(ordered):
            if i <= pivot:
                r = (i - pivot) / neg_count
            else:
                r = (i - pivot) / pos_count
            w = warp(r)
            v = 0.5 + 0.5 * np.sign(r) * w
            colors[idx] = cmap(np.clip(v, 0.0, 1.0))
        return colors

    def _mix_component_color(
        self,
        recs: List[Dict[str, float]],
        base_colors: Dict[int, Any],
        default=(0.4, 0.4, 0.4, 1.0)
    ) -> Tuple[float, float, float, float]:
        if not recs:
            return default
        num   = np.zeros(3, dtype=float)
        denom = 0.0
        for r in recs:
            idx = int(r["comp_idx"])
            w   = max(float(r["ov"]), 0.0)
            c   = base_colors.get(idx, default)
            num[:3] += w * np.array(c[:3], dtype=float)
            denom  += w
        if denom <= 0:
            return default
        rgb = num / denom
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)

    def load(self, path: str):
        rows = _read_rect_txt_delimited(path)
        simple_pairs: Dict[int, float] = {}
        metal_pairs:  Dict[int, float] = {}
        for r in rows:
            s, m = r["simple"], r["metal"]
            if s["idx"] > 0:
                simple_pairs.setdefault(s["idx"], s["E"])
            if m["idx"] > 0:
                metal_pairs.setdefault(m["idx"], m["E"])
        return {
            "rows": rows,
            "simple_pairs": list(simple_pairs.items()),
            "metal_pairs":  list(metal_pairs.items()),
        }

    def plot(self, path: str, ax: Optional[plt.Axes] = None, bonding: bool = False):
        data = self.load(path)
        rows = data["rows"]

        ov_all_path = os.path.join(
            os.path.dirname(path),
            "band_matches_rectangular_all.txt"
        )
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]] = {}
        ov_all_simple_pairs: List[Tuple[int, float]] = []
        ov_all_metal_pairs:  List[Tuple[int, float]] = []
        if os.path.isfile(ov_all_path):
            try:
                by_full, ov_all_simple_pairs, ov_all_metal_pairs = _read_ov_all(ov_all_path)
            except Exception:
                by_full = {}
        
        # --- now unconditionally dump the summaries via an instance call ---
        classifier = StateBehaviorClassifier()
        out_dir    = os.path.dirname(path) or "."
        simple_out = os.path.join(out_dir, "simple_behavior.txt")
        metal_out  = os.path.join(out_dir, "metal_behavior.txt")
        
        # This must be an instance method on your classifier:
        classifier.write_component_summaries(by_full, simple_out, metal_out)
        # choose between normal vs. zero‐cross (“bonding”) classification
        classify_fn = (
            classifier.classify_state_bonding
            if bonding
            else classifier.classify_state
        )

        # group full rec-lists (keep 'E', 'dE', 'ov') by component index
        simple_groups: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        metal_groups:  Dict[int, List[Dict[str, float]]] = defaultdict(list)
        for comps in by_full.values():
            for rec in comps.get("simple", []):
                simple_groups[rec["comp_idx"]].append(rec)
            for rec in comps.get("metal", []):
                metal_groups[rec["comp_idx"]].append(rec)

        # build classification maps using the selected function
        simple_class_map = {
            idx: classify_fn(records)
            for idx, records in simple_groups.items()
        }
        metal_class_map = {
            idx: classify_fn(records)
            for idx, records in metal_groups.items()
        }

        # build color dictionaries
        simple_map = dict(data["simple_pairs"])
        for idx, E in ov_all_simple_pairs:
            simple_map.setdefault(idx, E)
        simple_colors = self._build_colors_rank_pivot(
            list(simple_map.items()),
            self.cfg.cmap_name_simple,
            self.cfg.center_simple,
            power_neg=self.cfg.power_simple_neg,
            power_pos=self.cfg.power_simple_pos,
            mode="power"
        )

        metal_map = dict(data["metal_pairs"])
        for idx, E in ov_all_metal_pairs:
            metal_map.setdefault(idx, E)
        metal_colors = self._build_colors_rank_pivot(
            list(metal_map.items()),
            self.cfg.cmap_name_metal,
            self.cfg.center_metal,
            power_neg=self.cfg.power_metal_neg,
            power_pos=self.cfg.power_metal_pos,
            mode="power"
        )

        # prepare figure & axes
        if ax is not None:
            fig = ax.figure; fig.clf()
            axes = fig.subplots(3, 1, sharex=True)
        else:
            fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=self.cfg.figsize
            )
        ax_m, ax_s, ax_f = axes

        # fermi lines at 0eV
        if self.cfg.show_fermi_line:
            for a in axes:
                a.axvline(0.0,
                          color=self.cfg.fermi_line_color,
                          linestyle=self.cfg.fermi_line_style,
                          linewidth=1.0, alpha=0.7)

        # --- metal sticks + hover classification ---
        # --- metal sticks + hover classification ---
        artists_m: List[Any] = []
        hover_m:   List[str] = []
        
        for comp_idx, E in sorted(data["metal_pairs"], key=lambda t: t[1]):
            color = metal_colors.get(comp_idx, "black")
            line  = ax_m.vlines(
                E, 0, 1,
                color=color,
                linestyle="-",
                lw=self.cfg.lw_stick
            )
            artists_m.append(line)
        
            # fetch the classifier output (mode, mean_shift, variance, E+/I+/E-/I-)
            info = metal_class_map.get(
                comp_idx,
                {"mode": "shift", "mean_shift": 0.0, "variance": 0.0}
            )
            ms  = float(info["mean_shift"])
            var = float(info["variance"])
        
            prefix = f"band idx {comp_idx}, E {E:+.3f} eV\n"
        
            if info["mode"] == "shift":
                # pure shift: show net shift + variance
                body = (
                    f"net shift  {ms:+.3f} eV\n"
                    f"variance   {var:.3f} eV^2"
                )
            else:
                # split: show up/down then mean_shift + variance
                Ep = float(info["E_plus"]);  Ip = float(info["I_plus"])
                Ez = float(info['E_zero']);  Iz = float(info['I_zero'])
                Em = float(info["E_minus"]); Im = float(info["I_minus"])
                body = (
                    f"up    E={Ep:+.3f}, I={Ip:.3f}\n"
                    f"zero  E={Ez:+.3f}, I={Iz:.3f}\n"
                    f"down  E={Em:+.3f}, I={Im:.3f}\n"
                    f"mean shift {ms:+.3f} eV\n"
                    f"variance   {var:.3f} eV^2"
                )
        
            hover_m.append(prefix + body)
        
        ax_m.set_title(self.cfg.title_metal)
        ax_m.set_ylabel(self.cfg.ylabel)
        
        if self.cfg.annotate_on_hover and self.cfg.interactive and artists_m and HAS_MPLCURSORS:
            cursor_m = mplcursors.cursor(artists_m, hover=True)
            @cursor_m.connect("add")
            def _on_add_m(sel):
                idx = artists_m.index(sel.artist)
                sel.annotation.set_text(hover_m[idx])
                sel.annotation.get_bbox_patch().set(alpha=0.9)
        elif self.cfg.annotate_on_hover and self.cfg.interactive:
            ax_m.text(
                0.01, 0.01,
                "Tip: pip install mplcursors for hover details",
                transform=ax_m.transAxes, fontsize=8, color="0.4"
            )
            
        # --- simple sticks + hover classification ---
        # --- simple sticks + hover classification ---
        artists_s: List[Any] = []
        hover_s:   List[str] = []
        
        for comp_idx, E in sorted(data["simple_pairs"], key=lambda t: t[1]):
            color = simple_colors.get(comp_idx, "black")
            line  = ax_s.vlines(
                E, 0, 1,
                color=color,
                linestyle="-",
                lw=self.cfg.lw_stick
            )
            artists_s.append(line)
        
            info = simple_class_map.get(
                comp_idx,
                {"mode": "shift", "mean_shift": 0.0, "variance": 0.0}
            )
            ms  = float(info["mean_shift"])
            var = float(info["variance"])
        
            prefix = f"band idx {comp_idx}, E {E:+.3f} eV\n"
        
            if info["mode"] == "shift":
                body = (
                    f"net shift  {ms:+.3f} eV\n"
                    f"variance   {var:.3f} eV^2"
                )
            else:
                Ep = float(info["E_plus"]);  Ip = float(info["I_plus"])
                Ez = float(info['E_zero']);  Iz = float(info['I_zero'])
                Em = float(info["E_minus"]); Im = float(info["I_minus"])
                body = (
                    f"up    E={Ep:+.3f}, I={Ip:.3f}\n"
                    f"zero  E={Ez:+.3f}, I={Iz:.3f}\n"
                    f"down  E={Em:+.3f}, I={Im:.3f}\n"
                    f"mean shift {ms:+.3f} eV\n"
                    f"variance   {var:.3f} eV^2"
                )
        
            hover_s.append(prefix + body)
        
        ax_s.set_title(self.cfg.title_simple)
        ax_s.set_ylabel(self.cfg.ylabel)
        
        if self.cfg.annotate_on_hover and self.cfg.interactive and artists_s and HAS_MPLCURSORS:
            cursor_s = mplcursors.cursor(artists_s, hover=True)
            @cursor_s.connect("add")
            def _on_add_s(sel):
                idx = artists_s.index(sel.artist)
                sel.annotation.set_text(hover_s[idx])
                sel.annotation.get_bbox_patch().set(alpha=0.9)
        elif self.cfg.annotate_on_hover and self.cfg.interactive:
            ax_s.text(
                0.01, 0.01,
                "Tip: pip install mplcursors for hover details",
                transform=ax_s.transAxes, fontsize=8, color="0.4"
            )

        # --- full sticks with existing hover info ---
        artists_f: List[Any] = []
        hover_f:   List[str] = []
        for rec in rows:
            E_full = float(rec["E_full"])
            s, m   = rec["simple"], rec["metal"]
            ws, wm = float(s["w_span"]), float(m["w_span"])
            mode   = self.cfg.pick_primary

            # original True/False/blended logic unchanged...
            if mode is True:
                prefer_simple = ws >= wm
                if prefer_simple and s["idx"] > 0:
                    color = simple_colors.get(s["idx"], "black")
                elif m["idx"] > 0:
                    color = metal_colors.get(m["idx"], "black")
                else:
                    color = "0.4"
            elif mode is False:
                if s["idx"] > 0 and ws > self.cfg.min_simple_wspan:
                    color = simple_colors.get(s["idx"], "black")
                else:
                    color = metal_colors.get(m["idx"], "black")
            elif mode == "blended":
                full_idx = int(rec["full_idx"])
                have_all = full_idx in by_full
                if have_all:
                    recs_s = by_full[full_idx]["simple"]
                    recs_m = by_full[full_idx]["metal"]
                    c_s = np.array(self._mix_component_color(recs_s, simple_colors))
                    c_m = np.array(self._mix_component_color(recs_m, metal_colors))
                else:
                    c_s = np.array(simple_colors.get(s["idx"], (0.4,0.4,0.4,1.0)))
                    c_m = np.array(metal_colors.get(m["idx"],  (0.4,0.4,0.4,1.0)))
                total_w = ws + wm
                if total_w > 0:
                    blend = (c_s[:3]*ws + c_m[:3]*wm) / total_w
                else:
                    blend = np.array([0.4,0.4,0.4])
                color = (*blend[:3], 1.0)
            else:
                raise ValueError("pick_primary must be True, False, or 'blended'")

            line = ax_f.vlines(E_full, 0, 1,
                               color=color,
                               linestyle="-",
                               lw=self.cfg.lw_stick)
            artists_f.append(line)
            hover_f.append(
                f"full_idx: {int(rec['full_idx'])}\n"
                f"E_full: {E_full:+.3f}\n"
                f"simple: idx {int(s['idx'])}, E {s['E']:+.3f}, "
                f"dE {s['dE']:+.3f}, ov_best {s['ov_best']:.5f}, "
                f"w_span {s['w_span']:.5f}\n"
                f"metal:  idx {int(m['idx'])}, E {m['E']:+.3f}, "
                f"dE {m['dE']:+.3f}, ov_best {m['ov_best']:.5f}, "
                f"w_span {m['w_span']:.5f}\n"
                f"residual: {rec['residual']:.5f}"
            )

        ax_f.set_title(self.cfg.title_full)
        ax_f.set_ylabel(self.cfg.ylabel)
        ax_f.set_xlabel(self.cfg.xlabel)

        if self.cfg.energy_range:
            for a in axes:
                a.set_xlim(self.cfg.energy_range)

        fig.tight_layout()

        if self.cfg.annotate_on_hover and self.cfg.interactive and artists_f and HAS_MPLCURSORS:
            cursor_f = mplcursors.cursor(artists_f, hover=True)
            @cursor_f.connect("add")
            def _(sel):
                idx = artists_f.index(sel.artist)
                sel.annotation.set_text(hover_f[idx])
                sel.annotation.get_bbox_patch().set(alpha=0.9)
        elif self.cfg.annotate_on_hover and self.cfg.interactive:
            ax_f.text(
                0.01, 0.01,
                "Tip: pip install mplcursors for hover details",
                transform=ax_f.transAxes, fontsize=8, color="0.4"
            )

        return fig, axes
