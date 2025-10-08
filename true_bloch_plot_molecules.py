from __future__ import annotations
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import mplcursors
    HAS_MPLCURSORS = True
except Exception:
    HAS_MPLCURSORS = False

# classifier import (adjust module name if yours differs)
from component_bonding_behavior_molecules import StateBehaviorClassifier


def _read_rect_txt_delimited(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            blocks = [b.strip() for b in line.split("|")]
            # Case A: 3-block (full | simple | metal) or 4-block (with residual)
            if len(blocks) in (3, 4):
                try:
                    full_fields = blocks[0].split()
                    s_fields = blocks[1].split()
                    m_fields = blocks[2].split()
                    full_idx = int(full_fields[0]); E_full = float(full_fields[1])
                    s_idx = int(s_fields[0]) if len(s_fields) > 0 else 0
                    s_E = float(s_fields[1]) if len(s_fields) > 1 else 0.0
                    s_dE = float(s_fields[2]) if len(s_fields) > 2 else 0.0
                    s_ov = float(s_fields[3]) if len(s_fields) > 3 else 0.0
                    s_ws = float(s_fields[4]) if len(s_fields) > 4 else 0.0
                    m_idx = int(m_fields[0]) if len(m_fields) > 0 else 0
                    m_E = float(m_fields[1]) if len(m_fields) > 1 else 0.0
                    m_dE = float(m_fields[2]) if len(m_fields) > 2 else 0.0
                    m_ov = float(m_fields[3]) if len(m_fields) > 3 else 0.0
                    m_ws = float(m_fields[4]) if len(m_fields) > 4 else 0.0
                    residual = 0.0
                    if len(blocks) == 4:
                        try:
                            residual = float(blocks[3])
                        except Exception:
                            residual = 0.0
                except Exception:
                    continue
                rows.append(dict(
                    full_idx=full_idx,
                    E_full=E_full,
                    simple=dict(idx=s_idx, E=s_E, dE=s_dE, ov_best=s_ov, w_span=s_ws),
                    metal=dict(idx=m_idx, E=m_E, dE=m_dE, ov_best=m_ov, w_span=m_ws),
                    residual=residual
                ))
                continue

            # Case B: single '|' used (full | rest) where rest contains simple then metal then extras
            if len(blocks) == 2:
                try:
                    full_fields = blocks[0].split()
                    rest = blocks[1].split()
                    full_idx = int(full_fields[0]); E_full = float(full_fields[1])
                    # Expect at least simple: idx, E, dE, ov, w_span (5 entries)
                    # then metal: idx, E, dE, ov, w_span (5 entries)
                    if len(rest) < 10:
                        # not enough columns to parse both simple+metal reliably
                        continue
                    s_idx = int(rest[0]); s_E = float(rest[1]); s_dE = float(rest[2])
                    s_ov = float(rest[3]); s_ws = float(rest[4])
                    m_idx = int(rest[5]); m_E = float(rest[6]); m_dE = float(rest[7])
                    m_ov = float(rest[8]); m_ws = float(rest[9])
                    # trailing columns beyond the 10th are ignored here (residual may appear later)
                    residual = 0.0
                    # try to extract a trailing residual if present (common at column 16 in some outputs)
                    if len(rest) >= 16:
                        try:
                            residual = float(rest[15])
                        except Exception:
                            residual = 0.0
                except Exception:
                    continue
                rows.append(dict(
                    full_idx=full_idx,
                    E_full=E_full,
                    simple=dict(idx=s_idx, E=s_E, dE=s_dE, ov_best=s_ov, w_span=s_ws),
                    metal=dict(idx=m_idx, E=m_E, dE=m_dE, ov_best=m_ov, w_span=m_ws),
                    residual=residual
                ))
                continue

            # Fallback: try whitespace-separated minimal parse (first two cols full, next simple idx/E)
            parts = s.split()
            if len(parts) >= 5:
                try:
                    full_idx = int(parts[0]); E_full = float(parts[1])
                    s_idx = int(parts[2]); s_E = float(parts[3]); s_dE = float(parts[4])
                except Exception:
                    continue
                rows.append(dict(
                    full_idx=full_idx,
                    E_full=E_full,
                    simple=dict(idx=s_idx, E=s_E, dE=s_dE, ov_best=0.0, w_span=0.0),
                    metal=dict(idx=0, E=0.0, dE=0.0, ov_best=0.0, w_span=0.0),
                    residual=0.0
                ))
    return rows


def _read_ov_all(path: str) -> Tuple[
    Dict[int, Dict[str, List[Dict[str, float]]]],
    Dict[str, List[Tuple[int, float]]]
]:
    """
    Parse band_matches_rectangular_all.txt and return:
      - by_full: full_idx -> { component_label -> [records] }
      - comp_pairs: component_label -> list[(comp_idx, E)]
    Component label is taken verbatim from column 0.
    """
    by_full: Dict[int, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    comp_idx_E: Dict[str, Dict[int, float]] = defaultdict(dict)

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            comp = parts[0]
            try:
                full_idx = int(parts[1]); comp_idx = int(parts[2])
                E_comp = float(parts[3]); dE_comp = float(parts[4])
                ov = float(parts[5]); w_span = float(parts[6])
            except Exception:
                continue
            rec = dict(comp_idx=comp_idx, E=E_comp, dE=dE_comp, ov=ov, w_span=w_span)
            by_full[full_idx][comp].append(rec)
            comp_idx_E[comp].setdefault(comp_idx, E_comp)

    comp_pairs = {comp: list(d.items()) for comp, d in comp_idx_E.items()}
    return by_full, comp_pairs


@dataclass
class PlotConfig:
    cmap_name_simple: str = "managua_r"
    cmap_name_metal: str = "vanimo_r"
    center_simple: Optional[int] = None
    center_metal: Optional[int] = None
    power_simple_neg: float = 0.25
    power_simple_pos: float = 0.75
    power_metal_neg: float = 0.075
    power_metal_pos: float = 0.075
    min_simple_wspan: float = 0.01
    figsize: Tuple[float, float] = (8.0, 3.0)
    lw_stick: float = 2.0
    xlabel: str = "Energy (eV)"
    ylabel: str = "Normalized"
    show_fermi_line: bool = True
    fermi_line_style: str = ":"
    fermi_line_color: str = "k"
    annotate_on_hover: bool = True
    interactive: bool = True
    shared_molecule_color: bool = False
    pick_primary: Any = False
    energy_range: Optional[Tuple[float, float]] = None
    title_full: str = "Full system"  # added default for title used later

class RectAEPAWColorPlotter:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.cfg = config or PlotConfig()
        # persistent holders
        self._artists_by_comp: Dict[str, List[Any]] = {}
        self._hover_by_comp: Dict[str, List[str]] = {}
        self._cursor_by_comp: Dict[str, Any] = {}
        self._artists_f: List[Any] = []
        self._hover_f: List[str] = []
        self._cursor_f = None

    def _get_cmap(self, name: str):
        try:
            return plt.get_cmap(name)
        except Exception:
            return plt.get_cmap("viridis")

    def _build_colors_rank_pivot(self, pairs: List[Tuple[int, float]], cmap_name: str,
                                 center_idx: Optional[int], power_neg: float, power_pos: float) -> Dict[int, Tuple]:
        if not pairs:
            return {}
        ordered = sorted(pairs, key=lambda t: t[1])
        idxs = [idx for idx, _ in ordered]
        n = len(ordered)
        pivot = idxs.index(center_idx) if (center_idx is not None and center_idx in idxs) else n // 2
        neg_count = max(pivot, 1)
        pos_count = max(n - pivot - 1, 1)
        cmap = self._get_cmap(cmap_name)

        def warp(r):
            a = abs(r)
            exp = power_neg if r < 0 else power_pos
            return a ** exp

        colors = {}
        for i, (idx, _) in enumerate(ordered):
            if i <= pivot:
                r = (i - pivot) / neg_count
            else:
                r = (i - pivot) / pos_count
            w = warp(r)
            v = 0.5 + 0.5 * np.sign(r) * w
            colors[idx] = cmap(np.clip(v, 0.0, 1.0))
        return colors

    def _mix_component_color(self, recs: List[Dict[str, float]], base_colors: Dict[int, Any],
                             default=(0.4, 0.4, 0.4, 1.0)):
        if not recs:
            return default
        num = np.zeros(3); denom = 0.0
        for r in recs:
            idx = int(r["comp_idx"]); w = max(float(r.get("ov", 0.0)), 0.0)
            c = base_colors.get(idx, default)
            num[:3] += w * np.array(c[:3], dtype=float)
            denom += w
        if denom <= 0:
            return default
        rgb = num / denom
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)

    def load(self, path: str):
        rows = _read_rect_txt_delimited(path)
        simple_pairs = {}
        metal_pairs = {}
        for r in rows:
            s, m = r["simple"], r["metal"]
            if s["idx"] > 0:
                simple_pairs.setdefault(s["idx"], s["E"])
            if m["idx"] > 0:
                metal_pairs.setdefault(m["idx"], m["E"])
        return {"rows": rows, "simple_pairs": list(simple_pairs.items()), "metal_pairs": list(metal_pairs.items())}

    def plot(self, path: str, ax: Optional[plt.Axes] = None, bonding: bool = False):
        data = self.load(path)
        rows = data["rows"]
    
        ov_all_path = os.path.join(os.path.dirname(path), "band_matches_rectangular_all.txt")
        by_full: Dict[int, Dict[str, List[Dict[str, float]]]] = {}
        comp_pairs: Dict[str, List[Tuple[int, float]]] = {}
    
        if os.path.isfile(ov_all_path):
            try:
                res = _read_ov_all(ov_all_path)
                if isinstance(res, tuple) and len(res) == 2:
                    by_full, comp_pairs = res
                else:
                    by_full = {}
                    comp_pairs = {}
            except Exception:
                by_full = {}
                comp_pairs = {}
    
        if not comp_pairs and by_full:
            all_labels = set()
            for compdict in by_full.values():
                all_labels.update(compdict.keys())
            comp_pairs = {}
            for lbl in sorted(all_labels):
                seen = {}
                for compdict in by_full.values():
                    for rec in compdict.get(lbl, []):
                        seen.setdefault(int(rec["comp_idx"]), rec["E"])
                comp_pairs[lbl] = list(seen.items())
    
        # LEGACY compatibility: first molecule label -> legacy simple
        mol_labels_ordered = [lbl for lbl in comp_pairs.keys() if lbl.lower() != "metal"]
        legacy_simple_pairs = list(comp_pairs.get(mol_labels_ordered[0], [])) if mol_labels_ordered else []
        legacy_metal_pairs = list(comp_pairs.get("metal", [])) if "metal" in comp_pairs else []
    
        simple_map = dict(legacy_simple_pairs)
        simple_colors = self._build_colors_rank_pivot(
            list(simple_map.items()),
            self.cfg.cmap_name_simple,
            self.cfg.center_simple,
            power_neg=self.cfg.power_simple_neg,
            power_pos=self.cfg.power_simple_pos
        )
    
        metal_map = dict(legacy_metal_pairs)
        metal_colors = self._build_colors_rank_pivot(
            list(metal_map.items()),
            self.cfg.cmap_name_metal,
            self.cfg.center_metal,
            power_neg=self.cfg.power_metal_neg,
            power_pos=self.cfg.power_metal_pos
        )
    
        # order labels
        comp_labels_all = list(comp_pairs.keys())
        metal_present = any(lbl.lower() == "metal" for lbl in comp_labels_all)
        mol_labels = [lbl for lbl in comp_labels_all if lbl.lower() != "metal"]
        comp_labels = (["metal"] if metal_present else []) + mol_labels
    
        # build component colors and class maps (optionally shared molecule map)
        component_colors: Dict[str, Dict[int, Tuple]] = {}
        component_class_maps: Dict[str, Dict[int, Dict[str, float]]] = {}
        try:
            classifier = StateBehaviorClassifier()
        except Exception:
            classifier = None
    
        # shared molecule color map if requested
        if self.cfg.shared_molecule_color and mol_labels:
            shared_map = {}
            for lbl in mol_labels:
                for idx, E in comp_pairs.get(lbl, []):
                    shared_map.setdefault(int(idx), float(E))
            shared_pairs = sorted(shared_map.items(), key=lambda t: t[1])
            shared_mol_colors = self._build_colors_rank_pivot(
                shared_pairs, self.cfg.cmap_name_simple, self.cfg.center_simple,
                self.cfg.power_simple_neg, self.cfg.power_simple_pos
            )
        else:
            shared_mol_colors = None
    
        for comp_label in comp_labels:
            pairs = comp_pairs.get(comp_label, [])
            if comp_label.lower() == "metal":
                component_colors[comp_label] = self._build_colors_rank_pivot(
                    pairs, self.cfg.cmap_name_metal, self.cfg.center_metal,
                    self.cfg.power_metal_neg, self.cfg.power_metal_pos
                )
            else:
                if shared_mol_colors is not None:
                    component_colors[comp_label] = shared_mol_colors
                else:
                    component_colors[comp_label] = self._build_colors_rank_pivot(
                        pairs, self.cfg.cmap_name_simple, self.cfg.center_simple,
                        self.cfg.power_simple_neg, self.cfg.power_simple_pos
                    )
    
            groups = defaultdict(list)
            for full_idx, compdict in by_full.items():
                for rec in compdict.get(comp_label, []):
                    groups[int(rec["comp_idx"])].append(rec)
            comp_map = {}
            if classifier is not None:
                for idx, recs in groups.items():
                    try:
                        comp_map[idx] = classifier.classify_state(recs) if not bonding else classifier.classify_state_bonding(recs)
                    except Exception:
                        comp_map[idx] = {"mode": "shift", "mean_shift": 0.0, "variance": 0.0}
            component_class_maps[comp_label] = comp_map
    
        # prepare figure & axes
        n_mol = len(mol_labels)
        n_comp_axes = n_mol + (1 if metal_present else 0)
        total_rows = max(1, n_comp_axes) + 1  # +1 full axis
    
        if ax is not None:
            fig = ax.figure
            fig.clf()
            axes = fig.subplots(total_rows, 1, sharex=True)
        else:
            figsize = (8, max(3, 1.5 * total_rows))
            fig, axes = plt.subplots(total_rows, 1, sharex=True, figsize=figsize)
    
        if total_rows == 1:
            axes = [axes]
        else:
            axes = list(axes)
    
        comp_axes = axes[:n_comp_axes] if n_comp_axes > 0 else []
        ax_f = axes[-1]
    
        # draw component axes (metal first then molecules), with per-component persistent lists and hover text
        comp_iter_order = (["metal"] if metal_present else []) + mol_labels
        for i, comp_label in enumerate(comp_iter_order):
            axc = comp_axes[i] if i < len(comp_axes) else ax_f
            if self.cfg.show_fermi_line:
                axc.axvline(0.0, color=self.cfg.fermi_line_color, linestyle=self.cfg.fermi_line_style, linewidth=1.0, alpha=0.7)
    
            pairs = comp_pairs.get(comp_label, [])
            colors_map = component_colors.get(comp_label, {})
            class_map = component_class_maps.get(comp_label, {})
    
            # persistent per-component containers
            artists = self._artists_by_comp.setdefault(comp_label, [])
            hovers = self._hover_by_comp.setdefault(comp_label, [])
            # clear previous content for redraw
            artists.clear(); hovers.clear()
    
            for comp_idx, E in sorted(pairs, key=lambda t: t[1]):
                color = colors_map.get(comp_idx, "black")
    
                info = class_map.get(comp_idx, {"mode": "shift", "mean_shift": 0.0, "variance": 0.0})
                # compute component shift key for draw order
                comp_shift_key = 0.0
                try:
                    comp_shift_key = float(info.get("mean_shift", 0.0))
                except Exception:
                    comp_shift_key = 0.0
                zc = abs(comp_shift_key)
                # draw with zorder so larger shifts are on top
                art = axc.vlines(E, 0, 1, color=color, linewidth=self.cfg.lw_stick, zorder=10 + zc)
                artists.append(art)
    
                if info.get("mode", "shift") == "shift":
                    body = (
                        f"net shift  {float(info.get('mean_shift',0.0)):+.3f} eV\n"
                        f"variance   {float(info.get('variance',0.0)):.3f} eV^2"
                    )
                else:
                    Ep = float(info.get("E_plus", 0.0)); Ip = float(info.get("I_plus", 0.0))
                    Ez = float(info.get("E_zero", 0.0)); Iz = float(info.get("I_zero", 0.0))
                    Em = float(info.get("E_minus", 0.0)); Im = float(info.get("I_minus", 0.0))
                    body = (
                        f"up    E={Ep:+.3f}, I={Ip:.3f}\n"
                        f"zero  E={Ez:+.3f}, I={Iz:.3f}\n"
                        f"down  E={Em:+.3f}, I={Im:.3f}"
                    )
    
                # component subplot hover: do NOT include w_span here
                hovers.append(f"{comp_label} band {comp_idx}, E {E:+.3f} eV\n{body}")
    
            axc.set_ylabel(self.cfg.ylabel)
            axc.set_title(comp_label)
            if self.cfg.annotate_on_hover and self.cfg.interactive and artists and HAS_MPLCURSORS:
                # persistent cursor per component
                cur = mplcursors.cursor(artists, hover=True)
                self._cursor_by_comp[comp_label] = cur
                @cur.connect("add")
                def _on_add_comp(sel, hovers=hovers, artists=artists):
                    try:
                        idx = artists.index(sel.artist)
                        sel.annotation.set_text(hovers[idx])
                        sel.annotation.get_bbox_patch().set(alpha=0.9)
                    except Exception:
                        pass
            elif self.cfg.annotate_on_hover and self.cfg.interactive:
                axc.text(0.01, 0.01, "Tip: pip install mplcursors for hover details", transform=axc.transAxes, fontsize=8, color="0.4")
    
        # FULL axis: draw using summed-top-two-molecule-w_span vs metal-top-w_span decision
        if self.cfg.show_fermi_line:
            ax_f.axvline(0.0, color=self.cfg.fermi_line_color, linestyle=self.cfg.fermi_line_style, linewidth=1.0, alpha=0.7)
    
        # build simple_colors/metal_colors fallback for blended logic
        simple_colors = {}
        metal_colors = {}
        if mol_labels:
            simple_label = mol_labels[0]
            simple_colors = self._build_colors_rank_pivot(comp_pairs.get(simple_label, []),
                                                          self.cfg.cmap_name_simple, self.cfg.center_simple,
                                                          self.cfg.power_simple_neg, self.cfg.power_simple_pos)
        if "metal" in comp_pairs:
            metal_colors = self._build_colors_rank_pivot(comp_pairs.get("metal", []),
                                                         self.cfg.cmap_name_metal, self.cfg.center_metal,
                                                         self.cfg.power_metal_neg, self.cfg.power_metal_pos)
    
        # order by magnitude of shift (legacy behavior)
        full_order: List[Tuple[float, Dict[str, Any]]] = []
        for rec in rows:
            s, m = rec["simple"], rec["metal"]
            ws, wm = float(s["w_span"]), float(m["w_span"])
            mode = self.cfg.pick_primary
            if mode is True:
                shift = s["dE"] if ws >= wm else m["dE"]
            elif mode is False:
                if s["idx"] > 0 and ws > self.cfg.min_simple_wspan:
                    shift = s["dE"]
                else:
                    shift = m["dE"]
            elif mode == "blended":
                info = {}
                if mol_labels:
                    comp_map = component_class_maps.get(mol_labels[0], {})
                    info = comp_map.get(s["idx"], {"mean_shift": 0.0})
                shift = float(info.get("mean_shift", 0.0))
            else:
                shift = s["dE"] if s["idx"] > 0 else m["dE"]
            full_order.append((abs(shift), rec))
        full_order.sort(key=lambda x: x[0])
    
        # reset persistent full lists
        self._artists_f.clear(); self._hover_f.clear()
    
        # We'll enforce per-band metal-first then molecule-split draw by buffering per-iteration
        for _, rec in full_order:
            E_full = float(rec["E_full"]); s, m = rec["simple"], rec["metal"]
            ws, wm = float(s["w_span"]), float(m["w_span"])
            mode = self.cfg.pick_primary
    
            full_idx = int(rec["full_idx"])
            comps = by_full.get(full_idx, {})
    
            # compute top-two molecule matches (for hover and coloring) and sum their w_spans
            mol_top_recs = []
            for mol_label in mol_labels[:2]:
                lst = comps.get(mol_label, [])
                mol_top_recs.append(max(lst, key=lambda r: r.get("ov", 0.0)) if lst else None)
    
            total_mol_wspan = 0.0
            for top in mol_top_recs:
                if top is not None:
                    try:
                        total_mol_wspan += float(top.get("w_span", 0.0))
                    except Exception:
                        pass
    
            # metal top-match w_span for comparison
            metal_list = comps.get("metal", [])
            metal_top = max(metal_list, key=lambda r: r.get("ov", 0.0)) if metal_list else None
            metal_wspan = float(metal_top.get("w_span", 0.0)) if metal_top is not None else 0.0
    
            # legacy prefer_simple retained
            if mode is True:
                prefer_simple = ws >= wm
            elif mode is False:
                prefer_simple = (s["idx"] > 0 and ws > self.cfg.min_simple_wspan)
            elif mode == "blended":
                comp_map = component_class_maps.get(mol_labels[0], {}) if mol_labels else {}
                info = comp_map.get(s["idx"], {"mean_shift": 0.0})
                prefer_simple = float(info.get("mean_shift", 0.0)) != 0.0
            else:
                prefer_simple = False
    
            # decide by comparing summed top-two molecule w_spans to metal top w_span
            prefer_to_use_molecules = (total_mol_wspan > metal_wspan)
            if not prefer_to_use_molecules and prefer_simple:
                prefer_to_use_molecules = True
    
            # determine top-two molecule colors for split halves
            mol_colors = []
            for i, mol_label in enumerate(mol_labels[:2]):
                top = mol_top_recs[i] if i < len(mol_top_recs) else None
                if top is None:
                    mol_colors.append(None)
                else:
                    base = component_colors.get(mol_label, {})
                    mol_colors.append(base.get(int(top["comp_idx"]), None))
    
            default_metal_color = self._mix_component_color(comps.get("metal", []), component_colors.get("metal", {}))
    
            # build hover_text (includes w_span per-component and total_molecule_w_span)
            comp_lines: List[str] = []
            order_labels = (["metal"] if "metal" in comps else []) + [lbl for lbl in mol_labels if lbl in comps]
            for comp_label in order_labels:
                entries = comps.get(comp_label, [])
                if not entries:
                    continue
                top = max(entries, key=lambda r: r.get("ov", 0.0))
                comp_lines.append(
                    f"{comp_label}: idx {int(top['comp_idx'])}, E {top['E']:+.3f}, ov {top['ov']:.5f}, w_span {float(top.get('w_span',0.0)):.5f}"
                )
    
            hover_text = (
                f"full_idx {full_idx}\n"
                f"E_full {E_full:+.3f}\n"
                f"total_molecule_w_span {total_mol_wspan:.5f}\n"
                f"components:\n" + "\n".join(comp_lines) + "\n"
                f"residual {rec.get('residual',0.0):.5f}"
            )
    
            # Create buckets so metal-style artists are appended first (drawn first), molecules after
            metal_bucket = []
            mol_bucket = []
    
            # Create artist objects but buffer them
            if prefer_to_use_molecules and any(c is not None for c in mol_colors):
                top_color = mol_colors[0] or default_metal_color
                bottom_color = (mol_colors[1] if len(mol_colors) > 1 else None) or default_metal_color
                art_top = ax_f.vlines(E_full, 0.5, 1.0, color=top_color, linewidth=self.cfg.lw_stick)
                art_bot = ax_f.vlines(E_full, 0.0, 0.5, color=bottom_color, linewidth=self.cfg.lw_stick)
                mol_bucket.append((art_top, art_bot, hover_text))
            else:
                if prefer_simple and s["idx"] > 0:
                    color = simple_colors.get(s["idx"], (0.4, 0.4, 0.4, 1.0))
                else:
                    color = default_metal_color or metal_colors.get(m["idx"], (0.4, 0.4, 0.4, 1.0))
                art = ax_f.vlines(E_full, 0, 1, color=color, linewidth=self.cfg.lw_stick)
                metal_bucket.append((art, hover_text))
    
            # append metal bucket first then molecule bucket so molecules plot over metal
            for art, htext in metal_bucket:
                self._artists_f.append(art)
                self._hover_f.append(htext)
            for art_top, art_bot, htext in mol_bucket:
                self._artists_f.extend([art_top, art_bot])
                self._hover_f.extend([htext, htext])
    
        ax_f.set_title(self.cfg.title_full)
        ax_f.set_ylabel(self.cfg.ylabel); ax_f.set_xlabel(self.cfg.xlabel)
        if self.cfg.energy_range:
            ax_f.set_xlim(self.cfg.energy_range)
    
        # attach persistent cursor for full axis
        if self.cfg.annotate_on_hover and self.cfg.interactive and self._artists_f and HAS_MPLCURSORS:
            self._cursor_f = mplcursors.cursor(self._artists_f, hover=True)
            @self._cursor_f.connect("add")
            def _on_add_f(sel):
                try:
                    idx = self._artists_f.index(sel.artist)
                    sel.annotation.set_text(self._hover_f[idx])
                    sel.annotation.get_bbox_patch().set(alpha=0.9)
                except Exception:
                    pass
        elif self.cfg.annotate_on_hover and self.cfg.interactive:
            ax_f.text(0.01, 0.01, "Tip: pip install mplcursors for hover details", transform=ax_f.transAxes, fontsize=8, color="0.4")
    
        fig.tight_layout()
        return fig, axes