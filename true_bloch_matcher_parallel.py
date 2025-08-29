# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 12:19:18 2025

@author: Benjamin Kafin
"""
import os
import numpy as np
from typing import Dict, List, Optional, Sequence
from scipy.sparse import issparse
from scipy.optimize import linear_sum_assignment
from ase.io import read as ase_read
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from vaspwfc import vaspwfc
from aewfc import vasp_ae_wfc
from true_bloch_plot import RectAEPAWColorPlotter, PlotConfig

def _process_component(matcher, sys_data, Cslice, T, out_dir, W):
    """
    Build AE projectors, lift into full-system space, fuse with PW,
    and save the resulting true Bloch states.
    """
    # 1) build & lift AE projectors
    B_native = matcher.form_B_for_slice(sys_data.ae, Cslice)
    B_lifted = matcher.lift_B(B_native, T)

    # 2) fuse PW + AE into true Bloch coordinates
    psi, norms = matcher.fuse_true_bloch_rr(Cslice, B_lifted, W)

    # 3) save to disk
    matcher.save_true_bloch(out_dir, psi, norms)

    return psi

class RectangularTrueBlochMatcher:
    def __init__(self,
                 simple_dir,
                 metal_dir,
                 full_dir,
                 k_index=1,
                 tol_map=1e-3,
                 check_species=True,
                 band_window_simple=None,
                 band_window_metal=None,
                 band_window_full=None,
                 reuse_cached=False,
                 align_full_to_metal_min_band: int | None = None):

        self.simple_dir = simple_dir
        self.metal_dir = metal_dir
        self.full_dir = full_dir
        self.k_index = k_index
        self.tol_map = tol_map
        self.check_species = check_species
        self.band_window_simple = band_window_simple
        self.band_window_metal = band_window_metal
        self.band_window_full = band_window_full
        self.reuse_cached = reuse_cached
        self.align_full_to_metal_min_band = align_full_to_metal_min_band
        

    class SystemData:
        def __init__(self, name: str, directory: str):
            self.name = name
            self.directory = directory
            self.ps: vaspwfc = None
            self.ae: vasp_ae_wfc = None
            self.atoms = None
            self.C_by_k: Dict[int, np.ndarray] = {}
            self.kpoints: List[int] = []
            self.nspins: int = 1
            self.gamma_energies: Optional[np.ndarray] = None
            self.ch_per_atom: List[int] = []
            self.nproj_total: int = 0

    @staticmethod
    def read_fermi_from_doscar(directory: str) -> float:
        with open(os.path.join(directory, "DOSCAR"), "r") as f:
            return float(f.readlines()[5].split()[3])

    @staticmethod
    def read_gamma_energies_from_eigenval(directory: str, nspins: int) -> np.ndarray:
        path = os.path.join(directory, "EIGENVAL")
        with open(path, "r") as f:
            lines = f.readlines()
        nk, nb = [int(x) for x in lines[5].split()[1:3]]
        idx = 6
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        idx += 1
        E = np.zeros((nspins, nb), float)
        for ib in range(nb):
            parts = lines[idx].split()
            E[0, ib] = float(parts[1])
            if nspins > 1:
                E[1, ib] = float(parts[2])
            idx += 1
        return E

    @staticmethod
    def _wrap_delta_frac(df: np.ndarray) -> np.ndarray:
        return df - np.round(df)

    @staticmethod
    def _pairwise_min_image_dists_frac(comp_frac: np.ndarray, full_frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
        df = comp_frac[:, None, :] - full_frac[None, :, :]
        df = RectangularTrueBlochMatcher._wrap_delta_frac(df)
        dcart = np.einsum("...j,ij->...i", df, cell)
        return np.linalg.norm(dcart, axis=-1)

    @staticmethod
    def map_atoms_by_coords(comp_atoms, full_atoms, tol=1e-3, check_species=True) -> np.ndarray:
        comp_frac = comp_atoms.get_scaled_positions(wrap=True)
        full_frac = full_atoms.get_scaled_positions(wrap=True)
        cell = full_atoms.cell.array
        comp_syms = comp_atoms.get_chemical_symbols()
        full_syms = full_atoms.get_chemical_symbols()
        Nc = len(comp_atoms)
        mapping = -np.ones(Nc, dtype=int)
        if check_species:
            from collections import defaultdict
            comp_by = defaultdict(list)
            full_by = defaultdict(list)
            for i, s in enumerate(comp_syms):
                comp_by[s].append(i)
            for j, s in enumerate(full_syms):
                full_by[s].append(j)
            for s, idx_c in comp_by.items():
                idx_f = full_by.get(s, [])
                if len(idx_f) < len(idx_c):
                    raise ValueError(f"Full has fewer {s} atoms")
                D = RectangularTrueBlochMatcher._pairwise_min_image_dists_frac(
                    comp_frac[idx_c], full_frac[idx_f], cell
                )
                big = 1e6
                C_masked = np.where(D > tol, big, D)
                rows, cols = linear_sum_assignment(C_masked)
                for r, c in zip(rows, cols):
                    if D[r, c] > tol:
                        raise ValueError(f"No {s} match within tol")
                    mapping[idx_c[r]] = idx_f[c]
        else:
            D = RectangularTrueBlochMatcher._pairwise_min_image_dists_frac(comp_frac, full_frac, cell)
            big = 1e6
            C_masked = np.where(D > tol, big, D)
            rows, cols = linear_sum_assignment(C_masked)
            for r, c in zip(rows, cols):
                if D[r, c] > tol:
                    raise ValueError("No match within tol")
                mapping[r] = c
        return mapping

    @staticmethod
    def proj_offsets_from_channels(ch_per_atom: Sequence[int]) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(ch_per_atom[:-1])))

    @staticmethod
    def build_T_injection(full_ch_per_atom, comp_ch_per_atom, atom_map_comp_to_full, phase: complex = 1.0):
        from scipy.sparse import coo_matrix

        full_ch = np.asarray(full_ch_per_atom, dtype=int)
        comp_ch = np.asarray(comp_ch_per_atom, dtype=int)
        off_full = RectangularTrueBlochMatcher.proj_offsets_from_channels(full_ch)
        off_comp = RectangularTrueBlochMatcher.proj_offsets_from_channels(comp_ch)

        rows: List[int] = []
        cols: List[int] = []
        data: List[complex] = []
        for a_comp, a_full in enumerate(atom_map_comp_to_full):
            ncf = int(full_ch[a_full])
            ncc = int(comp_ch[a_comp])
            if ncf != ncc:
                raise ValueError("Channel mismatch")
            rf0, cf0 = int(off_full[a_full]), int(off_comp[a_comp])
            for i in range(ncf):
                rows.append(rf0 + i)
                cols.append(cf0 + i)
                data.append(phase)

        return coo_matrix((data, (rows, cols)),
                          shape=(int(full_ch.sum()), int(comp_ch.sum()))).tocsr()

    @staticmethod
    def form_B_for_slice(ae: vasp_ae_wfc, Cslice: np.ndarray) -> np.ndarray:
        def build_vec(coeff_row):
            return np.asarray(ae.get_beta_njk(coeff_row), dtype=np.complex128)

        nb = Cslice.shape[0]
        B_rows = Parallel(n_jobs=-1, prefer="threads")(
            delayed(build_vec)(Cslice[ib, :]) for ib in range(nb)
        )
        return np.ascontiguousarray(B_rows)

    @staticmethod
    def lift_B(B_comp: np.ndarray, T) -> np.ndarray:
        return B_comp @ T.conj().T

    @staticmethod
    def build_whitener(Q, tol: Optional[float] = None):
        Qm = Q.toarray() if issparse(Q) else np.asarray(Q)
        Qm = 0.5 * (Qm + Qm.conj().T)
        w, U = np.linalg.eigh(Qm)
        if tol is None:
            tol = max(1e-10, 1e-8 * float(w.max() if w.size else 1.0))
        keep = (w > tol)
        if not np.any(keep):
            raise ValueError("Q has no positive eigenvalues; cannot build whitener.")
        W = U[:, keep] * np.sqrt(w[keep])[None, :]
        info = {
            "rank": int(keep.sum()),
            "nproj": int(Qm.shape[0]),
            "tol": float(tol),
            "wmin": float(w.min()) if w.size else 0.0,
            "wmax": float(w.max()) if w.size else 0.0,
        }
        return W, info

    @staticmethod
    def fuse_true_bloch_rr(C: np.ndarray, B: np.ndarray, W: np.ndarray):
        B_ortho = B @ W
        psi = np.hstack([C, B_ortho])
        norms = np.sqrt(np.einsum("ij,ij->i", psi.conj(), psi).real)
        psi /= norms[:, None]
        return psi, norms

    @staticmethod
    def save_true_bloch(directory, psi, norms):
        np.savez_compressed(os.path.join(directory, "true_blochstates.npz"),
                            psi=psi, norms=norms)

    @staticmethod
    def load_true_bloch(directory):
        path = os.path.join(directory, "true_blochstates.npz")
        if not os.path.exists(path):
            return None
        d = np.load(path)
        return d["psi"], d["norms"]

    def load_system(self, directory: str, name: str, k_index: int = 1) -> SystemData:
        sys = RectangularTrueBlochMatcher.SystemData(name, directory)
        wav = os.path.join(directory, "WAVECAR")
        sys.ps = vaspwfc(wav)
        sys.nspins = sys.ps._nspin

        # ─── read PW coefficients with 1-based indexing ─────────────────────
        rows = []
        for iband in range(1, sys.ps._nbands + 1):
            coeff = sys.ps.readBandCoeff(
                ispin=1,
                ikpt=k_index,
                iband=iband,    # valid: 1 ≤ iband ≤ sys.ps._nbands
                norm=False
            )
            rows.append(np.asarray(coeff, dtype=np.complex128))
        sys.C_by_k[k_index] = np.vstack(rows)
        sys.kpoints.append(k_index)

        # ─── atoms & AE data ───────────────────────────────────────────────
        sys.atoms = ase_read(os.path.join(directory, "POSCAR"))
        sys.ae = vasp_ae_wfc(
            sys.ps,
            poscar=os.path.join(directory, "POSCAR"),
            potcar=os.path.join(directory, "POTCAR")
        )
        elem_idx = sys.ae._element_idx
        sys.ch_per_atom = [sys.ae._pawpp[it].lmmax for it in elem_idx]
        sys.nproj_total = int(sum(sys.ch_per_atom))

        sys.gamma_energies = self.read_gamma_energies_from_eigenval(directory,
                                                                    sys.nspins)
        return sys

    def run(self, output_path: Optional[str] = None):
        k_index = self.k_index
        skip_build = False

        # ─── EARLY cache pre‐check ─────────────────────────────────────────
        if self.reuse_cached:
            print("[STEP] Early attempt to load cached true Bloch states…")
            loaded_f = self.load_true_bloch(self.full_dir)
            loaded_s = self.load_true_bloch(self.simple_dir)
            loaded_m = self.load_true_bloch(self.metal_dir)
            if loaded_f and loaded_s and loaded_m:
                psi_f, norms_f = loaded_f
                psi_s, norms_s = loaded_s
                psi_m, norms_m = loaded_m
                print("[CACHE] Cache valid; skipping all heavy I/O and projection steps.")
                skip_build = True
            else:
                print("[CACHE] Cache incomplete; performing full rebuild.")

        # ─── HEAVY SETUP & BUILD (only if skip_build is False) ───────────
        if not skip_build:
            print(f"[STEP] Loading systems for k-point {k_index}…")
            self.sys_s = self.load_system(self.simple_dir, "simple", k_index)
            self.sys_m = self.load_system(self.metal_dir, "metal", k_index)
            self.sys_f = self.load_system(self.full_dir, "full", k_index)

            print("[STEP] Mapping atoms between component and full systems…")
            map_s_to_f = self.map_atoms_by_coords(
                self.sys_s.atoms, self.sys_f.atoms,
                tol=self.tol_map, check_species=self.check_species
            )
            map_m_to_f = self.map_atoms_by_coords(
                self.sys_m.atoms, self.sys_f.atoms,
                tol=self.tol_map, check_species=self.check_species
            )

            print("[STEP] Building injection matrices…")
            T_s = self.build_T_injection(
                self.sys_f.ch_per_atom, self.sys_s.ch_per_atom, map_s_to_f
            )
            T_m = self.build_T_injection(
                self.sys_f.ch_per_atom, self.sys_m.ch_per_atom, map_m_to_f
            )

            print("[STEP] Extracting PW coefficients…")
            Cf_all   = self.sys_f.C_by_k[k_index]
            Cc_s_all = self.sys_s.C_by_k[k_index]
            Cc_m_all = self.sys_m.C_by_k[k_index]
            # always build from the full arrays
            Cf   = Cf_all
            Cc_s = Cc_s_all
            Cc_m = Cc_m_all



            print("[STEP] Retrieving and symmetrizing Q matrix…")
            Q = self.sys_f.ae.get_qijs()
            Q = 0.5 * (Q + (Q.getH() if issparse(Q) else Q.conj().T))

            print("[STEP] Building reduced-rank whitener…")
            W, qinfo = self.build_whitener(Q)
            rank = qinfo["rank"]
            print(f"[STEP] Whitener built: rank={rank}/{qinfo['nproj']}")

            if self.reuse_cached:
                print("[STEP] Validating cached shapes…")
                valid = True
                for loaded, Cslice in [(loaded_f, Cf), (loaded_s, Cc_s), (loaded_m, Cc_m)]:
                    if not loaded:
                        valid = False
                        break
                    psi_cached, _ = loaded
                    if psi_cached.shape[1] != Cslice.shape[1] + rank:
                        valid = False
                        break
                if valid:
                    print("[CACHE] Shapes OK; skipping AE/PW fusion.")
                    skip_build = True
                else:
                    print("[CACHE] Shape mismatch; rebuilding AE/PW fusion.")

            if not skip_build:
                print("[STEP] Parallel fusing PW + AE into true Bloch coords…")
                from scipy.sparse import identity
                nproj_full = sum(self.sys_f.ch_per_atom)
                T_full = identity(nproj_full, format="csr")

                jobs = [
                    (self, self.sys_f, Cf,   T_full, self.full_dir),
                    (self, self.sys_s, Cc_s, T_s,    self.simple_dir),
                    (self, self.sys_m, Cc_m, T_m,    self.metal_dir),
                ]
                psi_f, psi_s, psi_m = Parallel(n_jobs=3, prefer="threads")(
                    delayed(_process_component)(matcher, sys_data, Cslice, T, out_dir, W)
                    for matcher, sys_data, Cslice, T, out_dir in jobs
                )


        # ─── 3) ADJUST GAMMA‐POINT ENERGIES ────────────────────────────────
        print("[STEP] Adjusting energies (Fermi + optional full↔metal‐min)…")
        
        # read Fermi shifts
        ef_full  = self.read_fermi_from_doscar(self.full_dir)
        ef_metal = self.read_fermi_from_doscar(self.metal_dir)
        
        # load or reuse cached gamma‐point energies
        e_f = (self.sys_f.gamma_energies.copy()
               if not skip_build else
               self.read_gamma_energies_from_eigenval(self.full_dir, 1))
        e_m = (self.sys_m.gamma_energies.copy()
               if not skip_build else
               self.read_gamma_energies_from_eigenval(self.metal_dir, 1))
        e_s = (self.sys_s.gamma_energies.copy()
               if not skip_build else
               self.read_gamma_energies_from_eigenval(self.simple_dir, 1))
        
        # subtract each system’s Fermi
        e_f -= ef_full
        e_m -= ef_metal
        
        # optional: align full‐state band N to the metal‐system minimum
        if self.align_full_to_metal_min_band is not None:
            fb = int(self.align_full_to_metal_min_band) - 1
            # chosen full‐band energy
            full_val = float(e_f[fb] if e_f.ndim == 1 else e_f[0, fb])
            # metal‐system minimum
            metal_min = float(np.min(e_m) if e_m.ndim == 1 else np.min(e_m[0]))
            delta = full_val - metal_min
            e_f -= delta
            print(
                f"[ALIGN] Shifted full E by {-delta:+.3f} eV so band {fb+1} "
                f"({full_val:+.3f} eV) lines up with metal min {metal_min:+.3f} eV"
            )
        
        # finally realign simple‐system minimum to full‐system minimum
        if e_s.ndim == 1:
            sf_min, ff_min = e_s.min(), e_f.min()
        else:
            sf_min, ff_min = e_s[0].min(), e_f[0].min()
        e_s -= (sf_min - ff_min)

        # ─── 4) POST‐BUILD SLICING ─────────────────────────────────────────
        if self.band_window_full is not None:
            psi_f = psi_f[self.band_window_full, :]
            if e_f.ndim == 2:
                e_f = e_f[:, self.band_window_full]
            else:
                e_f = e_f[self.band_window_full]

        if self.band_window_simple is not None:
            psi_s = psi_s[self.band_window_simple, :]
            if e_s.ndim == 2:
                e_s = e_s[:, self.band_window_simple]
            else:
                e_s = e_s[self.band_window_simple]

        if self.band_window_metal is not None:
            psi_m = psi_m[self.band_window_metal, :]
            if e_m.ndim == 2:
                e_m = e_m[:, self.band_window_metal]
            else:
                e_m = e_m[self.band_window_metal]

        # ─── 5) CALCULATE OVERLAPS ─────────────────────────────────────────
        print("[STEP] Calculating overlaps (simple↔full & metal↔full)…")
        S_sf = psi_s @ psi_f.conj().T
        S_mf = psi_m @ psi_f.conj().T

        # ─── 6) SCAN BANDS & RECORD MATCHES (unchanged) ───────────────────
        def _process_band(j):
            band_rows, ov_entries = [], []

            # SIMPLE→FULL
            if S_sf.shape[0] > 0:
                mags_s    = np.abs(S_sf[:, j])**2
                i_s_loc   = int(np.argmax(mags_s))
                ov_s_best = float(mags_s[i_s_loc])
                idx_s     = i_s_loc + 1
                Es        = float(e_s[0, idx_s-1]) if e_s.ndim==2 else float(e_s[idx_s-1])
                dEs       = float(e_f[0, j]   - Es   ) if e_f.ndim==2 else float(e_f[j]   - Es   )
            else:
                idx_s, Es, dEs, ov_s_best = -1, 0.0, 0.0, 0.0

            # METAL→FULL
            if S_mf.shape[0] > 0:
                mags_m    = np.abs(S_mf[:, j])**2
                i_m_loc   = int(np.argmax(mags_m))
                ov_m_best = float(mags_m[i_m_loc])
                idx_m     = i_m_loc + 1
                Em        = float(e_m[0, idx_m-1]) if e_m.ndim==2 else float(e_m[idx_m-1])
                dEm       = float(e_f[0, j]   - Em   ) if e_f.ndim==2 else float(e_f[j]   - Em   )
            else:
                idx_m, Em, dEm, ov_m_best = -1, 0.0, 0.0, 0.0

            w_span_s = float(np.sum(np.abs(S_sf[:, j])**2)) if S_sf.size else 0.0
            w_span_m = float(np.sum(np.abs(S_mf[:, j])**2)) if S_mf.size else 0.0
            residual = float(np.clip(1.0 - (w_span_s + w_span_m), 0.0, 1.0))

            band_rows.append({
                "full_idx": j+1,
                "E_full":   float(e_f[0,j]) if e_f.ndim==2 else float(e_f[j]),
                "simple":   dict(idx=idx_s, E=Es,  dE=dEs,  ov_best=ov_s_best, w_span=w_span_s),
                "metal":    dict(idx=idx_m, E=Em,  dE=dEm,  ov_best=ov_m_best, w_span=w_span_m),
                "residual": residual
            })

            # record every overlap entry
            if S_sf.shape[0] > 0:
                for i, z in enumerate(S_sf[:, j]):
                    ov = float(np.abs(z)**2)
                    Es_i = float(e_s[0,i]) if e_s.ndim==2 else float(e_s[i])
                    dEs_i= float(e_f[0,j]-Es_i) if e_f.ndim==2 else float(e_f[j]-Es_i)
                    ov_entries.append(("simple", j+1, i+1, Es_i, dEs_i, ov, w_span_s))
            if S_mf.shape[0] > 0:
                for i, z in enumerate(S_mf[:, j]):
                    ov = float(np.abs(z)**2)
                    Em_i = float(e_m[0,i]) if e_m.ndim==2 else float(e_m[i])
                    dEm_i= float(e_f[0,j]-Em_i) if e_f.ndim==2 else float(e_f[j]-Em_i)
                    ov_entries.append(("metal", j+1, i+1, Em_i, dEm_i, ov, w_span_m))

            return band_rows, ov_entries

        print("[STEP] Scanning bands to identify best matches…")
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_process_band)(j) for j in range(psi_f.shape[0])
        )
        rows, ov_all_lines = [], []
        for band_rows, ov_entries in results:
            rows.extend(band_rows)
            ov_all_lines.extend(ov_entries)

        print("[STEP] Writing match tables to disk…")
        if output_path is None:
            output_path = os.path.join(self.full_dir, "band_matches_rectangular.txt")
        with open(output_path, "w") as f:
            f.write("# full_idx  E_full  |  simple_idx  E_s  dE_s  ov_best_s  w_span_s  |  "
                    "metal_idx  E_m  dE_m  ov_best_m  w_span_m  |  residual\n")
            for r in rows:
                s, m = r["simple"], r["metal"]
                f.write(f"{r['full_idx']:6d}  {r['E_full']:10.3f}  |  "
                        f"{s['idx']:6d}  {s['E']:9.3f}  {s['dE']:9.3f}  {s['ov_best']:8.5f}  {s['w_span']:8.5f}  |  "
                        f"{m['idx']:6d}  {m['E']:9.3f}  {m['dE']:9.3f}  {m['ov_best']:8.5f}  {m['w_span']:8.5f}  |  "
                        f"{r['residual']:8.5f}\n")
        print(f"[DONE] Wrote {len(rows)} rows to {output_path}")

        # … (the rest of your all‐overlaps dump, filtered diagnostics, heatmap & color plot) …

        ov_all_path = os.path.join(self.full_dir, "band_matches_rectangular_all.txt")
        with open(ov_all_path, "w") as f2:
            f2.write("# component  full_idx  comp_idx  E_comp  dE_comp  ov  w_span_comp\n")
            from collections import defaultdict
            per_band = defaultdict(lambda: {"simple": [], "metal": []})
            for comp, full_idx, comp_idx, E_comp, dE_comp, ov, w_span in ov_all_lines:
                per_band[full_idx][comp.strip()].append((comp_idx, E_comp, dE_comp, ov, w_span))
            for full_idx in sorted(per_band.keys()):
                for comp in ("simple", "metal"):
                    lst = per_band[full_idx][comp]
                    lst.sort(key=lambda x: x[3], reverse=True)
                    for comp_idx, E_comp, dE_comp, ov, w_span in lst:
                        f2.write(f"{comp:7s}  {full_idx:6d}  {comp_idx:6d}  "
                                 f"{E_comp:9.3f}  {dE_comp:9.3f}  {ov:8.5f}  {w_span:8.5f}\n")
        print(f"[DONE] Wrote ov_all data to {ov_all_path}")

        diag3_path = os.path.join(self.full_dir, "band_matches_rectangular_filtered.txt")
        with open(diag3_path, "w") as f3:
            f3.write("# full_idx  E_full  ||  SIMPLE component states (idx, E, dE, ov)  ||  METAL component states (idx, E, dE, ov)\n")
            for r in rows:
                full_idx = r['full_idx']
                E_full = r['E_full']
                sim_list = [x for x in per_band.get(full_idx, {}).get('simple', []) if round(x[3], 5) > 0]
                met_list = [x for x in per_band.get(full_idx, {}).get('metal', []) if round(x[3], 5) > 0]
                if not sim_list and not met_list:
                    continue
                sim_list.sort(key=lambda x: x[3], reverse=True)
                met_list.sort(key=lambda x: x[3], reverse=True)
                max_rows = max(len(sim_list), len(met_list))
                f3.write(f"\n# Full band {full_idx:4d}  E_full={E_full:9.3f}\n")
                f3.write(f"{'':15s}  {'SIMPLE':<45s}  {'METAL':<45s}\n")
                f3.write(f"{'':15s}  {'idx   E_comp     dE_comp   ov':<45s}  "
                         f"{'idx   E_comp     dE_comp   ov':<45s}\n")
                for i in range(max_rows):
                    sim_str = ""; met_str = ""
                    if i < len(sim_list):
                        si, se, sde, sov, _ = sim_list[i]
                        sim_str = f"{si:4d}  {se:9.3f}  {sde:9.3f}  {sov:.5f}"
                    if i < len(met_list):
                        mi, me, mde, mov, _ = met_list[i]
                        met_str = f"{mi:4d}  {me:9.3f}  {mde:9.3f}  {mov:.5f}"
                    f3.write(f"{'':15s}  {sim_str:<45s}  {met_str:<45s}\n")
        print(f"[DIAG] Wrote filtered overlaps to {diag3_path}")

        print("[STEP] Generating overlap heatmap…")
        try:
            import seaborn as sns
            M_sm = psi_s @ psi_m.conj().T
            plt.figure(figsize=(8, 6))
            sns.heatmap(np.abs(M_sm), cmap="viridis", cbar_kws={'label': '|<simple|metal>|'})
            plt.title("Cross-component overlap heatmap (true Bloch states)")
            plt.xlabel("Metal component band index")
            plt.ylabel("Simple component band index")
            plt.tight_layout()
            heatmap_path = os.path.join(self.full_dir, "cross_component_overlap_heatmap.png")
            plt.savefig(heatmap_path, dpi=150)
            plt.close()
            print(f"[PLOT] Saved cross-component heatmap to {heatmap_path}")
        except Exception as e:
            print(f"[WARN] Could not plot cross-component heatmap: {e}")
"""
        cfg = PlotConfig(
            cmap_name_simple="managua_r",
            cmap_name_metal="vanimo_r",
            center_simple=40,
            center_metal=602,
            power_simple_neg=0.25,
            power_simple_pos=0.75,
            power_metal_neg=0.075,
            power_metal_pos=0.075,
            pick_primary=False,
            min_simple_wspan=0.01,
            energy_range=(-25, 10),
        )
        RectAEPAWColorPlotter(cfg).plot(output_path)
        plt.show()
        print(f"[PLOT] Generated color plot from '{output_path}'")
"""

# ---------------------------
# Thin wrapper
# ---------------------------

def run_match(
    simple_dir: str,
    metal_dir: str,
    full_dir: str,
    k_index: int = 1,
    tol_map: float = 1e-3,
    check_species: bool = True,
    band_window_simple: slice | None = None,
    band_window_metal: slice | None  = None,
    band_window_full: slice | None   = None,
    output_path: str | None          = None,
    reuse_cached: bool               = False
) -> str:
    """
    Build (or reload) true Bloch states and write out the band‐matches file.
    Returns the path to band_matches_rectangular.txt.
    """
    matcher = RectangularTrueBlochMatcher(
        simple_dir=simple_dir,
        metal_dir=metal_dir,
        full_dir=full_dir,
        k_index=k_index,
        tol_map=tol_map,
        check_species=check_species,
        band_window_simple=band_window_simple,
        band_window_metal=band_window_metal,
        band_window_full=band_window_full,
        reuse_cached=reuse_cached,
        align_full_to_metal_min_band=17   # <-- baked‐in alignment
    )
    matcher.run(output_path=output_path)

    # if user didn’t supply a path, use the default name in full_dir
    if output_path is None:
        output_path = os.path.join(full_dir, "band_matches_rectangular.txt")
    return output_path


if __name__ == "__main__":
    simple_dir = r"dir"
    metal_dir  = r"dir"
    full_dir   = r"dir"

    # 1) build or reload the true‐Bloch matches
    match_file = run_match(
        simple_dir=simple_dir,
        metal_dir=metal_dir,
        full_dir=full_dir,
        k_index=1,
        tol_map=1e-3,
        check_species=True,
        band_window_simple=slice(0, 42),
        band_window_metal=None,
        band_window_full=None,
        output_path=None,
        reuse_cached=True
    )
    print(f"[MATCH] Written matches to '{match_file}'")

    # 2) now plot using your preferred color config
    cfg = PlotConfig(
        cmap_name_simple="managua_r",
        cmap_name_metal="vanimo_r",
        center_simple=40,
        center_metal=602,
        power_simple_neg=0.25,
        power_simple_pos=0.75,
        power_metal_neg=0.075,
        power_metal_pos=0.075,
        pick_primary=False,
        min_simple_wspan=0.011,
        energy_range=(-25, 10),
    )

    fig, axes = RectAEPAWColorPlotter(cfg).plot(match_file)
    plt.show()
    print(f"[PLOT] Generated color plot from '{match_file}'")
