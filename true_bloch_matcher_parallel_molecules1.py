# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 12:19:18 2025

Patched to support multiple molecule directories as separate components in-memory,
write the original legacy full files (combined) from memory, write per-component
files for each molecule and for metal, and call the plotter at the end exactly
as the original did. No classifier is run inside this matcher (plotter handles it).
"""
import os
import numpy as np
from typing import Dict, List, Optional, Sequence
from scipy.sparse import issparse
from scipy.optimize import linear_sum_assignment
from ase.io import read as ase_read
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from collections import defaultdict

from vaspwfc import vaspwfc
from aewfc import vasp_ae_wfc
from true_bloch_plot_molecules import RectAEPAWColorPlotter, PlotConfig

def _process_component(matcher, sys_data, Cslice, T, out_dir, W):
    """
    Build AE projectors, lift into full-system space, fuse with PW,
    and save the resulting true Bloch states.
    """
    B_native = matcher.form_B_for_slice(sys_data.ae, Cslice)
    B_lifted = matcher.lift_B(B_native, T)
    psi, norms = matcher.fuse_true_bloch_rr(Cslice, B_lifted, W)
    matcher.save_true_bloch(out_dir, psi, norms)
    return psi

class RectangularTrueBlochMatcher:
    def __init__(self,
                 molecule_dirs: Sequence[str],
                 metal_dir: str,
                 full_dir: str,
                 k_index=1,
                 tol_map=1e-3,
                 check_species=True,
                 band_window_molecules: Optional[Sequence[Optional[slice]]] = None,
                 band_window_metal: Optional[slice] = None,
                 band_window_full: Optional[slice] = None,
                 reuse_cached=False,
                 align_full_to_metal_min_band: int | None = None):

        # multiple molecule directories (ordered)
        self.molecule_dirs = list(molecule_dirs) if isinstance(molecule_dirs, (list, tuple)) else [molecule_dirs]
        # molecule labels used in combined_all first column and combined main header
        self.mol_labels = [os.path.basename(os.path.normpath(d)) or f"molecule_{i+1}"
                           for i, d in enumerate(self.molecule_dirs)]

        self.metal_dir = metal_dir
        self.full_dir = full_dir
        self.k_index = k_index
        self.tol_map = tol_map
        self.check_species = check_species
        self.band_window_molecules = list(band_window_molecules) if band_window_molecules is not None else [None]*len(self.molecule_dirs)
        if len(self.band_window_molecules) != len(self.molecule_dirs):
            raise ValueError("band_window_molecules length must match molecule_dirs length")
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

        rows = []
        for iband in range(1, sys.ps._nbands + 1):
            coeff = sys.ps.readBandCoeff(
                ispin=1,
                ikpt=k_index,
                iband=iband,
                norm=False
            )
            rows.append(np.asarray(coeff, dtype=np.complex128))
        sys.C_by_k[k_index] = np.vstack(rows)
        sys.kpoints.append(k_index)

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

        # -- early cache pre-check
        if self.reuse_cached:
            print("[STEP] Early attempt to load cached true Bloch states…")
            loaded_f = self.load_true_bloch(self.full_dir)
            loaded_m = self.load_true_bloch(self.metal_dir)
            loaded_molls = [self.load_true_bloch(d) for d in self.molecule_dirs]
            if loaded_f and loaded_m and all(l is not None for l in loaded_molls):
                psi_f, norms_f = loaded_f
                psi_m, norms_m = loaded_m
                psi_molecules = [l[0] for l in loaded_molls]
                print("[CACHE] Cache valid; skipping heavy I/O and projection steps.")
                skip_build = True
            else:
                print("[CACHE] Cache incomplete; performing full rebuild.")

        if not skip_build:
            print(f"[STEP] Loading systems for k-point {k_index}…")
            self.sys_molecules = []
            for i, md in enumerate(self.molecule_dirs):
                sys = self.load_system(md, f"molecule_{i+1}", k_index)
                self.sys_molecules.append(sys)
            self.sys_m = self.load_system(self.metal_dir, "metal", k_index)
            self.sys_f = self.load_system(self.full_dir, "full", k_index)

            print("[STEP] Mapping atoms between each molecule and full system…")
            map_molecules_to_f = []
            for sys_comp in self.sys_molecules:
                mapping = self.map_atoms_by_coords(sys_comp.atoms, self.sys_f.atoms, tol=self.tol_map, check_species=self.check_species)
                map_molecules_to_f.append(mapping)
            map_m_to_f = self.map_atoms_by_coords(self.sys_m.atoms, self.sys_f.atoms, tol=self.tol_map, check_species=self.check_species)

            print("[STEP] Building injection matrices…")
            T_molecules = [self.build_T_injection(self.sys_f.ch_per_atom, sys_comp.ch_per_atom, mapping)
                           for sys_comp, mapping in zip(self.sys_molecules, map_molecules_to_f)]
            T_m = self.build_T_injection(self.sys_f.ch_per_atom, self.sys_m.ch_per_atom, map_m_to_f)

            print("[STEP] Extracting PW coefficients…")
            Cf = self.sys_f.C_by_k[k_index]
            Cc_molecules = [sys.C_by_k[k_index] for sys in self.sys_molecules]
            Cc_m = self.sys_m.C_by_k[k_index]

            print("[STEP] Retrieving and symmetrizing Q matrix…")
            Q = self.sys_f.ae.get_qijs()
            Q = 0.5 * (Q + (Q.getH() if issparse(Q) else Q.conj().T))

            print("[STEP] Building reduced-rank whitener…")
            W, qinfo = self.build_whitener(Q)
            rank = qinfo["rank"]
            print(f"[STEP] Whitener built: rank={rank}/{qinfo['nproj']}")

            print("[STEP] Parallel fusing PW + AE into true Bloch coords…")
            from scipy.sparse import identity
            nproj_full = sum(self.sys_f.ch_per_atom)
            T_full = identity(nproj_full, format="csr")

            jobs = [(self, self.sys_f, Cf, T_full, self.full_dir),
                    (self, self.sys_m, Cc_m, T_m, self.metal_dir)] + [
                        (self, sys_comp, Cc, T, out_dir)
                        for sys_comp, Cc, T, out_dir in zip(self.sys_molecules, Cc_molecules, T_molecules, self.molecule_dirs)
                    ]

            results = Parallel(n_jobs=len(jobs), prefer="threads")(
                delayed(_process_component)(matcher, sys_data, Cslice, T, out_dir, W)
                for matcher, sys_data, Cslice, T, out_dir in jobs
            )
            psi_f = results[0]
            psi_m = results[1]
            psi_molecules = results[2:]

        # adjust energies...
        ef_full = self.read_fermi_from_doscar(self.full_dir)
        ef_metal = self.read_fermi_from_doscar(self.metal_dir)
        e_f = (self.sys_f.gamma_energies.copy() if not skip_build else self.read_gamma_energies_from_eigenval(self.full_dir, 1))
        e_m = (self.sys_m.gamma_energies.copy() if not skip_build else self.read_gamma_energies_from_eigenval(self.metal_dir, 1))
        e_ms = []
        for md in self.molecule_dirs:
            e_mol = (self.read_gamma_energies_from_eigenval(md, 1) if skip_build else None)
            e_ms.append(e_mol)
        if not skip_build:
            e_ms = [sys.gamma_energies.copy() for sys in self.sys_molecules]

        e_f -= ef_full
        e_m -= ef_metal
        if self.align_full_to_metal_min_band is not None:
            fb = int(self.align_full_to_metal_min_band) - 1
            full_val = float(e_f[fb] if e_f.ndim == 1 else e_f[0, fb])
            metal_min = float(np.min(e_m) if e_m.ndim == 1 else np.min(e_m[0]))
            delta = full_val - metal_min
            e_f -= delta
            print(f"[ALIGN] Shifted full E by {-delta:+.3f} eV to align band {fb+1}")

        for i, e_s in enumerate(e_ms):
            if e_s is None:
                continue
            if e_s.ndim == 1:
                sf_min, ff_min = e_s.min(), e_f.min()
            else:
                sf_min, ff_min = e_s[0].min(), e_f[0].min()
            e_ms[i] = e_s - (sf_min - ff_min)

        # optional slicing
        if self.band_window_full is not None:
            psi_f = psi_f[self.band_window_full, :]
            if e_f.ndim == 2:
                e_f = e_f[:, self.band_window_full]
            else:
                e_f = e_f[self.band_window_full]
        if self.band_window_metal is not None:
            psi_m = psi_m[self.band_window_metal, :]
            if e_m.ndim == 2:
                e_m = e_m[:, self.band_window_metal]
            else:
                e_m = e_m[self.band_window_metal]
        for idx, bw in enumerate(self.band_window_molecules):
            if bw is not None:
                psi_molecules[idx] = psi_molecules[idx][bw, :]
                if e_ms[idx] is not None:
                    e_ms[idx] = (e_ms[idx][:, bw] if e_ms[idx].ndim == 2 else e_ms[idx][bw])

        # compute overlaps for each molecule and metal, then build combined outputs in memory
        print("[STEP] Calculating overlaps (components -> full)...")
        S_components = [psi @ psi_f.conj().T for psi in psi_molecules] + [psi_m @ psi_f.conj().T]
        comp_labels = self.mol_labels + ["metal"]

        print("[STEP] Scanning bands to identify best matches...")
        n_full = psi_f.shape[0]
        rows = []
        ov_all_lines = []  # tuples: (label, full_idx, comp_idx, E_comp, dE_comp, ov, w_span_comp)

        for j in range(n_full):
            # for each full band compute each component's best and record entries (exact same logic as original)
            simple_comp = None
            metal_comp = None
            # process molecules in order; for writing per-molecule files later we'll need per-mol bests
            mol_bests = []
            for mi, label in enumerate(self.mol_labels):
                S = S_components[mi]
                if S.size:
                    mags = np.abs(S[:, j])**2
                    i_best = int(np.argmax(mags))
                    ov_best = float(mags[i_best])
                    w_span = float(mags.sum())
                    E_comp = float(e_ms[mi][0, i_best]) if e_ms[mi] is not None and e_ms[mi].ndim == 2 else (float(e_ms[mi][i_best]) if e_ms[mi] is not None else 0.0)
                    dE = float((e_f[0,j] if getattr(e_f, "ndim", 1)==2 else e_f[j]) - E_comp)
                    mol_bests.append((i_best+1, E_comp, dE, ov_best, w_span))
                    # record all overlaps for this molecule -> full j
                    for i_pair, z in enumerate(S[:, j]):
                        ov = float(np.abs(z)**2)
                        Epair = float(e_ms[mi][0, i_pair]) if e_ms[mi] is not None and e_ms[mi].ndim == 2 else (float(e_ms[mi][i_pair]) if e_ms[mi] is not None else 0.0)
                        ov_all_lines.append((label, j+1, i_pair+1, Epair, float((e_f[0,j] if getattr(e_f,'ndim',1)==2 else e_f[j]) - Epair), ov, w_span))
                else:
                    mol_bests.append((-1, 0.0, 0.0, 0.0, 0.0))
            # metal best
            S_m = S_components[-1]
            if S_m.size:
                mags_m = np.abs(S_m[:, j])**2
                i_m_best = int(np.argmax(mags_m))
                ov_m_best = float(mags_m[i_m_best])
                w_span_m = float(mags_m.sum())
                E_mbest = float(e_m[0, i_m_best]) if e_m.ndim == 2 else float(e_m[i_m_best])
                dEm = float((e_f[0,j] if getattr(e_f,'ndim',1)==2 else e_f[j]) - E_mbest)
                # record metal all-pairs
                for i_pair, z in enumerate(S_m[:, j]):
                    ov = float(np.abs(z)**2)
                    Epair = float(e_m[0, i_pair]) if e_m.ndim == 2 else float(e_m[i_pair])
                    ov_all_lines.append(("metal", j+1, i_pair+1, Epair, float((e_f[0,j] if getattr(e_f,'ndim',1)==2 else e_f[j]) - Epair), ov, w_span_m))
            else:
                i_m_best, ov_m_best, w_span_m, E_mbest, dEm = -1, 0.0, 0.0, 0.0, 0.0

            # build combined row with molecule blocks in order then metal block
            # for legacy combined main we need one "simple" column (we choose first molecule as simple analog)
            row_simple = mol_bests[0] if len(mol_bests) > 0 else (-1,0.0,0.0,0.0,0.0)
            row_metal = (i_m_best, E_mbest, dEm, ov_m_best, w_span_m)
            Efull = (e_f[0,j] if getattr(e_f,'ndim',1)==2 else e_f[j])
            rows.append({
                "full_idx": j+1,
                "E_full": float(Efull),
                "simple": dict(idx=int(row_simple[0]), E=float(row_simple[1]), dE=float(row_simple[2]),
                               ov_best=float(row_simple[3]), w_span=float(row_simple[4])),
                "metal": dict(idx=int(row_metal[0]), E=float(row_metal[1]), dE=float(row_metal[2]),
                              ov_best=float(row_metal[3]), w_span=float(row_metal[4])),
                "residual": float(np.clip(1.0 - (sum(r[4] for r in mol_bests) + w_span_m), 0.0, 1.0)),
                "molecules": mol_bests  # keep per-molecule bests for per-molecule files
            })

        # Write per-component molecule files (same format as original simple output)
        for mi, mol_dir in enumerate(self.molecule_dirs):
            mol_label = self.mol_labels[mi]
            os.makedirs(mol_dir, exist_ok=True)
            out_main = os.path.join(mol_dir, f"band_matches_{mol_label}.txt")
            out_all  = os.path.join(mol_dir, f"band_matches_{mol_label}_all.txt")
            with open(out_main, "w") as f:
                f.write("# full_idx  E_full  |  simple_idx  E_s  dE_s  ov_best_s  w_span_s  |  metal_idx  E_m  dE_m  ov_best_m  w_span_m  |  residual\n")
                for r in rows:
                    comp = r["molecules"][mi]
                    s_idx, s_E, s_dE, s_ov_best, s_wspan = comp
                    m = r["metal"]
                    f.write(f"{r['full_idx']:6d}  {r['E_full']:10.3f}  |  "
                            f"{int(s_idx):6d}  {s_E:9.3f}  {s_dE:9.3f}  {s_ov_best:8.5f}  {s_wspan:8.5f}  |  "
                            f"{m['idx']:6d}  {m['E']:9.3f}  {m['dE']:9.3f}  {m['ov_best']:8.5f}  {m['w_span']:8.5f}  |  "
                            f"{r['residual']:8.5f}\n")
            with open(out_all, "w") as f:
                f.write("# component  full_idx  comp_idx  E_comp  dE_comp  ov  w_span_comp\n")
                for comp_label, full_idx, comp_idx, Ecomp, dEcomp, ov, w_span in ov_all_lines:
                    if comp_label == mol_label:
                        f.write(f"{comp_label:15s}  {full_idx:6d}  {comp_idx:6d}  {Ecomp:9.3f}  {dEcomp:9.3f}  {ov:8.5f}  {w_span:8.5f}\n")

        # Write metal files (original names) into metal_dir
        os.makedirs(self.metal_dir, exist_ok=True)
        metal_main = os.path.join(self.metal_dir, "band_matches_metal.txt")
        metal_all  = os.path.join(self.metal_dir, "band_matches_metal_all.txt")
        with open(metal_main, "w") as f:
            f.write("# full_idx  E_full  |  simple_idx  E_s  dE_s  ov_best_s  w_span_s  |  metal_idx  E_m  dE_m  ov_best_m  w_span_m  |  residual\n")
            for r in rows:
                s = r["simple"]
                m = r["metal"]
                f.write(f"{r['full_idx']:6d}  {r['E_full']:10.3f}  |  "
                        f"{s['idx']:6d}  {s['E']:9.3f}  {s['dE']:9.3f}  {s['ov_best']:8.5f}  {s['w_span']:8.5f}  |  "
                        f"{m['idx']:6d}  {m['E']:9.3f}  {m['dE']:9.3f}  {m['ov_best']:8.5f}  {m['w_span']:8.5f}  |  "
                        f"{r['residual']:8.5f}\n")
        with open(metal_all, "w") as f:
            f.write("# component  full_idx  comp_idx  E_comp  dE_comp  ov  w_span_comp\n")
            for comp_label, full_idx, comp_idx, Ecomp, dEcomp, ov, w_span in ov_all_lines:
                if comp_label == "metal":
                    f.write(f"{comp_label:15s}  {full_idx:6d}  {comp_idx:6d}  {Ecomp:9.3f}  {dEcomp:9.3f}  {ov:8.5f}  {w_span:8.5f}\n")

        # Combined legacy files (written once, from memory). No copies into molecule dirs.
        os.makedirs(self.full_dir, exist_ok=True)
        combined_main = os.path.join(self.full_dir, "band_matches_rectangular.txt")
        combined_all  = os.path.join(self.full_dir, "band_matches_rectangular_all.txt")

        # main header: molecule blocks in order then metal
        with open(combined_main, "w") as f:
            mol_headers = []
            for label in self.mol_labels:
                mol_headers.extend([f"{label}_idx", f"{label}_E", f"{label}_dE", f"{label}_ov_best", f"{label}_w_span"])
            header = "# full_idx  E_full  |  " + "  ".join(f"{h:>12s}" for h in mol_headers) + "  |  metal_idx  E_m  dE_m  ov_best_m  w_span_m  |  residual\n"
            f.write(header)
            for r in rows:
                parts = [f"{r['full_idx']:6d}", f"{r['E_full']:10.3f}", "|"]
                for mi in range(len(self.mol_labels)):
                    comp = r["molecules"][mi]
                    parts.extend([
                        f"{int(comp[0]):6d}",
                        f"{float(comp[1]):9.3f}",
                        f"{float(comp[2]):9.3f}",
                        f"{float(comp[3]):8.5f}",
                        f"{float(comp[4]):8.5f}"
                    ])
                m = r["metal"]
                parts.extend([
                    f"{int(m['idx']):6d}",
                    f"{float(m['E']):9.3f}",
                    f"{float(m['dE']):9.3f}",
                    f"{float(m['ov_best']):8.5f}",
                    f"{float(m['w_span']):8.5f}"
                ])
                parts.append(f"{r.get('residual',0.0):8.5f}")
                f.write("  ".join(parts) + "\n")

        # combined_all: write per-pair lines grouped by full_idx; first column is component label
        with open(combined_all, "w") as f:
            f.write("# component  full_idx  comp_idx  E_comp  dE_comp  ov  w_span_comp\n")
            for comp_label, full_idx, comp_idx, Ecomp, dEcomp, ov, w_span in ov_all_lines:
                f.write(f"{comp_label:15s}  {full_idx:6d}  {comp_idx:6d}  {Ecomp:9.3f}  {dEcomp:9.3f}  {ov:8.5f}  {w_span:8.5f}\n")

        print(f"[DONE] Wrote per-component files and combined legacy files to {self.full_dir}")

        # Call plotter exactly as original did, on the combined main file (plotter will run classifier)
        try:
            cfg = PlotConfig(
                cmap_name_simple="managua_r",
                cmap_name_metal="vanimo_r",
                center_simple=40,
                center_metal=1065,
                power_simple_neg=0.25,
                power_simple_pos=0.75,
                power_metal_neg=0.075,
                power_metal_pos=0.075,
                pick_primary=False,
                min_simple_wspan=0.01,
                energy_range=(-25, 10),
            )
            RectAEPAWColorPlotter(cfg).plot(combined_main)
            plt.show()
            print(f"[PLOT] Generated color plot from '{combined_main}'")
        except Exception as e:
            print(f"[WARN] Plotter call failed: {e}")

# thin wrapper
def run_match(
    molecule_dirs, metal_dir, full_dir,
    k_index=1, tol_map=1e-3, check_species=True,
    band_window_molecules=None, band_window_metal=None, band_window_full=None,
    output_path=None, reuse_cached=False
):
    matcher = RectangularTrueBlochMatcher(
        molecule_dirs=molecule_dirs,
        metal_dir=metal_dir,
        full_dir=full_dir,
        k_index=k_index,
        tol_map=tol_map,
        check_species=check_species,
        band_window_molecules=band_window_molecules,
        band_window_metal=band_window_metal,
        band_window_full=band_window_full,
        reuse_cached=reuse_cached,
        align_full_to_metal_min_band=33
    )
    matcher.run(output_path=output_path)

    # Build a list of the meaningful match files the matcher writes
    written = []

    # combined legacy files (full_dir)
    combined_main = os.path.join(full_dir, "band_matches_rectangular.txt")
    combined_all  = os.path.join(full_dir, "band_matches_rectangular_all.txt")
    written.extend([combined_main, combined_all])

    # metal files
    metal_main = os.path.join(metal_dir, "band_matches_metal.txt")
    metal_all  = os.path.join(metal_dir, "band_matches_metal_all.txt")
    written.extend([metal_main, metal_all])

    # per-molecule files
    for md in molecule_dirs:
        label = os.path.basename(os.path.normpath(md)) or md
        mol_main = os.path.join(md, f"band_matches_{label}.txt")
        mol_all  = os.path.join(md, f"band_matches_{label}_all.txt")
        written.extend([mol_main, mol_all])

    # return list of files (caller should iterate this list)
    return written

if __name__ == "__main__":
    molecule_dirs = [
        r'dir',
        r'dir'
    ]
    metal_dir  = r'dir'
    full_dir   = r'dir'

    match_files = run_match(
        molecule_dirs=molecule_dirs,
        metal_dir=metal_dir,
        full_dir=full_dir,
        k_index=1,
        tol_map=1e-3,
        check_species=True,
        band_window_molecules = [slice(0, 42),slice(0, 42)],
        band_window_metal=None,
        band_window_full=None,
        reuse_cached=True
    )
    for mf in match_files:

        print(f"[MATCH] Written matches to '{mf}'")
