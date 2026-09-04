
"""
pharmacophore.py
----------------
Pipeline:
  1. Read all molecules in a directory (*.xyz + *.dens.cube + *.pote.cube)
  2. Align every molecule to the first (alphabetically) via principal-axes
     pre-alignment + Kabsch RMSD on heavy atoms
  3. Apply the same rotation/translation to the density/ESP isosurface points
  4. Coarse-grid the aligned ESP isosurface (default 1.0 Angstrom cells)
  5. Average ESP per cell; flag high-variability cells (std > threshold)
  6. Write aligned *.xyz files and a PyMOL CGO pharmacophore script
 
Usage:
    python pharmacophore.py <directory>
                            [--isovalue   0.001]
                            [--grid       1.0  ]
                            [--esp_pos    0.02 ]
                            [--esp_neg   -0.02 ]
                            [--esp_var    0.01 ]
                            [--outdir     <dir> ]
 
Colour convention (CGO spheres):
    red    ESP mean >  esp_pos             (positive / electron-poor)
    blue   ESP mean <  esp_neg             (negative / electron-rich)
    green  |ESP mean| <= |threshold|       (hydrophobic / neutral)
    purple std(ESP)   >  esp_var           (high variability across molecules)
"""
 
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
 
BOHR = 0.529177210903          # Bohr to Angstrom
 
MASSES = {
    'H': 1.008,  'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'S': 32.06,  'P': 30.974, 'CL': 35.45,
    'BR': 79.904,'I': 126.90, 'B': 10.81,  'SI': 28.085,
}
 
ELEMENT_SYMBOLS = {
    1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O',
    9:'F', 10:'Ne', 11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P',
    16:'S', 17:'Cl', 18:'Ar', 19:'K', 20:'Ca', 26:'Fe', 29:'Cu',
    30:'Zn', 35:'Br', 53:'I',
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FILE READERS
# ─────────────────────────────────────────────────────────────────────────────
 
def read_cube(path):
    """
    Parse a Gaussian cube file.
    Returns:
        atoms  : list of (symbol, x, y, z) in Angstrom
        origin : (3,) array in Angstrom
        voxel  : (3, 3) array, rows are voxel step vectors in Angstrom
        data   : (nx, ny, nz) float array of volumetric values
    """
    with open(path) as fh:
        lines = fh.readlines()
 
    # line 3: natoms + origin
    tok = lines[2].split()
    natoms = abs(int(tok[0]))          # negative means multiple MOs; take abs
    origin = np.array([float(v) for v in tok[1:4]]) * BOHR
 
    # lines 4-6: grid dimensions + step vectors
    def parse_grid_line(line):
        parts = line.split()
        n = int(parts[0])
        vec = np.array([float(v) for v in parts[1:4]]) * BOHR
        return n, vec
 
    nx, xv = parse_grid_line(lines[3])
    ny, yv = parse_grid_line(lines[4])
    nz, zv = parse_grid_line(lines[5])
    voxel = np.array([xv, yv, zv])    # shape (3, 3)
 
    # atom lines
    atoms = []
    for i in range(natoms):
        parts = lines[6 + i].split()
        atomic_num = int(parts[0])
        sym = ELEMENT_SYMBOLS.get(atomic_num, f'X{atomic_num}')
        x, y, z = float(parts[2]) * BOHR, float(parts[3]) * BOHR, float(parts[4]) * BOHR
        atoms.append((sym, x, y, z))
 
    # volumetric data
    raw = []
    for line in lines[6 + natoms:]:
        raw.extend(line.split())
    data = np.array(raw, dtype=np.float64).reshape(nx, ny, nz)
 
    return atoms, origin, voxel, data
 
 
def read_xyz(path):
    """
    Parse an XYZ file (with or without the count/comment header).
    Returns list of (symbol, x, y, z).
    """
    atoms = []
    with open(path) as fh:
        lines = [l for l in fh if l.strip()]
 
    # detect header
    first = lines[0].split()
    start = 2 if (len(first) == 1 and first[0].isdigit()) else 0
 
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def pairwise_esp_rmsd(all_surf_coords, all_esp_vals, esp_weight=10.0):
    """
    Compute pairwise ESP surface RMSD between all molecules.
    esp_weight scales ESP differences relative to spatial distances (in Angstrom).
    Returns (n_mols x n_mols) symmetric matrix.
    """
    from scipy.spatial import cKDTree
    n = len(all_surf_coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            tree = cKDTree(all_surf_coords[j])
            dists, idxs = tree.query(all_surf_coords[i])
            esp_diff = all_esp_vals[i] - all_esp_vals[j][idxs]
            rmsd = float(np.sqrt(np.mean(dists**2 + (esp_weight * esp_diff)**2)))
            matrix[i, j] = rmsd
            matrix[j, i] = rmsd
    return matrix
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ISOSURFACE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_isosurface(dens_path, pote_path, isovalue=0.001):
    """
    Find voxels near the density isovalue; sample ESP at those voxels.
 
    Returns:
        coords   : (N, 3) float array, Angstrom
        esp_vals : (N,)   float array, a.u.
    """
    atoms, origin, voxel, density, = read_cube(dens_path)[:4]
    _,     _,      _,     esp      = read_cube(pote_path)[:4]
 
    nx, ny, nz = density.shape
 
    band = isovalue * 0.15
    lo, hi = isovalue - band, isovalue + band
    mask = (density >= lo) & (density <= hi)
    idxs = np.argwhere(mask)
 
    if len(idxs) == 0:
        # fallback: top 5 % of density values
        thr = np.percentile(density, 95)
        mask = density >= thr
        idxs = np.argwhere(mask)
        print(f"    [warn] no voxels at isovalue {isovalue:.4f}; "
              f"falling back to top-5% threshold {thr:.5f} - {len(idxs):,} pts")
 
    # convert voxel indices to Cartesian coordinates
    # coords = origin + i*xv + j*yv + k*zv
    i = idxs[:, 0:1].astype(float)
    j = idxs[:, 1:2].astype(float)
    k = idxs[:, 2:3].astype(float)
    coords = origin + i * voxel[0] + j * voxel[1] + k * voxel[2]
 
    esp_vals = esp[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
 
    print(f"    surface: {len(coords):,} pts  "
          f"ESP [{esp_vals.min():.4f}, {esp_vals.max():.4f}]")
    return coords, esp_vals
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ALIGNMENT  (principal axes + Kabsch)
# ─────────────────────────────────────────────────────────────────────────────
 
def atom_mass(sym):
    return MASSES.get(sym.upper(), 12.0)
 
 
def centre_of_mass(atoms):
    coords = np.array([[a[1], a[2], a[3]] for a in atoms])
    masses = np.array([atom_mass(a[0]) for a in atoms])
    return np.average(coords, weights=masses, axis=0)
 
 
def principal_axes_rotation(atoms):
    """
    Returns (centroid, R) where R (3×3, rows = principal axes) rotates
    the molecule so its principal axes align with x/y/z.
    """
    coords = np.array([[a[1], a[2], a[3]] for a in atoms])
    masses = np.array([atom_mass(a[0]) for a in atoms])
    ctr = np.average(coords, weights=masses, axis=0)
    r = coords - ctr
 
    I = np.zeros((3, 3))
    for m, ri in zip(masses, r):
        I[0, 0] += m * (ri[1]**2 + ri[2]**2)
        I[1, 1] += m * (ri[0]**2 + ri[2]**2)
        I[2, 2] += m * (ri[0]**2 + ri[1]**2)
        I[0, 1] -= m * ri[0] * ri[1]
        I[0, 2] -= m * ri[0] * ri[2]
        I[1, 2] -= m * ri[1] * ri[2]
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
 
    _, evecs = np.linalg.eigh(I)   # columns = eigenvectors, ascending eigenvalue
    R = evecs.T                     # rows = principal axes
    return ctr, R


def kabsch_iterative(mob, ref, n_iter=10, reject_threshold=2.0):
    weights = np.ones(len(mob))
    for _ in range(n_iter):
        # weighted centring
        ctr_mob = np.average(mob, weights=weights, axis=0)
        ctr_ref = np.average(ref, weights=weights, axis=0)
        mob_c = mob - ctr_mob
        ref_c = ref - ctr_ref
        # weighted Kabsch
        H = (mob_c * weights[:,None]).T @ ref_c
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1,1,d]) @ U.T
        # residuals
        residuals = np.sqrt(np.sum((mob_c @ R.T - ref_c)**2, axis=1))
        # downweight outliers
        weights = np.where(residuals > reject_threshold, 0.0, 1.0)
        if weights.sum() < 3:
            break
    return R, ctr_mob, ctr_ref
 
def kabsch_rotation(P, Q): #deprecated
    """
    Find the optimal rotation R that minimises RMSD between P and Q
    (both already centred at origin).
    Returns R such that  P @ R.T  ≈  Q.
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    # ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R
 
 
def heavy_coords(atoms):
    """Return (N, 3) array of non-hydrogen atom coordinates."""
    return np.array([[a[1], a[2], a[3]] for a in atoms if a[0].upper() != 'H'])
 
 
def align_to_reference(ref_atoms_pa, mob_atoms, mob_surf_coords):
    """
    Align mobile molecule to the reference (which is already in PA frame).
 
    Parameters
    ----------
    ref_atoms_pa   : (N, 3) heavy-atom coords of reference in PA frame
    mob_atoms      : list of (sym, x, y, z) for mobile molecule
    mob_surf_coords: (M, 3) isosurface coordinates of mobile molecule
 
    Returns
    -------
    atoms_aligned  : (N_all, 3) all-atom coords after alignment
    surf_aligned   : (M, 3) surface coords after alignment
    rmsd           : float
    """
    # Step 1 – centre + rotate mobile to its own principal axes
    ctr_mob, R_pa = principal_axes_rotation(mob_atoms)
    all_coords = np.array([[a[1], a[2], a[3]] for a in mob_atoms])
    all_pa = np.array((all_coords - ctr_mob) @ R_pa.T)
 
    # Step 2 – Kabsch fit of PA-aligned heavy atoms onto reference heavy atoms
    mob_heavy_pa = (heavy_coords(mob_atoms) - ctr_mob) @ R_pa.T
    n = min(len(ref_atoms_pa), len(mob_heavy_pa))
    R_kab, ctr_mob, ctr_ref = kabsch_iterative(mob_heavy_pa[:n], ref_atoms_pa[:n])
 
    # Step 3 – apply combined rotation to all atoms and surface
    atoms_aligned = np.array(all_pa @ R_kab.T)
    surf_aligned  = (mob_surf_coords - ctr_mob) @ R_pa.T @ R_kab.T
 
    # RMSD on the fitted heavy atoms
    diff = mob_heavy_pa[:n] @ R_kab.T - ref_atoms_pa[:n]
    rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
 
    return atoms_aligned, surf_aligned, rmsd


def cluster_molecules(rmsd_matrix, stems, threshold=None):
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import squareform
    
    linkage_matrix = linkage(squareform(rmsd_matrix), method='average')
    
    if threshold is None:
        # automatic threshold: largest gap in linkage distances
        gaps = np.diff(linkage_matrix[:, 2])
        threshold = linkage_matrix[np.argmax(gaps), 2]
        print(f"\nAuto threshold: {threshold:.4f}")
    
    labels = fcluster(linkage_matrix, threshold, criterion='distance')
    
    print("\nClusters:")
    for cluster_id in sorted(set(labels)):
        members = [stems[i] for i, l in enumerate(labels) if l == cluster_id]
        print(f"  Cluster {cluster_id}: {members}")
    
    return labels, linkage_matrix
 
 
# ─────────────────────────────────────────────────────────────────────────────
# COARSE GRID + PHARMACOPHORE
# ─────────────────────────────────────────────────────────────────────────────
 
def build_pharmacophore(all_surf_coords, all_esp_vals, all_act,
                        grid_spacing=1.0,
                        esp_pos=0.02, esp_neg=-0.02, esp_var=0.01):
    """
    Bin all aligned surface points into a regular grid, average ESP per cell,
    flag high-variability cells.
 
    Parameters
    ----------
    all_surf_coords : list of (M_i, 3) arrays  (one per molecule)
    all_esp_vals    : list of (M_i,) arrays
    grid_spacing    : cell edge length in Angstrom
    esp_pos / esp_neg : thresholds for positive / negative colouring
    esp_var         : std(ESP) threshold for high-variability (purple)
 
    Returns
    -------
    List of (cx, cy, cz, mean_esp, std_esp, n_mols, colour_rgb)
        where colour_rgb is a (r, g, b) tuple with values in [0, 1].
    """
    # pool all points
    all_coords = np.vstack(all_surf_coords)   # (Ntotal, 3)
    all_esp    = np.concatenate(all_esp_vals) # (Ntotal,)

    # assign each point to a grid cell
    cell_idx = np.floor(all_coords / grid_spacing).astype(int)
    cell_keys = [tuple(row) for row in cell_idx]
 
    # accumulate ESP values per cell
    mol_ids = np.concatenate([
    np.full(len(c), i) for i, c in enumerate(all_surf_coords)
    ])

    cell_data = defaultdict(lambda: defaultdict(list))
    for key, esp, mol_id in zip(cell_keys, all_esp, mol_ids):
        cell_data[key][mol_id].append(esp)
 
    # also track how many distinct molecules contributed
    # build molecule-id array (which molecule each point came from)
 
    pharmacophore = []

    for key, mol_dict in cell_data.items():
        mol_means = np.array([np.mean(v) for v in mol_dict.values()])
        weights = np.array([all_act[mid] for mid in mol_dict.keys()])
        weights_norm = weights / weights.sum()
        cx = (key[0] + 0.5) * grid_spacing
        cy = (key[1] + 0.5) * grid_spacing
        cz = (key[2] + 0.5) * grid_spacing
        mean_esp = float(np.average(mol_means, weights=weights_norm))
        std_esp  = float(np.sqrt(np.average((mol_means - mean_esp)**2, weights=weights_norm))) if len(mol_means) > 1 else 0.0
        n_mols   = len(mol_dict)
        coverage = weights.sum() / sum(all_act)
        radius_modifier = coverage * (1.0 - 0.7 * min(std_esp / esp_var, 1.0))
        

        if mean_esp > 0:
            t = min(mean_esp / esp_pos, 1.0)*coverage
            colour =  (1.0 - t, 1.0 - t, 1.0)   # white → blue
        else:
            t = min(abs(mean_esp) / abs(esp_neg), 1.0)*coverage
            colour = (1.0, 1.0 - t, 1.0 - t)   # white → red

 
        pharmacophore.append((cx, cy, cz, mean_esp, std_esp, n_mols, colour, radius_modifier))
 
    return pharmacophore
 
 
# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────
 
def write_aligned_xyz(path, sym_list, coords_aligned):
    """Write aligned coordinates to an XYZ file."""
    n = len(sym_list)
    with open(path, 'w') as fh:
        fh.write(f"{n}\naligned\n")
        for sym, (x, y, z) in zip(sym_list, coords_aligned):
            fh.write(f"{sym:4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
 
 
def write_pymol_cgo(path, pharmacophore, sphere_radius=0.4):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("# Pharmacophore - generated by pharmacophore.py\n")
        fh.write("from pymol import cmd\n\n")

        for i, (cx, cy, cz, mean_esp, std_esp, n_mols, (r, g, b), radius_modifier) in enumerate(pharmacophore, 1):
            radius = sphere_radius * radius_modifier
            fh.write(f"pseudoatom esp_{i}, pos=[{cx:.3f},{cy:.3f},{cz:.3f}], b={std_esp:.4f}\n")
            fh.write(f"set_color col_{i}, [{r:.3f},{g:.3f},{b:.3f}]\n")
            fh.write(f"color col_{i}, esp_{i}\n")
            fh.write(f"show spheres, esp_{i}\n")
            fh.write(f"set sphere_scale, {radius:.3f}, esp_{i}\n")
        
        fh.write("\n# group all into one object\n")
        fh.write("group esp_consensus, esp_*\n")
        fh.write("\ncmd.show('spheres', 'esp_consensus')\n")
        #fh.write(f"cmd.set('sphere_scale', {sphere_radius:.3f}, 'esp_consensus')\n")
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
 
def discover_molecules(directory):
    """
    Find all molecules that have a matching .xyz, .dens.cube, and .pote.cube.
    Returns list of (stem, xyz_path, dens_path, pote_path) sorted by stem.
    """
    d = Path(directory)
    xyz_files = {p.stem: p for p in d.glob('*.xyz')}
    dens_files = {}
    pote_files = {}
 
    for p in d.glob('*.dens.cube'):
        stem = p.name[:-len('.dens.cube')]
        dens_files[stem] = p
    for p in d.glob('*.pote.cube'):
        stem = p.name[:-len('.pote.cube')]
        pote_files[stem] = p
 
    stems = sorted(set(xyz_files) & set(dens_files) & set(pote_files))
    if not stems:
        sys.exit(f"[error] No complete molecule sets found in '{directory}'.\n"
                 f"  Expected: <name>.xyz + <name>.dens.cube + <name>.pote.cube")

    activities = {}
    act_path = d / 'activities.txt'
    if act_path.exists():
        with open(act_path) as acts:
            for line in acts:
                parts = line.split()
                if len(parts) == 2 and parts[0] in stems:
                    activities[parts[0]] = float(parts[1])
    else:
        print("[warn] no activities.txt found, all weights set to 1.0")
        activities = {s: 1.0 for s in stems}
    
    return [(s, xyz_files[s], dens_files[s], pote_files[s], activities[s]) for s in stems]
 
 
def run(args):
    molecules = discover_molecules(args.directory)
    print(f"Found {len(molecules)} molecule(s): {[m[0] for m in molecules]}")
    print(f"Reference: {molecules[0][0]}")
 
    outdir = Path(args.outdir) if args.outdir else Path(args.directory) / 'pharmacophore_out'
    outdir.mkdir(parents=True, exist_ok=True)
 
    # ── 1. Reference molecule ────────────────────────────────────────────────
    ref_stem, ref_xyz, ref_dens, ref_pote, ref_act = molecules[0]
    print(f"\n[1/3] Processing reference: {ref_stem}")
 
    ref_atoms_raw = read_xyz(ref_xyz)
    ref_surf_coords, ref_esp = extract_isosurface(ref_dens, ref_pote, args.isovalue)
 
    # bring reference into its principal-axes frame
    ref_ctr, ref_R_pa = principal_axes_rotation(ref_atoms_raw)
    ref_all_coords = np.array([[a[1], a[2], a[3]] for a in ref_atoms_raw])
    ref_all_pa     = (ref_all_coords - ref_ctr) @ ref_R_pa.T
    ref_heavy_pa   = (heavy_coords(ref_atoms_raw) - ref_ctr) @ ref_R_pa.T
    ref_surf_pa    = (ref_surf_coords - ref_ctr) @ ref_R_pa.T
 
    # save aligned reference xyz
    ref_syms = [a[0] for a in ref_atoms_raw]
    write_aligned_xyz(outdir / f"{ref_stem}_aligned.xyz", ref_syms, ref_all_pa)
    print(f"    RMSD = 0.0000 Angstrom (reference)")
 
    all_surf_coords = [ref_surf_pa]
    all_esp_vals    = [ref_esp]
    all_act = [ref_act]

 
    # ── 2. Mobile molecules ──────────────────────────────────────────────────
    print(f"\n[2/3] Aligning {len(molecules)-1} mobile molecule(s) ...")
 
    for stem, xyz_path, dens_path, pote_path, act in molecules[1:]:
        print(f"  {stem}")
        mob_atoms = read_xyz(xyz_path)
        mob_surf, mob_esp = extract_isosurface(dens_path, pote_path, args.isovalue)
 
        atoms_al, surf_al, rmsd = align_to_reference(ref_heavy_pa, mob_atoms, mob_surf)
        print(f"    RMSD = {rmsd:.4f} Angstrom")
 
        mob_syms = [a[0] for a in mob_atoms]
        write_aligned_xyz(outdir / f"{stem}_aligned.xyz", mob_syms, atoms_al)
 
        all_surf_coords.append(surf_al)
        all_esp_vals.append(mob_esp)
        all_act.append(act)

    ESP_RMSE_matrix = pairwise_esp_rmsd(all_surf_coords, all_esp_vals)
    stems = [m[0] for m in molecules]
    #print("\nPairwise ESP surface RMSD matrix:")#printing matrix of ESP RMSEs
    #print(f"{'':15s}" + "".join(f"{s:15s}" for s in stems))
    #for i, row in enumerate(ESP_RMSE_matrix):
    #    print(f"{stems[i]:15s}" + "".join(f"{v:15.4f}" for v in row))
 
    cluster_labels, linkage_matrix = cluster_molecules(ESP_RMSE_matrix, stems)

    


    # ── 3. Pharmacophore ─────────────────────────────────────────────────────
    print(f"\n[3/3] Building pharmacophore grid (spacing={args.grid} Angstrom) ...")
 
    pharmacophore = build_pharmacophore(
        all_surf_coords, all_esp_vals,
        all_act,
        grid_spacing = args.grid,
        esp_pos      = args.esp_pos,
        esp_neg      = args.esp_neg,
        esp_var      = args.esp_var,
    )
 
 
    pml_path = outdir / 'pharmacophore.pml'
    write_pymol_cgo(pml_path, pharmacophore, sphere_radius=args.grid * 0.45)
    print(f"\nDone.  Output written to: {outdir}/")
    print(f"  Aligned XYZs : *_aligned.xyz")
    print(f"  PyMOL script : pharmacophore.pml")
    print(f"\nIn PyMOL:  run pharmacophore.pml")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
 
def parse_args():
    p = argparse.ArgumentParser(
        description='ESP-based pharmacophore from density/potential cube files.')
    p.add_argument('--directory', type=str,
                   help='Directory containing *.xyz, *.dens.cube, *.pote.cube files', default='.')
    p.add_argument('--esp_cluster_threshold', type=float, default=None)
    p.add_argument('--isovalue', type=float, default=0.001,
                   help='Electron density isovalue (default: 0.001 a.u.)')
    p.add_argument('--grid',     type=float, default=1.0,
                   help='Coarse grid spacing in Angstrom (default: 1.0)')
    p.add_argument('--esp_pos',  type=float, default=0.05,
                   help='ESP threshold for positive (red) colouring (default: +0.02 a.u.)')
    p.add_argument('--esp_neg',  type=float, default=-0.05,
                   help='ESP threshold for negative (blue) colouring (default: -0.02 a.u.)')
    p.add_argument('--esp_var',  type=float, default=0.01,
                   help='ESP std-dev threshold for high-variability (purple) (default: 0.01 a.u.)')
    p.add_argument('--outdir',   type=str,   default=None,
                   help='Output directory (default: <directory>/pharmacophore_out)')
    return p.parse_args()
 
 
if __name__ == '__main__':
    run(parse_args())
 
if __name__ == '__main__':
    run(parse_args())
