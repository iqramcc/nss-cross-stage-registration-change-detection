"""nss_lib.py -- shared library for the CV5570 NSS project.

Sections
  0. config + small geometry helpers
  1. data loading (format auto-detection), pose graph, pair generation
  2. cloud cache (voxel, normals, FPFH, structure masks)
  3. registration: baselines (RANSAC-6DoF, FGR), ours (BEV-XC, RANSAC-4DoF, hypothesis verification),
     one weighted point-to-plane solver for every refiner (ICP, Tukey, trimmed, GNC = method B v2)
  4. visibility: scanner origins, spherical depth maps, three-class change detection, injected objects
  5. building-scale evidence fusion
  6. plotting helpers
Every function is deterministic given C["SEED"].
"""
import os, re, json, time, math, zipfile, warnings
from pathlib import Path
from collections import OrderedDict, deque
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import cKDTree
from scipy import ndimage

warnings.filterwarnings("ignore", category=RuntimeWarning)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
REG = o3d.pipelines.registration

# =====================================================================================
# 0. CONFIG + GEOMETRY HELPERS
# =====================================================================================
C = dict(
    SEED=0, UP=2,
    VOXEL=0.05,            # m, working resolution
    TAU=0.15,              # m, change threshold (same as the earlier experiments)
    FOOT_CELL=0.25,        # m, XY cell for footprint overlap
    MAXD0=1.0,             # m, initial correspondence distance of every refiner (same for all = fair)
    DEPTH_RES="auto",      # deg, angular resolution of spherical depth maps ("auto" = from point density)
    DEPTH_MAX_PTS=1_500_000,
    FS_MARGIN_REL=0.02,    # free-space margin grows with range (beam footprint, grazing angles)
    ORIGIN_MODE="auto",    # "local" (scanner at fragment origin), "estimate", "auto"
    BEV_CELL=0.10, BEV_YAW_STEP=2.0, GNC_DECAY=1.15,
    SUCC_COARSE=(0.5, 5.0), SUCC_FINE=(0.10, 1.0),   # (m, deg): identical to the earlier runs
)
def configure(**kw):
    C.update(kw)

def tf(P, T): return P @ T[:3, :3].T + T[:3, 3]
def inv(T):
    Ti = np.eye(4); Ti[:3, :3] = T[:3, :3].T; Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]; return Ti
def rotz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; return T
def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if t is not None: T[:3, 3] = t
    return T
def hax():
    return [i for i in range(3) if i != C["UP"]]

def reg_errors(T_est, T_gt, src_pts):
    dR = T_est[:3, :3] @ T_gt[:3, :3].T
    rre = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    rte = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    disp = float(np.sqrt(np.mean(np.sum((tf(src_pts, T_est) - tf(src_pts, T_gt)) ** 2, 1))))
    ct, cr = C["SUCC_COARSE"]; ft, fr = C["SUCC_FINE"]
    return dict(rre=rre, rte=rte, disp=disp, succ_coarse=bool(rte < ct and rre < cr), succ_fine=bool(rte < ft and rre < fr))

def bench_frame(e, yaw_deg):
    """Realistic registration input: move the source to its own centroid and give it a random heading.
    Returns M (apply to the source). New GT = T_gt @ inv(M); an estimate T maps back via T @ M."""
    c = e["pts"].mean(0); Tc = make_T(t=-c)
    return rotz(yaw_deg) @ Tc if C["UP"] == 2 else _rot_up(yaw_deg) @ Tc

def failure_type(e):
    if e["succ_coarse"]: return "ok"
    if e["rre"] > 20: return "flip"        # wrong heading (symmetric layout)
    if e["rre"] < 5: return "slide"        # right heading, slid along a plane / repetitive structure
    return "other"

# =====================================================================================
# 1. DATA LOADING
# =====================================================================================
PC_EXT = {".ply", ".pcd", ".npz", ".bin"}
ANNOT_EXT = {".csv", ".json", ".txt", ".pkl", ".pickle", ".yaml", ".yml", ".log", ".npy"}
INDEX = {}

def find_and_extract(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    for z in sorted(root.rglob("*.zip")):
        dst = out / z.stem; marker = dst / ".done"
        if marker.exists(): continue
        print(f"extracting {z.name} ({z.stat().st_size/1e9:.2f} GB) -> {dst}")
        with zipfile.ZipFile(z) as zf: zf.extractall(dst)
        marker.touch()
    return [root, out]

def _is_pc(p):
    s = p.suffix.lower()
    if s in PC_EXT: return True
    if s == ".npy":
        try:
            a = np.load(p, mmap_mode="r"); return a.ndim == 2 and a.shape[1] >= 3 and a.shape[0] > 100
        except Exception: return False
    return False

def read_points(path):
    path = Path(path); s = path.suffix.lower()
    if s in (".ply", ".pcd"): return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)
    if s == ".npy": return np.load(path)[:, :3].astype(np.float64)
    if s == ".npz":
        z = np.load(path); a = z[[k for k in z.files if z[k].ndim == 2 and z[k].shape[1] >= 3][0]]
        return a[:, :3].astype(np.float64)
    if s == ".bin":
        a = np.fromfile(path, dtype=np.float32); return a.reshape(-1, 4 if a.size % 4 == 0 else 3)[:, :3].astype(np.float64)
    raise ValueError(path)

def _read_table(p):
    s = p.suffix.lower()
    try:
        if s == ".csv": return pd.read_csv(p)
        if s == ".json":
            obj = json.loads(p.read_text())
            if isinstance(obj, list): return pd.json_normalize(obj)
            if isinstance(obj, dict):
                vals = list(obj.values())
                if vals and all(isinstance(v, list) and v and isinstance(v[0], dict) for v in vals):
                    return pd.concat([pd.json_normalize(v).assign(_group=k) for k, v in obj.items()], ignore_index=True)
                if vals and all(isinstance(v, dict) for v in vals):
                    return pd.DataFrame.from_dict(obj, orient="index").reset_index().rename(columns={"index": "_key"})
                try: return pd.DataFrame(obj)
                except Exception: return None
        if s in (".pkl", ".pickle"):
            obj = pd.read_pickle(p)
            return obj if isinstance(obj, pd.DataFrame) else (pd.DataFrame(obj) if isinstance(obj, (list, dict)) else None)
        if s in (".txt", ".log"):
            df = pd.read_csv(p, sep=None, engine="python")
            return df if df.shape[1] > 1 else None
    except Exception:
        return None
    return None

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
def to_mat(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    try:
        arr = np.array([float(x) for x in NUM_RE.findall(v)]) if isinstance(v, str) else np.asarray(v, float).ravel()
    except Exception: return None
    if arr.size == 16: return arr.reshape(4, 4)
    if arr.size == 12: return np.vstack([arr.reshape(3, 4), [0, 0, 0, 1]])
    return None

def norm_key(v):
    s = str(v).strip().replace("\\", "/"); name = s.split("/")[-1]
    if name in INDEX: return name
    for ext in (".ply", ".pcd", ".npy", ".npz", ".bin"):
        if name.lower().endswith(ext) and name[: -len(ext)] in INDEX: return name[: -len(ext)]
    two = "/".join(s.split("/")[-2:])
    for ext in ("", ".ply", ".pcd"):
        k = two[: -len(ext)] if ext and two.endswith(ext) else two
        if k in INDEX: return k
    return Path(name).stem

def _match_rate(series):
    vals = series.dropna().astype(str).head(500)
    return 0.0 if len(vals) == 0 else float(np.mean([norm_key(v) in INDEX for v in vals]))

def _matrix_from_cols(df, prefix=None):
    groups = {}
    for c in df.columns:
        m = re.match(r"^(.*?)[_\-\[]?(\d)[_\-,\]]?(\d)?\]?$", str(c))
        if m and pd.api.types.is_numeric_dtype(df[c]): groups.setdefault(m.group(1), []).append(c)
    for pre, cols in groups.items():
        if prefix is not None and pre != prefix: continue
        if len(cols) in (12, 16):
            cols = sorted(cols, key=lambda c: [int(x) for x in re.findall(r"\d+", str(c))[-2:]])
            return pre, [to_mat(r) for r in df[cols].to_numpy(float)]
    return None, None

def _rt_cols(df):
    """Transform given as separate rotation (3x3 or quaternion x,y,z,w) and translation (3) columns."""
    rc = [c for c in df.columns if re.search(r"(^r$|rot)", str(c), re.I)]
    tc = [c for c in df.columns if re.search(r"(^t$|trans|^pos)", str(c), re.I) and c not in rc]
    for r_ in rc:
        for t_ in tc:
            def mk(rv, tv):
                try:
                    R = np.asarray(rv, float).ravel() if not isinstance(rv, str) else np.array([float(x) for x in NUM_RE.findall(rv)])
                    t = np.asarray(tv, float).ravel() if not isinstance(tv, str) else np.array([float(x) for x in NUM_RE.findall(tv)])
                except Exception: return None
                if t.size != 3: return None
                if R.size == 9: Rm = R.reshape(3, 3)
                elif R.size == 4:
                    from scipy.spatial.transform import Rotation
                    Rm = Rotation.from_quat(R).as_matrix()
                else: return None
                return make_T(Rm, t)
            mats = [mk(a, b) for a, b in zip(df[r_], df[t_])]
            if sum(m is not None for m in mats[:20]) >= min(5, len(mats)): return f"{r_}+{t_}", mats
    return None, None

def _matrix_col(df, explicit=None, prefix=None):
    if prefix: return _matrix_from_cols(df, prefix)
    cands = [explicit] if explicit else [c for c in df.columns if re.search(r"(^t$|trans|pose|rt|^gt|matrix|extrinsic|tsfm|xform|^tf$)", str(c), re.I)]
    cands = sorted(cands, key=lambda c: ("global" in str(c).lower(), len(str(c))))   # NSS: 'tsfm' before 'tsfm_global' 
    for c in cands:
        if c in df.columns:
            sample = df[c].dropna().head(5).map(to_mat)
            if len(sample) and sample.map(lambda m: m is not None).all(): return c, df[c].map(to_mat).tolist()
    pre, mats = _matrix_from_cols(df)
    if mats is not None: return pre, mats
    return _rt_cols(df)

SUFFIX_PAIRS = [("src", "tgt"), ("source", "target"), ("_a", "_b"), ("a", "b"), ("0", "1"), ("1", "2"), ("ref", "qry"),
                ("ref", "query"), ("i", "j"), ("anchor", "positive"), ("first", "second")]
def _find_pair_cols(df):
    best = None; cols = [str(c) for c in df.columns]
    for c1 in cols:
        for c2 in cols:
            if c1 == c2: continue
            for a, b in SUFFIX_PAIRS:
                if a in c1.lower() and b in c2.lower() and c1.lower().replace(a, "#", 1) == c2.lower().replace(b, "#", 1):
                    r = min(_match_rate(df[c1]), _match_rate(df[c2]))
                    if best is None or r > best[2]: best = (c1, c2, r)
    return best

def _parse_meta(key, path):
    s = f"{path.parent.parent.name}/{path.parent.name}/{key}"
    def g(pat):
        m = re.search(pat, s, re.I); return m.group(1) if m else None
    b = g(r"(b(?:ui)?ld(?:in)?g[_\-]?\d+)")
    st = g(r"stage[_\-]?(\d+)") or g(r"(?:epoch|time)[_\-]?(\d+)")
    sp = g(r"(?:spot|scan|frag(?:ment)?)[_\-]?(\d+)")
    return (b.capitalize() if b else "B?"), st, sp

def _mat_from_any(v):
    """A 4x4/3x4 matrix, or a dict holding one (any key) or rotation+translation entries."""
    M = to_mat(v) if not isinstance(v, dict) else None
    if M is not None: return M
    if isinstance(v, dict):
        rot = next((v[k] for k in v if re.search(r"(^r$|rot)", str(k), re.I)), None)
        tr = next((v[k] for k in v if re.search(r"(^t$|trans|^pos(ition)?$)", str(k), re.I)), None)
        if rot is not None and tr is not None:
            try:
                R = np.asarray(rot, float).ravel(); t = np.asarray(tr, float).ravel()
                if t.size == 3 and R.size == 9: return make_T(R.reshape(3, 3), t)
                if t.size == 3 and R.size == 4:
                    from scipy.spatial.transform import Rotation
                    return make_T(Rotation.from_quat(R).as_matrix(), t)
            except Exception: pass
        for k in v:
            if re.search(r"(pose|trans|matrix|extrinsic|^t$|gt)", str(k), re.I):
                M = _mat_from_any(v[k])
                if M is not None: return M
    return None

def _poses_in_json(obj, depth=0):
    """Walk any JSON structure; return {fragment: 4x4} for dicts keyed by fragment names or
    lists of records that carry a fragment name plus a pose."""
    out = {}
    if depth > 5: return out
    if isinstance(obj, dict):
        ks = list(obj.keys())[:500]
        hit = [k for k in ks if norm_key(k) in INDEX]
        if ks and len(hit) >= 0.5 * len(ks):
            for k in obj:
                if norm_key(k) in INDEX:
                    M = _mat_from_any(obj[k])
                    if M is not None: out[norm_key(k)] = M
            if out: return out
        for v in obj.values():
            if isinstance(v, (dict, list)): out.update(_poses_in_json(v, depth + 1))
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        name_keys = [k for k in obj[0] if isinstance(obj[0][k], str) and norm_key(obj[0][k]) in INDEX]
        if len(name_keys) == 1:                       # exactly one fragment name per record = per-fragment pose list
            nk = name_keys[0]
            for rec in obj:
                rest = {k: v for k, v in rec.items() if k != nk}
                M = _mat_from_any(rest)
                if M is not None and isinstance(rec.get(nk), str): out[norm_key(rec[nk])] = M
    return out

def load_nss(root, extract_dir, overrides=None, verbose=True):
    """Index point clouds, parse annotation tables. Returns dict D."""
    ov = dict(pair_file=None, src_col=None, tgt_col=None, T_col=None, T_prefix=None, overlap_col=None,
              frag_file=None, frag_key_col=None, pose_col=None, pose_prefix=None)
    ov.update(overrides or {})
    roots = find_and_extract(root, extract_dir)
    files = [p for r in roots for p in Path(r).rglob("*") if p.is_file() and not p.name.startswith(".")
             and "nss_out" not in p.parts and "nss_prep" not in p.parts]       # never re-read our own outputs
    pcs = [p for p in files if _is_pc(p)]
    INDEX.clear()
    for p in pcs:
        k = p.stem
        if k in INDEX and INDEX[k] != p: k = f"{p.parent.name}/{p.stem}"
        INDEX[k] = p
    pcset = set(pcs)
    annots = [p for p in files if p.suffix.lower() in ANNOT_EXT and p not in pcset]
    if verbose: print(f"{len(files)} files, {len(INDEX)} point clouds, {len(annots)} annotation-like files")
    if not INDEX:
        raise RuntimeError(
            f"No point clouds found under {root}. Attach the NSS dataset (pointclouds + annotation) with 'Add Input'. "
            "nss-prep alone is not enough: it only holds poses and pairs, the notebooks still read the scans.")
    tables = {}
    for p in sorted(annots)[:300]:
        df = _read_table(p)
        if df is not None and not df.empty: tables[str(p)] = df
    # per-fragment poses
    frag_pose, meta_cols = {}, {}
    for name, df in tables.items():
        if ov["frag_file"] and ov["frag_file"] not in name: continue
        kc = ov["frag_key_col"]
        if kc is None:
            rates = {c: _match_rate(df[c]) for c in df.columns if not pd.api.types.is_float_dtype(df[c])}
            if not rates: continue
            kc, r = max(rates.items(), key=lambda kv: kv[1])
            fp = _find_pair_cols(df)
            if r < 0.5 or (fp and fp[2] > 0.5): continue
        pc, mats = _matrix_col(df, ov["pose_col"], ov["pose_prefix"])
        if mats is None: continue
        for k, M in zip(df[kc], mats):
            if M is not None: frag_pose[norm_key(k)] = M
        for c in df.columns:
            if re.search(r"(bldg|building|stage|spot|scene|epoch)", str(c), re.I):
                meta_cols.setdefault(c, {}).update(dict(zip(df[kc].map(norm_key), df[c])))
        if verbose: print(f"[fragment poses] {Path(name).name}: key={kc} pose={pc} rows={len(df)}")
    # poses hidden inside JSON as {fragment_name: matrix} or {fragment_name: {rotation, translation}} (any depth)
    for p in annots:
        if p.suffix.lower() != ".json": continue
        try: obj = json.loads(p.read_text())
        except Exception: continue
        got = _poses_in_json(obj)
        new = {k: M for k, M in got.items() if k not in frag_pose}
        if new:
            frag_pose.update(new)
            if verbose: print(f"[fragment poses] {'/'.join(p.parts[-2:])}: {len(new)} poses found inside JSON")
    for p in annots:
        k = norm_key(p.name)
        if k in INDEX and k not in frag_pose and p.suffix.lower() in (".txt", ".npy"):
            try:
                M = to_mat(np.load(p) if p.suffix == ".npy" else p.read_text())
                if M is not None: frag_pose[k] = M
            except Exception: pass
    # pair tables
    rows = []
    for name, df in tables.items():
        if ov["pair_file"] and ov["pair_file"] not in name: continue
        if ov["src_col"]:
            if ov["src_col"] not in df.columns: continue
            sc, tc, r = ov["src_col"], ov["tgt_col"], 1.0
        else:
            fp = _find_pair_cols(df)
            if fp is None or fp[2] < 0.5: continue
            sc, tc, r = fp
        Tc, mats = _matrix_col(df, ov["T_col"], ov["T_prefix"])
        oc = ov["overlap_col"] or next((c for c in df.columns if "overlap" in str(c).lower()), None)
        if verbose: print(f"[pairs] {'/'.join(Path(name).parts[-2:])}: src={sc} tgt={tc} match={r:.0%} T={Tc} overlap={oc} n={len(df)}"
                          f"\n        columns: {list(df.columns)[:25]}")
        cols = list(df.columns)
        for i, row in enumerate(df.itertuples(index=False)):
            rd = dict(zip(cols, row))
            rows.append(dict(src=norm_key(rd[sc]), tgt=norm_key(rd[tc]), T_raw=(mats[i] if mats is not None else None),
                             overlap_annot=(float(rd[oc]) if oc is not None and pd.notna(rd[oc]) else np.nan),
                             ann_file=Path(name).name))
    pa = pd.DataFrame(rows, columns=["src", "tgt", "T_raw", "overlap_annot", "ann_file"])
    if len(pa):
        pa = pa[pa.src.isin(INDEX) & pa.tgt.isin(INDEX) & (pa.src != pa.tgt)]
        pa = pa.assign(uk=[tuple(sorted(x)) for x in zip(pa.src, pa.tgt)]).drop_duplicates("uk").drop(columns="uk").reset_index(drop=True)
    # fragment metadata
    fr = []
    for k, p in INDEX.items():
        b, st, sp = _parse_meta(k, p)
        for c, m in meta_cols.items():
            v = m.get(k); cl = str(c).lower()
            if v is None: continue
            if "bldg" in cl or "building" in cl or "scene" in cl: b = str(v)
            elif "stage" in cl or "epoch" in cl: st = str(v)
            elif "spot" in cl: sp = str(v)
        fr.append(dict(key=k, path=str(p), building=b, stage=st, spot=sp, size_mb=p.stat().st_size / 1e6, has_pose=k in frag_pose))
    frags = pd.DataFrame(fr).set_index("key")
    frags["stage_num"] = pd.to_numeric(frags.stage.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    if verbose:
        print(f"annotated pairs: {len(pa)} | fragment poses: {len(frag_pose)}")
        print(frags.groupby("building").agg(fragments=("path", "size"), stages=("stage_num", "nunique"),
                                            size_gb=("size_mb", lambda x: x.sum() / 1e3)).round(2).to_string())
    return dict(index=dict(INDEX), frags=frags, pairs_annot=pa, frag_pose=frag_pose, tables=tables)

# =====================================================================================
# 2. CLOUD CACHE
# =====================================================================================
class CloudCache:
    """Voxelised fragments with normals (oriented to the scanner), structure masks and lazy FPFH."""
    def __init__(self, max_items=400):
        self.d = OrderedDict(); self.max_items = max_items; self.origins = {}
    def origin(self, key):
        return self.origins.get(key, np.zeros(3))
    def __call__(self, key):
        if key in self.d:
            self.d.move_to_end(key); return self.d[key]
        P = read_points(INDEX[key])
        e = make_entry(P, self.origin(key))
        self.d[key] = e
        if len(self.d) > self.max_items: self.d.popitem(last=False)
        return e
    def fpfh(self, key): return entry_fpfh(self(key))

def make_entry(P, origin=None, voxel=None):
    v = voxel or C["VOXEL"]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(v)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * v, max_nn=30))
    if origin is not None: pcd.orient_normals_towards_camera_location(np.asarray(origin, float))
    return finish_entry(pcd)

def finish_entry(pcd):
    pts = np.asarray(pcd.points).copy(); nrm = np.asarray(pcd.normals).copy()
    nu = np.abs(nrm[:, C["UP"]])
    return dict(pcd=pcd, pts=pts, nrm=nrm, tree=cKDTree(pts), wall=nu < 0.3, horiz=nu > 0.9)

def entry_fpfh(e):
    if "fpfh" not in e:
        f = REG.compute_fpfh_feature(e["pcd"], o3d.geometry.KDTreeSearchParamHybrid(radius=5 * C["VOXEL"], max_nn=100))
        e["fpfh"] = f; e["fpfh_np"] = np.asarray(f.data).T.copy()
    return e["fpfh"]

def transformed_entry(e, T):
    """Rigidly moved copy of an entry. FPFH is rotation invariant, so the features are reused (no recompute)."""
    pts = tf(e["pts"], T); nrm = e["nrm"] @ T[:3, :3].T
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)); pcd.normals = o3d.utility.Vector3dVector(nrm)
    out = finish_entry(pcd)
    if "fpfh" not in e: entry_fpfh(e)
    out["fpfh"] = e["fpfh"]; out["fpfh_np"] = e["fpfh_np"]
    return out

# ---------- ground truth: convention check + pose graph ----------
def _fitness(src_pts, tgt, T, thr):
    d, _ = tgt["tree"].query(tf(src_pts, T), distance_upper_bound=thr * 10)
    return float(np.mean(d < thr))

def check_convention(D, building, cloud, n=30, verbose=True):
    """Try every GT convention on a few pairs; scored on walls only (floors make any height-correct guess look good)."""
    pa = D["pairs_annot"]; frags = D["frags"]; fp = D["frag_pose"]
    pa = pa[pa.src.map(frags.building) == building]
    if pa.empty:     # no pair file: poses only
        ks = list(frags[(frags.building == building) & frags.has_pose].index)
        pa = pd.DataFrame([dict(src=a, tgt=b, T_raw=None) for a, b in zip(ks[:-1], ks[1:])])
    if "overlap_annot" in pa and pa.overlap_annot.notna().any(): pa = pa.sort_values("overlap_annot", ascending=False)
    res = []
    for row in pa.head(n).itertuples():
        S, Tg = cloud(row.src), cloud(row.tgt)
        cands = {"identity": np.eye(4)}
        if getattr(row, "T_raw", None) is not None: cands.update(pair_T=row.T_raw, pair_T_inv=inv(row.T_raw))
        if row.src in fp and row.tgt in fp:
            cands.update(pose_frag2world=inv(fp[row.tgt]) @ fp[row.src], pose_world2frag=fp[row.tgt] @ inv(fp[row.src]))
        for name, T in cands.items():
            res.append(dict(src=row.src, tgt=row.tgt, cand=name, fitness=_fitness(S["pts"][S["wall"]], Tg, T, 1.5 * C["VOXEL"])))
    res = pd.DataFrame(res)
    votes = res.loc[res.groupby(["src", "tgt"]).fitness.idxmax(), "cand"].value_counts()
    med = res.groupby("cand").fitness.median().sort_values(ascending=False)
    conv = med.index[0]; fit = float(med.iloc[0])      # best median fit over pairs (votes tie when scanners barely moved)
    if verbose: print("median wall fitness per candidate:", med.round(3).to_dict())
    if verbose:
        print(res.pivot_table(index=["src", "tgt"], columns="cand", values="fitness").round(3).to_string())
        print(f"-> convention {conv!r} (votes {votes.to_dict()}), median wall fitness {fit:.2f}")
        if fit < 0.3: print("!! low fitness: check units / pose columns before trusting anything")
        if set(res.cand) == {"identity"}:
            print("   (no transforms in the annotations: only 'identity' could be tested. Several pairs scoring >0.5 means the\n"
                  "    fragments are stored in one common frame; low scores are low-overlap or heavily changed pairs.)")
    return conv, fit, res

def world_poses(D, building, conv, cloud=None, verbose=True):
    """Pose of every fragment in one frame. Per-fragment poses if available, else chained through the pair graph
    (this is what lets us create MANY cross-stage pairs even if only a few are annotated).
    Robust chaining: some annotated transforms are wrong, and one bad edge in a naive chain corrupts everything after
    it. So every annotated edge is scored by how well it aligns the two fragments' WALLS, and the poses are chained
    along the MAXIMUM SPANNING TREE of that score. All other edges then serve as independent loop-closure checks."""
    frags = D["frags"]; fp = D["frag_pose"]
    keys = list(frags[frags.building == building].index)
    if conv == "identity":
        return ({k: np.eye(4) for k in keys}, {k: 0 for k in keys},
                dict(method="fragments already in a common frame (GT = identity)", n_posed=len(keys), n_total=len(keys),
                     components=1, largest_component=len(keys), loop_residual_med_m=None, loop_residual_max_m=None))
    if conv.startswith("pose_"):
        W = {k: (fp[k] if conv == "pose_frag2world" else inv(fp[k])) for k in keys if k in fp}
        return W, {k: 0 for k in W}, dict(method="per-fragment poses", n_posed=len(W), n_total=len(keys))
    pa = D["pairs_annot"]; pa = pa[pa.src.isin(keys) & pa.tgt.isin(keys) & pa.T_raw.notna()].reset_index(drop=True)
    Ts = [r.T_raw if conv == "pair_T" else inv(r.T_raw) for r in pa.itertuples()]          # src -> tgt
    if cloud is not None:
        t0 = time.time(); w = []
        for i, (r, T) in enumerate(zip(pa.itertuples(), Ts)):
            S_, T_ = cloud(r.src), cloud(r.tgt)
            w.append(_fitness(S_["pts"][S_["wall"]][::3], T_, T, 1.5 * C["VOXEL"]))
            if verbose and i % 300 == 0: print(f"  scoring annotated edges {i}/{len(pa)} ({time.time()-t0:.0f}s)")
        w = np.array(w)
    else:
        w = np.ones(len(pa))
    pa["edge_fit"] = w
    # Kruskal: strongest edges first
    parent = {k: k for k in keys}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    adj = {}
    for i in np.argsort(-w):
        r = pa.iloc[i]; a_, b_ = find(r.src), find(r.tgt)
        if a_ == b_: continue
        parent[a_] = b_; T = Ts[i]
        adj.setdefault(r.src, []).append((r.tgt, inv(T)))       # W[tgt] = W[src] @ inv(T)
        adj.setdefault(r.tgt, []).append((r.src, T))            # W[src] = W[tgt] @ T
    W, comp = {}, {}; ci = 0
    for start in sorted(adj, key=lambda k: -len(adj[k])):
        if start in W: continue
        W[start] = np.eye(4); comp[start] = ci; q = deque([start])
        while q:
            a_ = q.popleft()
            for b_, M in adj[a_]:
                if b_ not in W:
                    W[b_] = W[a_] @ M; comp[b_] = ci; q.append(b_)
        ci += 1
    # loop-closure residual of EVERY annotated edge vs the chained poses = how many annotations agree with the tree
    res = []
    for r, T in zip(pa.itertuples(), Ts):
        if comp.get(r.src) is not None and comp.get(r.src) == comp.get(r.tgt):
            E = inv(W[r.tgt]) @ W[r.src] @ inv(T); res.append(np.linalg.norm(E[:3, 3]))
        else: res.append(np.nan)
    pa["loop_residual_m"] = res
    D["edges_scored"] = pa
    sizes = pd.Series(comp).value_counts()
    rr = pa.loop_residual_m.dropna()
    info = dict(method="robust pose graph (max spanning tree of wall-fit) from annotated pair transforms",
                n_posed=len(W), n_total=len(keys), components=int(ci), largest_component=int(sizes.iloc[0]) if len(sizes) else 0,
                edges=len(pa), edge_fit_median=float(np.median(w)) if len(w) else None,
                loop_residual_med_m=float(rr.median()) if len(rr) else None,
                share_edges_consistent_10cm=float((rr < 0.10).mean()) if len(rr) else None)
    return W, comp, info

def footprint(P, cell=None):
    cell = cell or C["FOOT_CELL"]; ax = hax()
    return set(map(tuple, np.floor(P[:, ax] / cell).astype(np.int64)))

def make_pairs(D, building, W, comp, cloud, min_ov=0.3, min_geom=0.10, seed=0, verbose=True):
    """All fragment pairs (same component) with XY-footprint overlap >= min_ov and at least min_geom of their
    wall points overlapping under GT (otherwise no method could register them), labelled same/cross stage."""
    frags = D["frags"]; keys = [k for k in frags[frags.building == building].index if k in W]
    t0 = time.time(); feet = {}
    for i, k in enumerate(keys):
        feet[k] = footprint(tf(cloud(k)["pts"], W[k]))
        if verbose and i % 25 == 0: print(f"  footprints {i}/{len(keys)} ({time.time()-t0:.0f}s)")
    ann = set(map(tuple, map(sorted, zip(D["pairs_annot"].src, D["pairs_annot"].tgt))))
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if comp.get(a) != comp.get(b): continue
            fa, fb = feet[a], feet[b]
            ov = len(fa & fb) / max(1, min(len(fa), len(fb)))
            if ov < min_ov: continue
            sa, sb = frags.stage_num[a], frags.stage_num[b]
            if (sa if pd.notna(sa) else 0) > (sb if pd.notna(sb) else 0): a2, b2 = b, a
            else: a2, b2 = a, b
            Tab = inv(W[b2]) @ W[a2]
            # geometric overlap on vertical structure (what registration can actually use)
            S_, T_ = cloud(a2), cloud(b2)
            g1 = _fitness(S_["pts"][S_["wall"]][::4], T_, Tab, 2 * C["VOXEL"])
            g2 = _fitness(T_["pts"][T_["wall"]][::4], S_, inv(Tab), 2 * C["VOXEL"])
            rows.append(dict(src=a2, tgt=b2, stage_src=frags.stage_num[a2], stage_tgt=frags.stage_num[b2],
                             cross_stage=bool(sa != sb), ov_foot=ov, ov_geom=min(g1, g2),
                             annotated=tuple(sorted((a, b))) in ann, T_gt=Tab))
    P = pd.DataFrame(rows)
    pa = D["pairs_annot"]
    if len(P) and pa.overlap_annot.notna().any():
        ao = {tuple(sorted(k)): v for k, v in zip(zip(pa.src, pa.tgt), pa.overlap_annot)}
        P["overlap_annot"] = [ao.get(tuple(sorted((a, b))), np.nan) for a, b in zip(P.src, P.tgt)]
        m = P.overlap_annot.notna()
        if m.sum() >= 10:
            rho = P.loc[m, ["overlap_annot", "ov_geom"]].corr("spearman").iloc[0, 1]
            if verbose:
                print(f"  GT sanity check: Spearman(annotated overlap, our wall overlap under GT) = {rho:.2f} on {int(m.sum())} pairs")
                share = (D.get("edges_scored") is not None and len(D["edges_scored"]) and
                         float((D["edges_scored"].loop_residual_m < 0.10).mean())) or None
                if rho < 0.4 and (share is None or share < 0.6):
                    print("  !! LOW: the ground truth used here is probably WRONG (fragments misaligned). Do not run NB1-NB3; send the NB0 output.")
                elif rho < 0.4:
                    print("     (weak check: NSS defines overlap differently from our wall overlap; the pose graph agrees with "
                          f"{share:.0%} of the annotated transforms, which is the stronger evidence)")
    n0 = len(P); P = P[P.ov_geom >= min_geom].reset_index(drop=True)
    if verbose: print(f"  dropped {n0 - len(P)} pairs with wall overlap < {min_geom}")
    rng = np.random.default_rng(seed)
    P["pair_id"] = P.src + "__" + P.tgt
    P["yaw_deg"] = rng.uniform(0, 360, len(P))          # random heading applied to the source = fair benchmark
    P["ov_bin"] = pd.cut(P.ov_geom, [0, .2, .35, .5, 1.01], labels=["<0.2", "0.2-0.35", "0.35-0.5", ">0.5"])
    # location split: 8 m cells of the pair midpoint, ~30% dev
    cen = {k: tf(cloud(k)["pts"].mean(0)[None], W[k])[0] for k in set(P.src) | set(P.tgt)} if len(P) else {}
    mids = np.array([(cen[r.src] + cen[r.tgt]) / 2 for r in P.itertuples()]) if len(P) else np.zeros((0, 3))
    ax = hax()
    P["loc"] = [f"{int(m[ax[0]] // 8)}_{int(m[ax[1]] // 8)}" for m in mids]
    locs = np.array(sorted(P["loc"].unique())); rng.shuffle(locs)
    dev = set(locs[: max(1, int(round(0.3 * len(locs))))]) if len(locs) > 1 else set()
    if len(locs) <= 1: print("!! only one location cell: everything goes to eval")
    P["split"] = np.where(P["loc"].isin(dev), "dev", "eval")
    if verbose:
        print(f"pairs with footprint overlap >= {min_ov}: {len(P)}")
        print(P.groupby(["split", "cross_stage"]).size().unstack(fill_value=0).to_string())
    return P

def stratified(df, n, by=("cross_stage", "ov_bin"), seed=0):
    if len(df) == 0: return df
    ng = max(1, df.groupby(list(by), observed=True).ngroups)
    return df.sample(frac=1, random_state=seed).groupby(list(by), observed=True).head(max(1, int(np.ceil(n / ng)))).reset_index(drop=True)

# =====================================================================================
# 3. REGISTRATION
# =====================================================================================
# ---------- baselines ----------
def reg_ransac6(S, Tg, iters=100_000):
    d = 1.5 * C["VOXEL"]
    r = REG.registration_ransac_based_on_feature_matching(
        S["pcd"], Tg["pcd"], entry_fpfh(S), entry_fpfh(Tg), True, d, REG.TransformationEstimationPointToPoint(False), 3,
        [REG.CorrespondenceCheckerBasedOnEdgeLength(0.9), REG.CorrespondenceCheckerBasedOnDistance(d)],
        REG.RANSACConvergenceCriteria(iters, 0.999))
    return np.asarray(r.transformation).copy()

def reg_fgr(S, Tg):
    r = REG.registration_fgr_based_on_feature_matching(S["pcd"], Tg["pcd"], entry_fpfh(S), entry_fpfh(Tg),
        REG.FastGlobalRegistrationOption(maximum_correspondence_distance=1.5 * C["VOXEL"]))
    return np.asarray(r.transformation).copy()

# ---------- hypothesis verification on STRUCTURE (walls/columns), not floors ----------
def structure_score(S, Tg, T, n=6000, seed=0):
    """Fraction of the source's vertical-structure points that land on target geometry.
    Floors/ceilings are excluded: they match under any horizontal slide, which is exactly the RANSAC failure mode."""
    idx = np.flatnonzero(S["wall"])
    if len(idx) < 50: idx = np.arange(len(S["pts"]))
    if len(idx) > n: idx = np.random.default_rng(seed).choice(idx, n, replace=False)
    d, j = Tg["tree"].query(tf(S["pts"][idx], T), distance_upper_bound=4 * C["VOXEL"])
    ok = np.isfinite(d)
    if not ok.any(): return 0.0
    nr = np.abs(np.einsum("ij,ij->i", tf(S["pts"][idx[ok]], T) - Tg["pts"][j[ok]], Tg["nrm"][j[ok]]))
    return float(np.sum(nr < 1.5 * C["VOXEL"]) / len(idx))

def feature_corr(S, Tg, seed=0):
    """Mutual nearest neighbours in FPFH space on non-horizontal points (cached per source/target pair)."""
    key = id(Tg)
    if S.get("_corr_key") == key: return S["_corr"]
    rng = np.random.default_rng(seed)
    entry_fpfh(S); entry_fpfh(Tg)
    ms = np.flatnonzero(~S["horiz"]); mt = np.flatnonzero(~Tg["horiz"])
    if len(ms) < 30 or len(mt) < 30: ms, mt = np.arange(len(S["pts"])), np.arange(len(Tg["pts"]))
    fs, ft = S["fpfh_np"][ms], Tg["fpfh_np"][mt]
    _, i_st = cKDTree(ft).query(fs, k=1, workers=-1); _, i_ts = cKDTree(fs).query(ft, k=1, workers=-1)
    mutual = i_ts[i_st] == np.arange(len(ms))
    cs, ct = ms[mutual], mt[i_st[mutual]]
    if len(cs) < 10: cs, ct = ms, mt[i_st]          # fall back to one-way matches
    if len(cs) > 6000:
        k = rng.choice(len(cs), 6000, replace=False); cs, ct = cs[k], ct[k]
    S["_corr_key"] = key; S["_corr"] = (S["pts"][cs], Tg["pts"][ct])
    return S["_corr"]

def feature_score(S, Tg, T):
    """Share of mutual FPFH matches that agree with T (within 3 voxels). Independent of the wall-overlap cue:
    a symmetric look-alike alignment can overlap walls well, but the local-shape matches will not agree with it."""
    P, Q = feature_corr(S, Tg)
    return float(np.mean(np.linalg.norm(tf(P, T) - Q, axis=1) < 3 * C["VOXEL"])) if len(P) else 0.0

# ---------- ours 1: gravity-aware RANSAC (4-DoF, 2-point minimal sample) ----------
def _prior_ok(cx, cy, prior):
    """Hypotheses that put the source centroid within prior radius (horizontal) of the rough position."""
    return np.hypot(cx - prior[0][0], cy - prior[0][1]) <= prior[1]

def reg_ransac4(S, Tg, n_hyp=30000, topk=8, seed=0, return_hyps=False, prior=None):
    """Scanners on site are levelled, so only yaw + 3D translation are unknown.
    A 2-correspondence minimal sample fixes (yaw, t): fewer random draws, no tilted solutions, no 3-point degeneracies.
    prior=(xy, radius): optional rough position of the source centroid in the target frame; hypotheses outside are
    discarded before scoring (operators know roughly where the scanner stood)."""
    rng = np.random.default_rng(seed); v = C["VOXEL"]
    P, Q = feature_corr(S, Tg, seed)
    ax0, ax1 = hax(); up = C["UP"]
    i = rng.integers(0, len(P), n_hyp); j = rng.integers(0, len(P), n_hyp)
    dp, dq = P[j] - P[i], Q[j] - Q[i]
    lp, lq = np.hypot(dp[:, ax0], dp[:, ax1]), np.hypot(dq[:, ax0], dq[:, ax1])
    ok = (lp > 0.5) & (np.abs(lp - lq) < 3 * v) & (np.abs(dp[:, up] - dq[:, up]) < 3 * v)
    i, j, dp, dq = i[ok], j[ok], dp[ok], dq[ok]
    if len(i) == 0: return (np.eye(4), []) if return_hyps else np.eye(4)
    yaw = np.arctan2(dq[:, ax1], dq[:, ax0]) - np.arctan2(dp[:, ax1], dp[:, ax0])
    c, s = np.cos(yaw), np.sin(yaw)
    def rot(X):
        Y = X.copy(); Y[:, ax0] = c * X[:, ax0] - s * X[:, ax1]; Y[:, ax1] = s * X[:, ax0] + c * X[:, ax1]; return Y
    t = 0.5 * ((Q[i] - rot(P[i])) + (Q[j] - rot(P[j])))
    if prior is not None:
        cs = S["pts"].mean(0)
        keep = _prior_ok(c * cs[ax0] - s * cs[ax1] + t[:, ax0], s * cs[ax0] + c * cs[ax1] + t[:, ax1], prior)
        if not keep.any(): return (np.eye(4), []) if return_hyps else np.eye(4)
        yaw, c, s, t = yaw[keep], c[keep], s[keep], t[keep]
    # score every hypothesis by correspondence inliers (chunked, vectorised)
    sel = np.arange(len(P)) if len(P) <= 1500 else rng.choice(len(P), 1500, replace=False)
    Psub, Qsub = P[sel], Q[sel]
    score = np.zeros(len(yaw)); thr2 = (3 * v) ** 2
    for a in range(0, len(yaw), 2000):
        cc, ss, tt = c[a:a+2000, None], s[a:a+2000, None], t[a:a+2000]
        X0 = cc * Psub[None, :, ax0] - ss * Psub[None, :, ax1] + tt[:, None, ax0] - Qsub[None, :, ax0]
        X1 = ss * Psub[None, :, ax0] + cc * Psub[None, :, ax1] + tt[:, None, ax1] - Qsub[None, :, ax1]
        X2 = Psub[None, :, up] + tt[:, None, up] - Qsub[None, :, up]
        score[a:a+2000] = ((X0**2 + X1**2 + X2**2) < thr2).sum(1)
    order = np.argsort(-score); hyps = []
    for o in order:
        Th = np.eye(4); R2 = np.array([[c[o], -s[o]], [s[o], c[o]]])
        Th[np.ix_([ax0, ax1], [ax0, ax1])] = R2; Th[:3, 3] = t[o]
        if all(np.linalg.norm(Th[:3, 3] - h[:3, 3]) > 0.5 or abs(_yaw_diff(Th, h)) > 5 for h in hyps): hyps.append(Th)
        if len(hyps) >= topk: break
    best = max(hyps, key=lambda T: structure_score(S, Tg, T)) if hyps else np.eye(4)
    return (best, hyps) if return_hyps else best

def _yaw_of(T):
    a0, a1 = hax(); return np.degrees(np.arctan2(T[a1, a0], T[a0, a0]))
def _yaw_diff(A, B):
    return (( _yaw_of(A) - _yaw_of(B) + 180) % 360) - 180

# ---------- ours 2: BEV cross-correlation (correspondence-free, globally exhaustive) ----------
def _raster(XY, origin, shape, cell, blur=1.0):
    ij = np.floor((XY - origin) / cell).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    img = np.zeros(shape, np.float32); img[ij[ok, 0], ij[ok, 1]] = 1.0      # binary: density-independent, change-robust
    return ndimage.gaussian_filter(img, blur) if blur else img

def _z_offset(S, Tg, bin_=0.02):
    zs = S["pts"][S["horiz"], C["UP"]]; zt = Tg["pts"][Tg["horiz"], C["UP"]]
    if len(zs) < 50 or len(zt) < 50: return float(np.median(Tg["pts"][:, C["UP"]]) - np.median(S["pts"][:, C["UP"]]))
    lo = min(zs.min(), zt.min()) - 1; hi = max(zs.max(), zt.max()) + 1
    b = np.arange(lo, hi + bin_, bin_)
    hs = np.histogram(zs, b)[0].astype(float); ht = np.histogram(zt, b)[0].astype(float)
    hs, ht = np.sqrt(hs), np.sqrt(ht)             # compress: floor slabs should not drown ceilings
    n = len(b) - 1; F = np.fft.rfft(ht, 2 * n) * np.conj(np.fft.rfft(hs, 2 * n))
    cc = np.fft.irfft(F, 2 * n); k = int(np.argmax(cc)); k = k - 2 * n if k > n else k
    return float(k * bin_)

def reg_bev(S, Tg, cell=None, yaw_step=None, topk=6, return_debug=False, prior=None):
    """Gravity-aligned bird's-eye-view cross-correlation.
    Walls/columns -> binary occupancy image. For every yaw (sweep), one FFT cross-correlation finds the best
    translation over the WHOLE plane at once: no features, no random sampling, globally optimal on the grid.
    Coarse-to-fine: yaw sweep on a 2.5x coarser grid, then refine the best yaws on the fine grid."""
    from scipy.fft import rfft2, irfft2, next_fast_len
    cell = cell or C["BEV_CELL"]; yaw_step = yaw_step or C["BEV_YAW_STEP"]; ax = hax()
    rng = np.random.default_rng(0)
    ws = S["pts"][S["wall"]][:, ax]; wt = Tg["pts"][Tg["wall"]][:, ax]
    if len(ws) < 50 or len(wt) < 50: ws, wt = S["pts"][:, ax], Tg["pts"][:, ax]
    if len(ws) > 60000: ws = ws[rng.choice(len(ws), 60000, replace=False)]
    if len(wt) > 200000: wt = wt[rng.choice(len(wt), 200000, replace=False)]
    rad = np.linalg.norm(ws, axis=1); keep = rad <= np.percentile(rad, 98); ws = ws[keep]
    rs = rad[keep].max() + 0.5
    tz = _z_offset(S, Tg)
    cs_xy = S["pts"].mean(0)[ax]
    levels = {}
    def level(c):
        if c not in levels:
            lo = wt.min(0) - rs; shape = tuple(int(x) for x in np.ceil((wt.max(0) + rs - lo) / c) + 1)
            ss = int(np.ceil(2 * rs / c)) + 1
            fsh = tuple(next_fast_len(n + ss) for n in shape)
            It = _raster(wt, lo, shape, c)
            levels[c] = dict(lo=lo, ss=ss, fsh=fsh, F=rfft2(It, fsh), n=np.sqrt((It ** 2).sum()), It=It)
        return levels[c]
    def corr_at(yaw, c, full=False):
        L_ = level(c); a = np.radians(yaw); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        so = -np.array([rs, rs]); Is = _raster(ws @ R.T, so, (L_["ss"], L_["ss"]), c)
        cc = irfft2(L_["F"] * np.conj(rfft2(Is, L_["fsh"])), L_["fsh"], workers=-1)
        if prior is not None:
            # translation t = lo + k*c - so; the source centroid lands at R*cs + t; keep only shifts near the prior
            ku = [np.where(np.arange(f) < f // 2, np.arange(f), np.arange(f) - f) for f in L_["fsh"]]
            tx = L_["lo"][0] + ku[0] * c - so[0] + (R @ cs_xy)[0] - prior[0][0]
            ty = L_["lo"][1] + ku[1] * c - so[1] + (R @ cs_xy)[1] - prior[0][1]
            mask = (tx[:, None] ** 2 + ty[None, :] ** 2) <= prior[1] ** 2
            if mask.any(): cc = np.where(mask, cc, -np.inf)
        k = np.unravel_index(int(np.argmax(cc)), cc.shape)
        peak = cc[k] / (L_["n"] * np.sqrt((Is ** 2).sum()) + 1e-9)
        sub = []
        for d in range(2):
            km = list(k); kp = list(k); km[d] -= 1; kp[d] = (kp[d] + 1) % cc.shape[d]
            y0, y1, y2 = cc[tuple(km)], cc[k], cc[tuple(kp)]; den = y0 - 2 * y1 + y2
            ok_ = np.isfinite(y0) and np.isfinite(y2) and abs(den) > 1e-12
            sub.append(0.5 * (y0 - y2) / den if ok_ else 0.0)
        kk = np.array([(k[d] if k[d] < L_["fsh"][d] // 2 else k[d] - L_["fsh"][d]) + sub[d] for d in range(2)])
        txy = L_["lo"] + kk * c - so
        return (peak, txy, cc) if full else (peak, txy)
    coarse = 2.5 * cell
    yaws = np.arange(0, 360, yaw_step); sc = np.array([corr_at(y, coarse)[0] for y in yaws])
    loc = [i for i in range(len(yaws)) if sc[i] >= sc[i - 1] and sc[i] >= sc[(i + 1) % len(yaws)]]
    loc = sorted(loc, key=lambda i: -sc[i])[:topk]
    hyps = []
    for i in loc:
        cand = []
        for y in np.arange(yaws[i] - yaw_step, yaws[i] + yaw_step + 1e-9, 0.5):
            p, t = corr_at(y, cell); cand.append((p, y, t))
        p0, y0, _ = max(cand, key=lambda r: r[0])
        for y in (y0 - 0.25, y0 + 0.25):
            p, t = corr_at(y, cell); cand.append((p, y, t))
        p, y, t = max(cand, key=lambda r: r[0])
        Th = rotz(y) if C["UP"] == 2 else _rot_up(y)
        Th[ax[0], 3], Th[ax[1], 3] = t; Th[C["UP"], 3] = tz
        hyps.append((Th, float(p)))
    Tbest = max(hyps, key=lambda h: structure_score(S, Tg, h[0]))[0]
    if return_debug:
        _, _, cc = corr_at(_yaw_of(Tbest), cell, full=True)
        return Tbest, dict(yaws=yaws, scores=sc, cc=np.fft.fftshift(np.where(np.isfinite(cc), cc, 0)), It=level(cell)["It"], hyps=hyps)
    return Tbest

def _rot_up(deg):
    a0, a1 = hax(); a = np.radians(deg); T = np.eye(4)
    T[a0, a0] = T[a1, a1] = np.cos(a); T[a0, a1] = -np.sin(a); T[a1, a0] = np.sin(a); return T

def _quick_polish(S, Tg, T, n=8000, iters=12, seed=0):
    """A few robust point-to-plane iterations (4-DoF) at short range: brings a roughly-right hypothesis onto the
    surfaces so that correct and wrong hypotheses become clearly separable when scored."""
    v = C["VOXEL"]; idx = np.arange(len(S["pts"]))
    if len(idx) > n: idx = np.random.default_rng(seed).choice(idx, n, replace=False)
    P0 = S["pts"][idx]
    for it in range(iters):
        dm = 6 * v if it < iters // 2 else 3 * v
        P = tf(P0, T); d, j = Tg["tree"].query(P, distance_upper_bound=dm, workers=-1); ok = np.isfinite(d)
        if ok.sum() < 30: break
        Pk, Q, N = P[ok], Tg["pts"][j[ok]], Tg["nrm"][j[ok]]
        r = np.einsum("ij,ij->i", Pk - Q, N); c = dm / 2
        w = np.where(np.abs(r) < c, (1 - (r / c) ** 2) ** 2, 0.0)
        dT, _ = _solve_p2plane(Pk, Q, N, w, 4); T = dT @ T
    return T

def sym_structure_score(S, Tg, T):
    """Structure agreement in BOTH directions: a wrong alignment that parks the source on a dense corridor of the
    target scores high one way but low the other way."""
    return 0.5 * (structure_score(S, Tg, T) + structure_score(Tg, S, inv(T)))

def reg_ours_global(S, Tg, seed=0, return_all=False, prior=None):
    """Pool hypotheses from BEV-XC (correspondence-free) and RANSAC-4DoF (feature-based). Every hypothesis is polished
    briefly and scored by symmetric wall agreement; the best wins. The two generators fail for different reasons,
    and verification after polishing separates right from wrong far better than scoring the raw hypotheses.
    Selection = symmetric wall agreement x sqrt(feature agreement): smoke test showed wall agreement alone can prefer
    a symmetric look-alike produced by BEV-XC."""
    _, dbg = reg_bev(S, Tg, return_debug=True, prior=prior)
    _, hr = reg_ransac4(S, Tg, seed=seed, return_hyps=True, prior=prior)
    nb = 3 if prior is None else 6                                    # with a prior, keep more (all are plausible)
    cands = [h[0] for h in sorted(dbg["hyps"], key=lambda h: -h[1])[:nb]] + hr[:(5 if prior is None else 8)]
    pol = [_quick_polish(S, Tg, T) for T in cands]
    st = np.array([sym_structure_score(S, Tg, T) for T in pol]); fe = np.array([feature_score(S, Tg, T) for T in pol])
    # two independent cues must agree: geometry (walls overlap) x appearance (local-shape matches agree)
    sc = st * np.sqrt(fe + 1e-3)
    if prior is not None:                                             # polishing must not have walked out of the prior
        cs = S["pts"].mean(0); ax_ = hax()
        inside = np.array([_prior_ok(*tf(cs[None], T)[0][ax_], (prior[0], prior[1] + 0.5)) for T in pol])
        if inside.any(): sc = np.where(inside, sc, -np.inf)
    best = int(np.argmax(sc))
    return (pol[best], dict(structure=st, features=fe, source=("bev" if best < nb else "r4"))) if return_all else pol[best]

# ---------- one weighted point-to-plane solver for every refiner ----------
def _solve_p2plane(P, Q, N, w, dof):
    r = np.einsum("ij,ij->i", P - Q, N)
    if dof == 6:
        J = np.hstack([np.cross(P, N), N])
    else:   # yaw about UP + translation
        e = np.zeros(3); e[C["UP"]] = 1.0
        J = np.hstack([np.einsum("ij,ij->i", np.cross(e[None], P), N)[:, None], N])
    A = J.T @ (w[:, None] * J); b = -J.T @ (w * r)
    try: x = np.linalg.solve(A + 1e-9 * np.eye(len(A)), b)
    except np.linalg.LinAlgError: return np.eye(4), 0.0
    T = np.eye(4)
    if dof == 6:
        T[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(x[:3]); T[:3, 3] = x[3:]
    else:
        T[:3, :3] = (rotz(np.degrees(x[0])) if C["UP"] == 2 else _rot_up(np.degrees(x[0])))[:3, :3]; T[:3, 3] = x[1:]
    return T, float(np.linalg.norm(x))

def refine(S, Tg, T0, mode="icp", dof=6, iters=None, mask=None, max_pts=30000, seed=0, return_info=False):
    """mode: icp (L2), tukey, trimmed, gnc (method B v2).
    All refiners share the same correspondence schedule (MAXD0 -> 2 voxel), so they differ ONLY in the robust weighting."""
    iters = iters or (90 if mode == "gnc" else 60)
    v = C["VOXEL"]; idx = np.arange(len(S["pts"])) if mask is None else np.flatnonzero(mask)
    if len(idx) > max_pts: idx = np.random.default_rng(seed).choice(idx, max_pts, replace=False)
    P0 = S["pts"][idx]; T = T0.copy()
    d0 = C["MAXD0"]
    if mode == "gnc":
        # adaptive basin: open the search radius only as wide as the current misalignment needs. A start that is
        # already right (median wall distance a few cm) must not be dragged away by changed geometry 1 m away.
        dd, _ = Tg["tree"].query(tf(S["pts"][S["wall"]][::5] if S["wall"].any() else P0[::5], T), workers=-1)
        d0 = float(np.clip(4 * np.median(dd), 4 * v, C["MAXD0"]))
    dmax = np.geomspace(d0, 2 * v, max(2, iters // 2)).tolist() + [2 * v] * (iters - max(2, iters // 2))
    cbar = 1.5 * v; mu = None; w_last = None
    for it in range(iters):
        P = tf(P0, T)
        d, j = Tg["tree"].query(P, distance_upper_bound=dmax[it], workers=-1)
        ok = np.isfinite(d)
        if ok.sum() < 30: break
        Pk, Q, N = P[ok], Tg["pts"][j[ok]], Tg["nrm"][j[ok]]
        r = np.einsum("ij,ij->i", Pk - Q, N)
        if mode == "icp":
            w = np.ones(len(r))
        elif mode == "tukey":
            c = max(dmax[it] / 2, 2 * v); w = np.where(np.abs(r) < c, (1 - (r / c) ** 2) ** 2, 0.0)
        elif mode == "trimmed":
            w = (np.abs(r) <= np.quantile(np.abs(r), 0.7)).astype(float)
        elif mode == "gnc":
            # Graduated Non-Convexity with Geman-McClure (Yang et al. 2020): start almost quadratic (wide basin),
            # shrink mu -> 1 so the cost becomes strongly robust (precise, ignores changed geometry)
            if mu is None: mu = max(1.0, 2 * (np.max(np.abs(r)) / cbar) ** 2)
            w = (mu * cbar ** 2 / (r ** 2 + mu * cbar ** 2)) ** 2
            mu = max(1.0, mu / C["GNC_DECAY"])
        else:
            raise KeyError(mode)
        dT, step = _solve_p2plane(Pk, Q, N, w, dof)
        T = dT @ T; w_last = (ok, w)
        if step < 1e-6 and it > 5 and (mode != "gnc" or mu <= 1.0): break
    if mode == "gnc" and C.get("GNC_POLISH6", True):
        # gravity prior for the basin, full 6-DoF for the last centimetres: scans of different stages are tilted
        # against each other by a few tenths of a degree, which a 4-DoF solution cannot absorb
        for _ in range(15):
            P = tf(P0, T); d, j = Tg["tree"].query(P, distance_upper_bound=2 * v, workers=-1); ok = np.isfinite(d)
            if ok.sum() < 30: break
            Pk, Q, N = P[ok], Tg["pts"][j[ok]], Tg["nrm"][j[ok]]
            r = np.einsum("ij,ij->i", Pk - Q, N)
            w = (cbar ** 2 / (r ** 2 + cbar ** 2)) ** 2
            dT, step = _solve_p2plane(Pk, Q, N, w, 6); T = dT @ T
            if step < 1e-6: break
    if return_info:
        return T, dict(iters=it + 1, inlier_frac=float(np.mean(w_last[1] > 0.5)) if w_last else 0.0)
    return T

def ref_status_rule(shift, f0, f1, max_shift=0.30):
    """gt_ok: GT already at the optimum (<5 cm). refined: robust ICP moved it <=30 cm without worsening the wall fit
    (the GT lies inside the ICP basin, so the local optimum is the better reference). unverified: anything else."""
    if shift < 0.05: return "gt_ok"
    if shift <= max_shift and f1 >= f0 - 0.01: return "refined"
    return "unverified"

def reference_T(S, Tg, T_gt, max_shift=0.30):
    """Audited reference: polish the benchmark GT with a robust 6-DoF ICP (Open3D, Cauchy kernel -- a different
    implementation from every evaluated method) and keep the polish if it is small and does not worsen the fit.
    status: see ref_status_rule. 'unverified' pairs keep the GT and are excluded from FINE accuracy metrics."""
    v = C["VOXEL"]; T = T_gt.copy()
    for dmax in (0.30, 0.15, 2 * v):
        est = REG.TransformationEstimationPointToPlane(REG.CauchyLoss(k=dmax / 2))
        T = np.asarray(REG.registration_icp(S["pcd"], Tg["pcd"], dmax, T, est, REG.ICPConvergenceCriteria(max_iteration=30)).transformation).copy()
    shift = float(np.mean(np.linalg.norm(tf(S["pts"], T) - tf(S["pts"], T_gt), axis=1)))
    f0 = _fitness(S["pts"][S["wall"]], Tg, T_gt, 1.5 * v); f1 = _fitness(S["pts"][S["wall"]], Tg, T, 1.5 * v)
    status = ref_status_rule(shift, f0, f1, max_shift)
    Tr = T_gt if status == "unverified" else T
    dR = T[:3, :3] @ T_gt[:3, :3].T
    rot = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return Tr, dict(ref_status=status, gt_shift_m=shift, gt_shift_deg=rot, fit_gt=f0, fit_ref=f1)

GLOBAL_METHODS = {
    "ransac6": lambda S, Tg: reg_ransac6(S, Tg),
    "fgr": lambda S, Tg: reg_fgr(S, Tg),
    "ransac4 (ours)": lambda S, Tg: reg_ransac4(S, Tg),
    "bev-xc (ours)": lambda S, Tg: reg_bev(S, Tg),
    "bev+r4 (ours)": lambda S, Tg: reg_ours_global(S, Tg),
}
REFINERS = {   # name: (mode, dof)
    "icp": ("icp", 6), "tukey": ("tukey", 6), "trimmed": ("trimmed", 6), "B v2: gnc+polish (ours)": ("gnc", 4),
}

# =====================================================================================
# 4. VISIBILITY-AWARE CHANGE DETECTION
# =====================================================================================
def _sph(Prel, res):
    r = np.linalg.norm(Prel, axis=1) + 1e-9
    a0, a1 = hax()
    az = np.arctan2(Prel[:, a1], Prel[:, a0]); el = np.arcsin(np.clip(Prel[:, C["UP"]] / r, -1, 1))
    W = int(round(360 / res)); H = int(round(180 / res))
    ix = np.clip(((az + np.pi) / (2 * np.pi) * W).astype(np.int64), 0, W - 1)
    iy = np.clip(((el + np.pi / 2) / np.pi * H).astype(np.int64), 0, H - 1)
    return r, iy, ix, (H, W)

def self_occlusion(P, o, res=0.5, n=40000, seed=0):
    """For a real scan, every point is the FIRST return along its ray. Measured from the true scanner position,
    almost no point hides behind another; from a wrong position, many do. Lower = more plausible origin."""
    if len(P) > n: P = P[np.random.default_rng(seed).choice(len(P), n, replace=False)]
    r, iy, ix, shp = _sph(P - o, res)
    D = np.full(shp, np.inf); np.minimum.at(D, (iy, ix), r)
    return float(np.mean(r > D[iy, ix] + 0.05 + 0.03 * r))

def estimate_origin(P, seed=0):
    """Scanner position = the viewpoint minimising self-occlusion (coarse-to-fine grid search)."""
    rng = np.random.default_rng(seed)
    Ps = P if len(P) <= 30000 else P[rng.choice(len(P), 30000, replace=False)]
    up, (a0, a1) = C["UP"], hax()
    z = Ps[:, up]; floor = np.percentile(z, 2)
    # TLS point density is highest right below the scanner: start there
    lowpts = Ps[z < floor + 0.3]
    if len(lowpts) > 100:
        H2, xe, ye = np.histogram2d(lowpts[:, a0], lowpts[:, a1], bins=40)
        i, j = np.unravel_index(np.argmax(ndimage.gaussian_filter(H2, 1)), H2.shape)
        c0 = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
    else:
        c0 = np.median(Ps[:, [a0, a1]], 0)
    best = (np.inf, None)
    for span, step, zs in [(4.0, 0.5, floor + np.array([0.4, 0.9, 1.4, 1.9])), (0.6, 0.15, None), (0.15, 0.05, None)]:
        if zs is None: zs = best[1][up] + np.array([-0.3, -0.15, 0, 0.15, 0.3]) * (step / 0.15)
        cxy = c0 if best[1] is None else best[1][[a0, a1]]
        g = np.arange(-span, span + 1e-9, step)
        for dx in g:
            for dy in g:
                for zz in zs:
                    o = np.zeros(3); o[a0], o[a1], o[up] = cxy[0] + dx, cxy[1] + dy, zz
                    s = self_occlusion(Ps, o, res=1.0, n=8000)
                    if s < best[0]: best = (s, o)
    return best[1], self_occlusion(P, best[1])

def resolve_origins(keys, cloud, mode=None, verbose=True):
    """Decide each fragment's scanner position: local origin if it is self-consistent, else estimated."""
    mode = mode or C["ORIGIN_MODE"]; rows = []
    if not keys:
        return pd.DataFrame(columns=["origin_src", "self_occ_local", "self_occ_used", "ox", "oy", "oz"])
    for k in keys:
        P = read_points(INDEX[k]); s_loc = self_occlusion(P, np.zeros(3))
        o, s, src = np.zeros(3), s_loc, "local"
        if mode == "estimate" or (mode == "auto" and s_loc > 0.05):
            o_est, s_est = estimate_origin(P)
            if mode == "estimate" or s_est < s_loc: o, s, src = o_est, s_est, "estimated"
        cloud.origins[k] = o; cloud.d.pop(k, None)
        rows.append(dict(key=k, origin_src=src, self_occ_local=s_loc, self_occ_used=s, ox=o[0], oy=o[1], oz=o[2]))
        if verbose: print(f"  {k}: local={s_loc:.3f} -> {src} ({s:.3f})")
        elif len(rows) % 25 == 0: print(f"  origins {len(rows)}/{len(keys)}")
    return pd.DataFrame(rows).set_index("key")

class DepthMap:
    """Spherical (equirectangular) range image of one scan seen from its scanner: D[pixel] = nearest return."""
    def __init__(self, P_local, origin, res=None, max_pts=None, seed=0):
        res = res or C["DEPTH_RES"]; max_pts = max_pts or C["DEPTH_MAX_PTS"]
        if len(P_local) > max_pts: P_local = P_local[np.random.default_rng(seed).choice(len(P_local), max_pts, replace=False)]
        if res == "auto":   # ~4 points per pixel on average: sparse fragments need coarser pixels or every pixel is empty
            res = float(np.clip(np.sqrt(4 * 41253.0 / max(len(P_local), 1)), 0.2, 3.0))
        self.o = np.asarray(origin, float); self.res = res
        r, iy, ix, shp = _sph(P_local - self.o, res)
        D = np.full(shp, np.inf, np.float32); np.minimum.at(D, (iy, ix), r.astype(np.float32))
        self.D = D
        self.Dmin = ndimage.minimum_filter(D, size=3, mode=["nearest", "wrap"])     # conservative: nearest in 3x3
        self.n = ndimage.uniform_filter(np.isfinite(D).astype(np.float32), size=3, mode=["nearest", "wrap"]) * 9
    def query(self, P_local):
        r, iy, ix, _ = _sph(P_local - self.o, self.res)
        return r, self.Dmin[iy, ix], self.n[iy, ix]
    def free(self, P_local, tau, margin_rel, min_returns=3):
        """True where this scanner measured a return BEYOND the point = it saw empty space there."""
        r, Dm, nret = self.query(P_local)
        return (nret >= min_returns) & (Dm > r + tau + margin_rel * r)

class NormalFreeSpace:
    """Fallback when a fragment has NO single scanner position (e.g. fused from a moving sensor).
    Every observed surface point q with normal n (oriented toward where the sensor was) certifies a short
    free-space tube in front of it: q + s*n for tau < s < L. A point inside such a tube was seen to be empty.
    Weaker than ray casting (only short range) but needs no viewpoint -- like the truncation band of a TSDF."""
    def __init__(self, entry, L=1.0, rho=None, k=24):
        self.e = entry; self.L = L; self.rho = rho or 2 * C["VOXEL"]; self.k = k
    def free(self, P_local, tau, margin_rel=0.0, min_returns=3):
        d, j = self.e["tree"].query(P_local, k=self.k, distance_upper_bound=self.L, workers=-1)
        ok = np.isfinite(d); j = np.where(ok, j, 0)
        D = P_local[:, None, :] - self.e["pts"][j]
        s_ = np.einsum("ijk,ijk->ij", D, self.e["nrm"][j])
        lat2 = np.einsum("ijk,ijk->ij", D, D) - s_ ** 2
        hit = ok & (s_ > tau) & (s_ < self.L) & (lat2 < self.rho ** 2)
        return hit.sum(1) >= min_returns

def visibility_model(key, cloud, origins, origin_table=None, max_selfocc=0.05):
    """Ray-casting depth map if the fragment has a credible single scanner position, else the normal fallback."""
    so = None
    if origin_table is not None and key in origin_table.index: so = float(origin_table.self_occ_used[key])
    if so is not None and so > max_selfocc:
        return NormalFreeSpace(cloud(key))
    return DepthMap(read_points(INDEX[key]), origins.get(key, np.zeros(3)))

def p2plane_dist(P, other):
    d, j = other["tree"].query(P, workers=-1)
    r = np.abs(np.einsum("ij,ij->i", P - other["pts"][j], other["nrm"][j]))
    return np.where(d > 3 * C["VOXEL"], d, r)

UNCHANGED, CHANGED, UNOBSERVED = 0, 1, 2
def classify(P, other, views, tau=None, margin_rel=None, min_returns=3):
    """Three-class change detection for points P (already in the frame of `other`).
       UNCHANGED  : a surface of the other epoch lies within tau (point-to-plane)
       CHANGED    : no such surface AND some scanner of the other epoch looked THROUGH this spot (free-space violation)
       UNOBSERVED : otherwise (occluded or out of view) -> we abstain instead of guessing.
    views: list of (DepthMap, T_frame_to_scan) for the other epoch's scanners."""
    tau = tau or C["TAU"]; mr = C["FS_MARGIN_REL"] if margin_rel is None else margin_rel
    d = p2plane_dist(P, other)
    lab = np.full(len(P), UNOBSERVED, np.int8); lab[d < tau] = UNCHANGED
    todo = np.flatnonzero(lab != UNCHANGED)
    for dm, T in views:
        if len(todo) == 0: break
        free = dm.free(tf(P[todo], T), tau, mr, min_returns)
        lab[todo[free]] = CHANGED
        todo = todo[~free]
    return lab, d

def classify_baseline(P, other, tau=None):
    tau = tau or C["TAU"]; d = p2plane_dist(P, other)
    return np.where(d < tau, UNCHANGED, CHANGED).astype(np.int8), d

def covisible(P, other_pts, cell=None):
    """Eroded XY co-coverage mask (same region definition for baseline and ours, so metrics compare like with like)."""
    cell = cell or 2 * C["FOOT_CELL"]; ax = hax()
    fp = footprint(other_pts, cell)
    inner = {c for c in fp if all((c[0] + i, c[1] + j) in fp for i in (-1, 0, 1) for j in (-1, 0, 1))}
    return np.array([tuple(c) in inner for c in np.floor(P[:, ax] / cell).astype(np.int64)])

# ---------- physically consistent object injection ----------
def _ray_box(o, P, lo, hi):
    """Entry/exit distances of rays o->P with an axis-aligned box (slab method)."""
    d = P - o; L = np.linalg.norm(d, axis=1) + 1e-12; u = d / L[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / u; t2 = (hi - o) / u
    tmin = np.nanmax(np.minimum(t1, t2), axis=1); tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return tmin, tmax, L

def box_surface(lo, hi, spacing, rng):
    pts, nrm = [], []
    for ax in range(3):
        for side, val in ((-1, lo[ax]), (1, hi[ax])):
            if ax == C["UP"] and side == -1: continue             # resting on the floor
            o_ = [a for a in range(3) if a != ax]
            n = max(20, int(np.prod(hi[o_] - lo[o_]) / spacing ** 2))
            X = np.zeros((n, 3)); X[:, ax] = val
            for a in o_: X[:, a] = rng.uniform(lo[a], hi[a], n)
            N = np.zeros((n, 3)); N[:, ax] = side
            pts.append(X); nrm.append(N)
    return np.vstack(pts), np.vstack(nrm)

def inject_object(P_raw, origin, lo, hi, rng, spacing=0.01):
    """Insert a box into a scan AS THE SCANNER WOULD SEE IT: only faces visible from the scanner are added and
    every existing return the box would hide is removed. Returns (new_raw_points, object_points)."""
    tmin, tmax, L = _ray_box(origin, P_raw, lo, hi)
    hidden = (tmax > tmin) & (tmin > 0) & (tmin < L - 0.01)
    S, _ = box_surface(lo, hi, spacing, rng)
    smin, smax, SL = _ray_box(origin, S, lo, hi)
    vis = np.abs(smin - SL) < 0.02
    obj = S[vis]
    return np.vstack([P_raw[~hidden], obj]), obj

# =====================================================================================
# 5. BUILDING-SCALE EVIDENCE FUSION
# =====================================================================================
def stage_map(keys, W, cloud, voxel=None):
    """All fragments of one stage in the common frame -> one voxelised map entry.
    Normals keep the per-scan orientation (towards the scanner that saw the point): FPFH features depend on normal
    direction, so re-estimating unoriented normals on the merged map breaks scan-to-map feature matching."""
    P = np.vstack([tf(cloud(k)["pts"], W[k]) for k in keys])
    N = np.vstack([cloud(k)["nrm"] @ W[k][:3, :3].T for k in keys])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pcd.normals = o3d.utility.Vector3dVector(N)
    pcd = pcd.voxel_down_sample(voxel or C["VOXEL"])                  # averages the normals inside each voxel
    Nm = np.asarray(pcd.normals).copy()
    est = o3d.geometry.PointCloud(pcd); est.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * C["VOXEL"], max_nn=30))
    Ne = np.asarray(est.normals)
    # clean local geometry from re-estimation, sign taken from the scanner-oriented average
    # (where opposite-facing normals cancelled out, ln ~ 0, the sign stays arbitrary: rare, thin walls only)
    Ne = Ne * np.where(np.einsum("ij,ij->i", Ne, Nm) < 0, -1.0, 1.0)[:, None]
    pcd.normals = o3d.utility.Vector3dVector(Ne)
    return finish_entry(pcd)

def fused_classify(P_world, other_map, views_world, tau=None, max_range=30.0):
    """Every scanner of the other epoch votes. Unchanged if any surface agrees; changed if any scanner saw through."""
    tau = tau or C["TAU"]
    lab, d = classify(P_world, other_map, [], tau)          # consistency only
    todo = np.flatnonzero(lab != UNCHANGED)
    for dm, Tw2s, ow in views_world:
        if len(todo) == 0: break
        near = np.linalg.norm(P_world[todo] - ow, axis=1) < max_range
        sub = todo[near]
        if len(sub) == 0: continue
        free = dm.free(tf(P_world[sub], Tw2s), tau, C["FS_MARGIN_REL"])
        lab[sub[free]] = CHANGED
        todo = np.setdiff1d(todo, sub[free], assume_unique=True)
    return lab, d

def floor_levels(e, scanner_z=None, bin_=0.05):
    """Storey floor heights. With scanner heights (recommended): for each group of scanners at a similar height,
    the floor is the strongest horizontal surface 0.5-3 m below them. Without: lowest strong horizontal peak."""
    up = C["UP"]; z = e["pts"][e["horiz"], up]
    if len(z) < 100: z = e["pts"][:, up]
    h, b = np.histogram(z, np.arange(z.min() - bin_, z.max() + 2 * bin_, bin_)); mid = (b[:-1] + b[1:]) / 2
    hs = ndimage.gaussian_filter1d(h.astype(float), 1)
    if scanner_z is None or len(scanner_z) == 0:
        pk = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and h[i] > 0.02 * len(z)]
        return [float(mid[min(pk)])] if pk else [float(np.percentile(z, 2))]
    groups = []
    for zz in np.sort(np.asarray(scanner_z)):
        if not groups or zz - groups[-1][-1] > 1.5: groups.append([zz])
        else: groups[-1].append(zz)
    out = []
    for g in groups:
        zs = np.median(g); m = (mid < zs - 0.5) & (mid > zs - 3.0)
        if m.any(): out.append(float(mid[m][np.argmax(hs[m])]))
    return out or [float(np.percentile(z, 2))]

# =====================================================================================
# 5b. ONE-CALL DATA PREPARATION (shared by every notebook, cached)
# =====================================================================================
PREP_VERSION = 3      # bump whenever the GT logic changes, so stale caches are never reused

def prepare(data_root, extract_dir, work, building, overrides=None, prep_from=None, min_ov=0.3, min_geom=0.10,
            origins=True, verbose=True):
    """Load NSS, verify the GT convention, chain poses, resolve scanner origins, generate same/cross-stage pairs.
    Cached in <work>/nss_prep/prep.pkl; pass prep_from=<folder with prep.pkl> to reuse a teammate's run."""
    import pickle, shutil
    pdir = Path(work) / "nss_prep"; pdir.mkdir(parents=True, exist_ok=True); f = pdir / "prep.pkl"
    def _bldg(path):
        try: return pickle.loads((Path(path) / "prep.pkl").read_bytes()).get("building")
        except Exception: return None
    if prep_from and (Path(prep_from) / "prep.pkl").exists() and _bldg(prep_from) != building:
        print(f"!! PREP_FROM holds a prep for {_bldg(prep_from)}, not {building}: ignored"); prep_from = None
    if (not prep_from or not (Path(prep_from) / "prep.pkl").exists()):
        hits = sorted(Path(data_root).rglob("prep.pkl")) if Path(data_root).exists() else []
        for h in hits:                       # take the first prep.pkl made for THIS building (several may be attached)
            if _bldg(h.parent) == building:
                prep_from = h.parent; print("found prep.pkl for", building, "automatically in", prep_from); break
        else:
            if hits: print(f"!! {len(hits)} prep.pkl attached, none made for {building}: the pair list will be rebuilt "
                           "(its order, and therefore the 240 selected pairs, can differ from other runs)")
    if prep_from and (Path(prep_from) / "prep.pkl").exists() and not f.exists():
        shutil.copy(Path(prep_from) / "prep.pkl", f); print("using prep from", prep_from)
    D = load_nss(data_root, extract_dir, overrides, verbose=verbose)
    cloud = CloudCache()
    if f.exists():
        R = pickle.loads(f.read_bytes())
        if R.get("building") == building and R.get("voxel") == C["VOXEL"] and R.get("version") == PREP_VERSION:
            cloud.origins = R["origins"]
            if verbose: print(f"loaded cached prep: {len(R['pairs'])} pairs, {len(R['W'])} posed fragments")
            R.update(D=D, cloud=cloud); return R
    t0 = time.time()
    conv, fit, conv_table = check_convention(D, building, cloud, verbose=verbose)
    W, comp, info = world_poses(D, building, conv, cloud, verbose)
    if verbose: print("pose info:", info)
    org = None
    if origins:
        if verbose: print("resolving scanner origins ...")
        org = resolve_origins(list(W), cloud, verbose=False)
        if verbose: print(org.origin_src.value_counts().to_dict(), "| median self-occlusion used:", round(org.self_occ_used.median(), 4))
    pairs = make_pairs(D, building, W, comp, cloud, min_ov=min_ov, min_geom=min_geom, verbose=verbose)
    R = dict(version=PREP_VERSION, building=building, voxel=C["VOXEL"], edges=D.get("edges_scored"), conv=conv, conv_fit=fit, conv_table=conv_table, W=W, comp=comp,
             pose_info=info, origins=dict(cloud.origins), origin_table=org, pairs=pairs, minutes=(time.time() - t0) / 60)
    f.write_bytes(pickle.dumps(R))
    if verbose: print(f"prep done in {R['minutes']:.1f} min -> {f}")
    R.update(D=D, cloud=cloud); return R

def scanner_world(R, key):
    """Scanner position of a fragment in the common frame."""
    return tf(R["origins"].get(key, np.zeros(3))[None], R["W"][key])[0]

# =====================================================================================
# 6. PLOTTING HELPERS
# =====================================================================================
LAB_COLORS = {UNCHANGED: "#b9bec7", CHANGED: "#d62728", UNOBSERVED: "#7fb3d5"}
METHOD_COLORS = {"ransac6": "#8c8c8c", "fgr": "#c7a76c", "ransac4 (ours)": "#6aa84f", "bev-xc (ours)": "#3d85c6",
                 "bev+r4 (ours)": "#1b5e20", "icp": "#4c78a8", "tukey": "#e45756", "trimmed": "#f58518",
                 "B v2: gnc+polish (ours)": "#2ca02c", "GT": "#222222"}

def topdown(ax, P, c, s=0.15, alpha=0.6, zband=None, title=None, rasterized=True):
    a0, a1 = hax(); P = np.asarray(P)
    if zband is not None:
        m = (P[:, C["UP"]] > zband[0]) & (P[:, C["UP"]] < zband[1]); P = P[m]
        c = np.asarray(c)[m] if not isinstance(c, str) and np.ndim(c) > 0 and len(np.atleast_1d(c)) == len(m) else c
    ax.scatter(P[:, a0], P[:, a1], c=c, s=s, alpha=alpha, linewidths=0, rasterized=rasterized)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=9)

def oblique(ax, P, colors, elev=35, azim=-60, s=0.3, n=120000, seed=0):
    """Fast pseudo-3D view: rotate, painter-sort by depth, 2D scatter (much faster than mplot3d)."""
    P = np.asarray(P); colors = np.asarray(colors)
    if len(P) > n:
        k = np.random.default_rng(seed).choice(len(P), n, replace=False); P, colors = P[k], colors[k]
    a, e = np.radians(azim), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    X = (P - P.mean(0)) @ (Rx @ Rz).T
    o = np.argsort(X[:, 1])[::-1]
    ax.scatter(X[o, 0], X[o, 2], c=colors[o], s=s, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")

def save(fig, path, dpi=170):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); return path

class _Tee:
    """Copy everything printed into a log file as well (so the upload zip contains the full console output)."""
    def __init__(self, stream, path): self.s = stream; self.f = open(path, "a", buffering=1)
    def write(self, x):
        try: self.f.write(x)
        except Exception: pass
        return self.s.write(x)
    def flush(self):
        try: self.f.flush()
        except Exception: pass
        self.s.flush()
    def __getattr__(self, k): return getattr(self.s, k)

def start_log(out):
    import sys
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    base = sys.stdout.s if isinstance(sys.stdout, _Tee) else sys.stdout
    sys.stdout = _Tee(base, out / "console_log.txt")
    print(f"\n===== log started {time.strftime('%Y-%m-%d %H:%M:%S')} =====")

def bundle(work, nb, out, extra=()):
    """One zip with everything this notebook produced (figures, CSVs, JSON, console log) for sharing/verification."""
    import sys
    work, out = Path(work), Path(out)
    if isinstance(sys.stdout, _Tee): sys.stdout.flush()
    meta = dict(notebook=nb, finished=time.strftime("%Y-%m-%d %H:%M:%S"), config={k: v for k, v in C.items()})
    (out / "run_info.json").write_text(json.dumps(meta, indent=1, default=str))
    zpath = work / f"UPLOAD_{nb}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file(): z.write(f, f.relative_to(out.parent))
        for f in extra:
            f = Path(f)
            if f.exists(): z.write(f, f"{nb}/extra/{f.name}")
    mb = zpath.stat().st_size / 1e6
    print(f"\n>>> UPLOAD THIS FILE: {zpath}  ({mb:.1f} MB)")
    print("    Kaggle: right panel -> Output -> /kaggle/working -> click the file to download.")
    return zpath

def append_csv(path, rows):
    path = Path(path); pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)
def done_keys(path, cols):
    path = Path(path)
    if not path.exists(): return set()
    d = pd.read_csv(path); return set(map(tuple, d[cols].astype(str).values))
