#!/usr/bin/env python3
"""Register two laser scans and write the 4x4 transformation matrix (source -> target).

    python register_two_scans.py SOURCE.ply TARGET.ply --out result/            # our pipeline
    python register_two_scans.py SOURCE.ply TARGET.ply --out result/ --compare  # also the standard pipeline

Needs: python 3.9+, open3d, numpy, scipy (and matplotlib for the preview picture), and nss_lib.py next to this file.

Assumptions (true for NSS and most tripod scans; say so if you use other data):
  * each scan is levelled: the z axis points up (our gravity prior);
  * the two scans overlap (share some walls); metres as unit;
  * the scans may start at any heading and any position: that is what we recover.

Outputs in --out:
  transformation.txt     4 plain rows: paste into CloudCompare (Edit > Apply transformation) on the SOURCE cloud
  transformation.json    same matrix + scores
  registered_source.ply  the source scan moved onto the target (full resolution)
  merged_preview.ply     source (red) + target (blue) together, open in CloudCompare
  preview.png            top-down before / after
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nss_lib as L                                   # noqa: E402
import open3d as o3d                                  # noqa: E402


def load(path):
    return L.read_points(path)


def register(P_src, P_tgt, voxel=0.05, standard=False, log=print):
    """Return (T 4x4 source->target, info dict). P_* are Nx3 arrays (metres, z up)."""
    L.configure(VOXEL=voxel, SEED=0)
    t0 = time.time()
    S = L.make_entry(P_src, voxel=voxel)
    Tg = L.make_entry(P_tgt, voxel=voxel)
    log(f"  voxelised: source {len(S['pts'])} pts, target {len(Tg['pts'])} pts ({time.time()-t0:.1f}s)")
    t1 = time.time()
    if standard:
        T0 = L.reg_ransac6(S, Tg)                     # FPFH + RANSAC with 6 unknowns
        T = L.refine(S, Tg, T0, "icp", 6)             # plain ICP
        name = "standard (FPFH + RANSAC-6DoF + ICP)"
    else:
        T0 = L.reg_ours_global(S, Tg)                 # gravity prior: RANSAC-4DoF + wall search, verified
        T = L.refine(S, Tg, T0, "gnc", 4)             # robust refinement
        name = "ours (gravity prior + wall search + verification + robust refinement)"
    v = voxel
    fit = L._fitness(S["pts"][S["wall"]], Tg, T, 1.5 * v)          # share of source wall points that land on a target surface
    fit_rev = L._fitness(Tg["pts"][Tg["wall"]], S, L.inv(T), 1.5 * v)
    info = dict(method=name, seconds=round(time.time() - t1, 1),
                wall_overlap_source_on_target=round(float(fit), 3), wall_overlap_target_on_source=round(float(fit_rev), 3),
                yaw_deg=round(float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), 2),
                shift_xyz_m=[round(float(x), 3) for x in T[:3, 3]])
    return T, info


def write_outputs(out, P_src_full, P_tgt_full, T, info, preview=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "transformation.txt").write_text("\n".join(" ".join(f"{x:.8f}" for x in row) for row in T) + "\n")
    (out / "transformation.json").write_text(json.dumps(dict(matrix=T.tolist(), direction="source -> target", **info), indent=1))
    reg = L.tf(P_src_full, T)
    def pc(P, col):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); p.paint_uniform_color(col); return p
    o3d.io.write_point_cloud(str(out / "registered_source.ply"), pc(reg, [0.84, 0.15, 0.16]))
    o3d.io.write_point_cloud(str(out / "merged_preview.ply"), pc(reg, [0.84, 0.15, 0.16]) + pc(P_tgt_full, [0.24, 0.52, 0.78]))
    if preview:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            rng = np.random.default_rng(0); sub = lambda P: P[rng.choice(len(P), min(len(P), 60000), replace=False)]
            fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
            for a, (S_, ttl) in zip(ax, ((sub(P_src_full), "before: as given"), (sub(reg), "after: source moved onto target"))):
                a.scatter(*sub(P_tgt_full)[:, :2].T, s=0.2, c="#3d85c6", alpha=.5); a.scatter(*S_[:, :2].T, s=0.2, c="#d62728", alpha=.5)
                a.set_aspect("equal"); a.set_title(ttl); a.set_xticks([]); a.set_yticks([])
            fig.suptitle("blue = target, red = source (top-down; red lying on blue = aligned)"); fig.tight_layout()
            fig.savefig(out / "preview.png", dpi=130); plt.close(fig)
        except Exception as e:                       # the picture is optional
            print("  (preview picture skipped:", type(e).__name__, ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source"); ap.add_argument("target"); ap.add_argument("--out", default="result")
    ap.add_argument("--voxel", type=float, default=0.05, help="working resolution in metres (default 0.05)")
    ap.add_argument("--compare", action="store_true", help="also run the standard pipeline and save it in OUT/standard/")
    ap.add_argument("--standard", action="store_true", help="run only the standard pipeline")
    a = ap.parse_args()
    print("reading scans ...")
    Ps, Pt = load(a.source), load(a.target)
    print(f"  source {len(Ps)} points, target {len(Pt)} points")
    T, info = register(Ps, Pt, a.voxel, standard=a.standard)
    write_outputs(a.out, Ps, Pt, T, info)
    print("\n" + info["method"]); print(json.dumps({k: v for k, v in info.items() if k != "method"}))
    print("\nsource -> target 4x4 matrix:\n" + "\n".join(" ".join(f"{x:12.8f}" for x in row) for row in T))
    if a.compare and not a.standard:
        T2, info2 = register(Ps, Pt, a.voxel, standard=True)
        write_outputs(Path(a.out) / "standard", Ps, Pt, T2, info2)
        print("\n" + info2["method"]); print(json.dumps({k: v for k, v in info2.items() if k != "method"}))
    print(f"\nwrote {a.out}/ (transformation.txt, registered_source.ply, merged_preview.ply, preview.png)")
    print("Check preview.png and merged_preview.ply by eye: a low wall overlap or walls that do not coincide means the pair needs a manual look.")


if __name__ == "__main__":
    main()
