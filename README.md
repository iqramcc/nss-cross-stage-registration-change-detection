# Registration and visibility-aware change detection across construction stages (NSS)

Course project, CV5570 Computer Vision, IIT Madras.

We align terrestrial laser scans of buildings under construction, taken at different stages, and detect what changed between them, on the public [Nothing Stands Still (NSS)](https://nothing-stands-still.com) benchmark (6 buildings, 4,829 scans).

* **Registration:** a gravity prior (levelled scanners → 4-DoF RANSAC), an exhaustive bird's-eye-view cross-correlation over wall points (BEV-XC), symmetric wall-agreement verification, and GNC point-to-plane ICP. Compared with FPFH + RANSAC + ICP, PREDATOR and GeoTransformer (pretrained and fine-tuned), and hybrids that feed learned correspondences into our gravity-aware search.
* **Change detection:** a point is *changed* only if the other scanner saw through it, *unobserved* if it was hidden, *unchanged* if a surface is within 15 cm. Compared with cloud-to-cloud, point-to-plane and voxel-occupancy differencing.

## Main results (coarse success: < 0.5 m and < 5°, 120 pairs per cell, from a random heading)

| Building | Same stage: standard → ours | Cross stage: standard → ours |
|---|---|---|
| Bldg1 | 60.8 → 84.2 % | 15.0 → 45.8 % |
| Bldg2 | 64.2 → 85.0 % | 23.3 → 60.8 % |
| Bldg3 | 52.5 → 62.5 % | 22.5 → 53.3 % |
| Bldg4 | 51.7 → 70.0 % | 57.5 → 76.7 % |
| Bldg5 | 60.8 → 79.2 % | 55.0 → 75.8 % |
| Bldg6 | 62.5 → 83.3 % | 42.5 → 71.7 % |

GeoTransformer fine-tuned on Bldg1/2/6 is stronger still in 3 of 4 test settings; its correspondences inside our gravity-aware search give the best cross-stage results (Bldg3 55.0 %, Bldg5 89.2 %).

Change detection (all six buildings, unchanged same-stage pairs): median false changes 8.8–15.3 % (point-to-plane) → 0.7–2.5 % (ours), with 8–14 % of points left unobserved. Full tables in `results/` and in the report.

## Repository layout

```
src/nss_lib.py          shared library: data loading, benchmark, all registration and change-detection methods
src/register_two_scans.py   minimal script: register two .ply scans (standard vs ours)
notebooks/              Kaggle notebooks (each embeds nss_lib.py, so it runs stand-alone)
tools/                  scripts that generate the notebooks (nbX_src.py + build.py, build_nbX.py)
results/                summary CSVs of the runs reported in the paper
```

| Notebook | What it does | Hardware | Inputs |
|---|---|---|---|
| NB0_prep | load NSS, check transform convention, chain poses, scanner origins, candidate pairs | CPU | NSS |
| NB1_registration | 240-pair benchmark per building: standard vs ours, ablation, refiners, reference audit | CPU | NSS |
| NB2_change_detection (+ NB2_Bldg1_Bldg2, NB2_Bldg4_Bldg6) | visibility-aware change detection: controls, injected objects, error sweep | CPU | NSS |
| NB3_building_maps | building-scale change maps by multi-scan evidence fusion, scan-to-map | CPU | NSS |
| NB4_predator / NB5_finetune_predator | PREDATOR pretrained / fine-tuned on Bldg1/2/6, hybrid with ours | GPU | NSS (+ NB1 results) |
| NB6_failure_diagnostic | selection vs generation failures, abstention | CPU | NSS + NB1 results |
| NB7_plane_check | floor/ceiling plane check of cross-stage references (no ICP) | CPU | NSS |
| NB8_Evaluation_Check | wall-fit check of every disagreement between methods on the real scans | CPU | NSS + NB1 results |
| NB9_Change_Baselines | C2C, point-to-plane, voxel, M3C2-style vs ours on the same points | CPU | NSS |
| NB10_Visibility_Verification | larger candidate pool + visibility-consistency verification | CPU | NSS + NB1 results |
| NB11_GeoTransformer | pretrained GeoTransformer and GeoTransformer + ours | GPU | NSS + NB1 results |
| NB12_Real_Change_Validation | detectors vs NSS temporal change ratio on real pairs, all buildings | CPU | NSS |
| NB13_GeoTransformer_Finetune | fine-tune GeoTransformer on Bldg1/2/6, test on Bldg3/5 | GPU | NSS + NB1 results |
| NB14_Scan_To_Map | single-scan vs submap relocalisation against the previous stage's map | CPU | NSS |

## How to run (Kaggle)
1. Add the NSS dataset as an input (and, where listed, a dataset with the NB1 result CSVs `registration_v3.csv` / `reference_audit_v3.csv`).
2. Settings: Internet **On**; Accelerator **None** (CPU notebooks) or **GPU T4** (NB4, NB5, NB11, NB13).
3. NB0–NB7: set `BUILDING` and `QUICK = False` in the CONFIG cell. NB8–NB14 need no edits.
4. *Save Version → Save & Run All*. Each notebook writes one `UPLOAD_*.zip` with CSVs, figures and the console log.

Locally: `pip install open3d numpy scipy pandas matplotlib` (plus `torch` for the learned methods), then `python src/register_two_scans.py source.ply target.ply`.

## Benchmark protocol
240 pairs per building (120 same-stage, 120 cross-stage), stratified by overlap. Each source scan is centred and given a random heading; scanner positions are hidden from all methods. Results are scored against the official NSS transforms and an ICP-refined version of them; coarse conclusions are identical under both. Paired sign tests compare methods on identical pairs.

## Third-party code
PREDATOR (prs-eth/OverlapPredator) and GeoTransformer (qinzheng93/GeoTransformer) are cloned at run time under their own licences; the notebooks apply small compatibility patches for current Python/PyTorch. The NSS data is not redistributed here.
