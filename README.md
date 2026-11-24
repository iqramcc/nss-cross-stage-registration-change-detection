# Registration and visibility-aware change detection across construction stages (NSS)

Course project, CV5570 Computer Vision, IIT Madras.
Authors: Mohammed Ikram C C, Nizamuddin, Hathim Hamza Ahmed (Department of Civil Engineering).

We align indoor 3-D scans of buildings under construction, taken at different stages, and detect what changed between them, on the public [Nothing Stands Still (NSS)](https://nothing-stands-still.com) benchmark: six building areas and 4,829 scan fragments, which NSS generates from Matterport RGB-D tripod scans.

* **Registration:** a gravity prior (levelled tripod sensors, so a 4-DoF RANSAC), an exhaustive bird's-eye-view cross-correlation over wall points (BEV-XC), symmetric wall-agreement verification, and GNC point-to-plane ICP. Compared with FPFH + RANSAC + ICP, PREDATOR and GeoTransformer (pretrained and fine-tuned), and hybrids that feed learned correspondences into our gravity-aware search.
* **Change detection:** a point is *changed* only if the other sensor saw through it, *unobserved* if it was hidden, and *unchanged* if a surface is within 15 cm. Compared with cloud-to-cloud, point-to-plane, voxel-occupancy differencing and M3C2.

Demo video: [add the YouTube link here]

## Main results (registration recall under the NSS criterion: rotation < 10° and translation < 0.2 m, 120 pairs per cell, from a random heading)

| Building | Same stage: standard → ours | Cross stage: standard → ours |
|---|---|---|
| Bldg1 | 60.8 → 84.2 % | 15.0 → 43.3 % |
| Bldg2 | 63.3 → 83.3 % | 22.5 → 57.5 % |
| Bldg3 | 50.8 → 62.5 % | 20.8 → 49.2 % |
| Bldg4 | 51.7 → 69.2 % | 31.7 → 45.8 % |
| Bldg5 | 60.8 → 78.3 % | 45.0 → 62.5 % |
| Bldg6 | 62.5 → 82.5 % | 42.5 → 71.7 % |

Fine success (< 10 cm and < 1°) is within 1 point of these numbers. Ours wins in every building (paired sign test, p < 0.003).

GeoTransformer fine-tuned on Bldg1, 2 and 6 is stronger in three of four test settings. Inside our gravity-aware search, its correspondences give the best cross-stage result on Bldg5 (89.2 %) and are on par with ours on Bldg3 (55.0 vs 53.3 %).

Change detection (all six buildings, unchanged same-stage pairs): median false changes fall from 8.8–15.3 % (point-to-plane) to 0.7–2.5 % (ours), with 8–14 % of points left unobserved. Full tables are in the report and in `results/`.

## Repository layout

```
src/nss_lib.py              shared library: data loading, benchmark, all registration and change-detection methods
src/register_two_scans.py   minimal script: register two .ply scans (standard vs ours)
notebooks/                  Kaggle notebooks (each embeds nss_lib.py, so it runs stand-alone)
results/                    summary CSVs of the runs reported in the paper
```

The notebooks are stored **without outputs**. All numbers in the report come from the CSVs in `results/` (one folder per experiment).

| Notebook | What it does | Hardware | Inputs |
|---|---|---|---|
| NB0_prep | load NSS, check transform convention, chain poses, sensor origins, candidate pairs | CPU | NSS |
| NB1_registration | 240-pair benchmark per building: standard vs ours, ablation, refiners, reference audit | CPU | NSS |
| NB2_change_detection | visibility-aware change detection: controls, simulated objects, error sweep | CPU | NSS |
| NB3_building_maps | building-scale change maps by multi-scan evidence fusion, scan-to-map | CPU | NSS |
| NB4_predator / NB5_finetune_predator | PREDATOR pretrained / fine-tuned on Bldg1, 2, 6; hybrid with ours | GPU | NSS (+ NB1 results) |
| NB6_failure_diagnostic | selection vs generation failures, abstention | CPU | NSS (+ NB1 results) |
| NB7_plane_check | floor and ceiling plane check of cross-stage references (no ICP) | CPU | NSS |
| NB8_Evaluation_Check | wall-fit check of every disagreement between methods on the real scans | CPU | NSS + NB1 results |
| NB9_Change_Baselines | C2C, point-to-plane and voxel baselines vs ours on the same points | CPU | NSS |
| NB10_Visibility_Verification | larger candidate pool + visibility-consistency verification | CPU | NSS + NB1 results |
| NB11_GeoTransformer | pretrained GeoTransformer and GeoTransformer + ours | GPU | NSS + NB1 results |
| NB12_Real_Change_Validation | detectors vs the NSS temporal change ratio on real pairs, all buildings | CPU | NSS |
| NB13_GeoTransformer_Finetune | fine-tune GeoTransformer on Bldg1, 2, 6 and test on Bldg3, 5 | GPU | NSS + NB1 results |
| NB14_Scan_To_Map_Bldg3 / _Bldg5 | single-scan vs submap relocalisation against the previous stage's map | CPU | NSS |
| NB15_M3C2_py4dgeo | M3C2 baseline (py4dgeo reference implementation) vs ours | CPU | NSS |
| NB16_Multiway_Registration | multi-way registration of up to 160 scans per building with a pose graph | CPU | NSS |
| NB17_Temporal_Consistency | label-free consistency of change verdicts over three stages | CPU | NSS |
| NB18_End_to_End | fine-tuned GeoTransformer + our search, followed by our change detection | GPU | NSS + `geot_nss_ft.pth` |

**Multi-building runners.** `NB2_Bldg1_Bldg2`, `NB2_Bldg4_Bldg6`, `NB3_Bldg1_Bldg2`, `NB3_Bldg4_Bldg6`, `NB3_Bldg6` and `NB6_Bldg1_Bldg2_Bldg4_Bldg6` run the unchanged code of NB2, NB3 and NB6 on several buildings in one Kaggle session (one output zip per building). `NB3_Bldg6` uses 200 instead of 400 scans per stage to stay within Kaggle's limits.

## How to run (Kaggle)
1. Add the NSS dataset as an input (and, where listed, a dataset with the NB1 result CSVs `registration_v3.csv` and `reference_audit_v3.csv`).
2. Settings: Internet **On**; Accelerator **None** (CPU notebooks) or **GPU T4** (NB4, NB5, NB11, NB13, NB18).
3. NB0 to NB7: set `BUILDING` and `QUICK = False` in the CONFIG cell. NB8 to NB18 need no edits.
4. *Save Version → Save & Run All*. Each notebook writes one `UPLOAD_*.zip` with CSVs, figures and the console log.

Locally: `pip install open3d numpy scipy pandas matplotlib` (plus `torch` for the learned methods), then `python src/register_two_scans.py source.ply target.ply`.

## Benchmark protocol
240 pairs per building (120 same-stage, 120 cross-stage), stratified by overlap. Each source scan is centred and given a random heading; sensor positions are hidden from all methods. Results are scored against an ICP-audited version of the official NSS transforms; coarse results under the unmodified official transforms differ by at most 1 point (checked on Bldg3 and Bldg5). Paired sign tests compare methods on identical pairs.

## Third-party code
PREDATOR (prs-eth/OverlapPredator) and GeoTransformer (qinzheng93/GeoTransformer) are cloned at run time under their own licences; the notebooks apply small compatibility patches for current Python and PyTorch. The NSS data is not redistributed here.
