# Manuscript-to-code index — V7

H0 uses same-year inputs; H1 uses preceding-year inputs. Each family includes
the input-set comparisons and internal selection implemented by its runner.

| ID | Family | Entry point (under `raster_fusion_ai/`) | Saved configuration |
|---|---|---|---|
| 01 | Linear | `run_enriched_linear.py` | `configs/enriched_linear.yaml` |
| 02 | Ridge | `run_enriched_ridge.py` | `configs/enriched_ridge.yaml` |
| 03 | HGB | `run_enriched_hgb.py` | `configs/enriched_hgb.yaml` |
| 04 | XGBoost | `run_enriched_xgboost.py` | `configs/enriched_xgboost.yaml` |
| 05 | Random Forest | `run_enriched_random_forest.py` | `configs/enriched_random_forest.yaml` |
| 06 | Extra Trees | `run_enriched_extra_trees.py` | `configs/enriched_extra_trees.yaml` |
| 07 | MLP | `run_enriched_mlp.py` | `configs/enriched_mlp.yaml` |
| 08 | CNN | `run_enriched_cnn.py` | `configs/enriched_cnn.yaml` |
| 09 | ResNet | `run_enriched_resnet.py` | `configs/enriched_resnet.yaml` |
| 10 | U-Net | `run_enriched_unet.py` | `configs/enriched_unet.yaml` |
| 11 | ResUNet | `run_enriched_resunet.py` | `configs/enriched_resunet.yaml` |
| 12 | Inception | `run_enriched_inception.py` | `configs/enriched_inception.yaml` |

| Control | Code | Manuscript scope |
|---|---|---|
| 08A | `run_enriched_cnn_local_control.py` | Local-only CNN |
| 09A | `run_enriched_resnet_local_control.py` | Local-only ResNet |
| 10A–10C | `run_enriched_unet_controls.py` | Core sampling, depth and gradient components |

Main model fits and external evaluations correspond to Supplementary Tables
S1–S2 and S8; local controls to S3; U-Net controls to S4–S5 and S7.
The V7 count is 176 final fitted configurations and 127 external evaluation
records, with four additional spatial-exclusion fits reported separately.

| Reported method/analysis | Code location |
|---|---|
| Gradient and proximity graph descriptors | `raster_fusion_ai/generate_gradient_inputs.py`, `generate_graph_inputs.py` |
| Enriched input organization | `raster_fusion_ai/organize_enriched_data.py` |
| Complementary external H1 pair for families 01–04 | `raster_fusion_ai/evaluate_enriched_target_2021.py` |
| Reference predictors, capture and spatial-block bootstrap | `raster_fusion_ai/src/services/enriched_linear.py` and shared evaluation code |
| Learning dynamics and representation results | `outputs/build_input_results_20261007.py` |
| Detection-point hotspot coverage and persistence | `outputs/Aerospace_pontos_previsoes_20261007/` |
| Northwestern external quadrant input support | `outputs/Aerospace_quadrante_externo_20261007/` |
| Spatial exclusion: paired 0.75 km / 41 km | `work/buffer_sensitivity_20261010/` |
| Frozen-result synthesis and supplementary tables/figures | `raster_fusion_ai/build_final_interpretation.py`, `outputs/mdpi_acrescimos_20261004/`, `outputs/build_supplementary_20261005.py` |

KDE inputs in this study were prepared outside the model-training pipeline.
The point audit checks compatibility with the cumulative quartic KDE definition;
it does not claim bitwise reproduction of the historical QGIS rasters.

Some support modules retain historical functions because the selected code
imports them. Their presence does not add pilot experiments to the study scope.
