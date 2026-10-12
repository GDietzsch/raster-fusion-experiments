# Raster regression algorithms

Algorithms from the Aerospace V7 article and supplementary material. **Code only: study data, trained weights and prediction maps are not distributed.**

| Folder | Algorithm |
|---|---|
| [01](01/) | Linear regression |
| [02](02/) | Ridge |
| [03](03/) | Histogram gradient boosting |
| [04](04/) | XGBoost |
| [05](05/) | Random Forest |
| [06](06/) | Extra Trees |
| [07](07/) | MLP |
| [08](08/) | CNN |
| [08A](08A/) | Local-only CNN |
| [09](09/) | Compact ResNet |
| [09A](09A/) | Local-only ResNet |
| [10](10/) | U-Net |
| [10A](10A/) | U-Net: fixed/rotating cores |
| [10B](10B/) | U-Net: network depth |
| [10C](10C/) | U-Net: gradient components |
| [11](11/) | ResUNet |
| [12](12/) | Dense Inception |

## Install

Use Python 3.11 or newer. From the repository root:

```sh
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA environment. The dense models require substantial memory; their configuration controls GPU requirements.

## Run

Edit the chosen folder's `config.yaml` to point to **your own data**, then run, for example:

```sh
python 01/run.py --check-only
python 01/run.py
```

The check also needs your input files. Keep the repository together: algorithms share routines in `_support/`.

## Input format

The pipeline expects aligned, single-band GeoTIFFs and a dataset `manifest.json`. Set `data_root` to your dataset directory and `output_root` to a writable results directory. Relative paths are resolved from the configuration file.

```text
<data_root>/<region>/<year>/inputs/<theme>/<product>.tif
<data_root>/<region>/<year>/auxiliares/<theme>/suporte.tif
<data_root>/treinamento/<year>/targets/target.tif
```

- Regions: `treinamento` and `teste_final`; configured years: 2019–2022.
- Themes: `Garimpos`, `Ocorrencias_Minerais`, `Pistas_Regulares`, `Pistas_Irregulares`.
- Products: `original`, `magnitude`, `direcao_seno`, `direcao_cosseno`, `conectividade`; graph support is `suporte`.
- The manifest records file hashes and external targets in `config.evaluation_targets`. The preparation helper `_support/raster_fusion_ai/organize_enriched_data.py` creates it from your original rasters and gradient/graph outputs (`--help` lists its paths).
- Distinguish valid zero values from NoData. All layers must have compatible CRS, resolution and grid alignment.

Update any `sample_source` or `source_run` to your own generated run. CNN and dense-model pipelines reuse the samples created by 07; local controls depend on 08/09, and U-Net controls depend on 10.

Shared routines, synthetic software tests, supplementary analysis scripts and historical code are retained in `_support/`. No study observations are included. No distribution license has been specified by the authors.
