# Execution and data requirements

Run the model scripts from `raster_fusion_ai/`. The saved YAML configurations
preserve the settings of the manuscript runs. Update filesystem paths only when
supplying a compatible private dataset. For example:

```sh
python run_enriched_linear.py --config configs/enriched_linear.yaml --check-only
python run_enriched_linear.py --config configs/enriched_linear.yaml
```

`--check-only` still requires the data and its manifest. It is not a data-free
demonstration. The supplied synthetic tests are the data-free validation route.

Expected enriched inputs use `treinamento` and `teste_final`, years 2019–2022,
and four phenomenon names: `Garimpos`, `Ocorrencias_Minerais`,
`Pistas_Regulares`, `Pistas_Irregulares`. Under each region/year, inputs are
`inputs/<phenomenon>/{original,magnitude,direcao_seno,direcao_cosseno,conectividade}.tif`;
graph support is `auxiliares/<phenomenon>/suporte.tif`. Development targets are
`targets/target.tif`. The dataset manifest records external targets under
`config.evaluation_targets`; its file inventory and hashes are checked by the
implementation. `organize_enriched_data.py` creates that manifest from private
source inputs. No original manifest or coordinate records are distributed.

All layers must have compatible georeferencing and valid-data definitions.
Zeros and NoData are distinct. See the checks in `src/services/enriched_linear.py`
for the precise contract; do not replace missing private data with invented results.

Control scripts require the corresponding main-run winners and caches. The
post-hoc analyses and figure scripts require saved predictions, metrics, search
histories or detection points. These artifacts are not bundled. Their historical
run identifiers are preserved to link the code to the article.

The four analysis entry points that used an absolute workspace path now locate
the project relative to this package. Historical source snapshots remain unchanged.
The original folder geometry
(`raster_fusion_ai/`, `outputs/`, `work/`) is retained to preserve relative paths.
Figure scripts may additionally require original supplementary CSVs or LaTeX
formatting assets, which are not part of the code-only dataset exclusion policy.

The package permits inspection of the implementation and synthetic tests without
the data. It cannot independently reproduce the published numerical results.
