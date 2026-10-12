# Raster regression experiments — Aerospace V7

Code-only review package for the Aerospace V7 manuscript and its supplementary
material (version of 11 October 2026). The scope is limited to
experiments and analyses reported in that version. Earlier pilot experiments,
future work, private observations, rasters, fitted weights and prediction outputs
are excluded.

Start with [the manuscript-to-code index](docs/EXPERIMENTS.md),
[execution and data requirements](docs/REPRODUCIBILITY.md), and
[packaging notes](docs/PACKAGING.md).

## Review without the study data

Read the model implementations, saved configurations, spatial/temporal splitting,
internal selection rules, metrics and diagnostic controls. The selected existing
tests generate synthetic arrays and rasters locally. They validate software
behavior; they do not reproduce the manuscript's numerical results.

From `raster_fusion_ai/`, use a Python environment with the dependencies in
`requirements.txt` and PyTorch installed, then run:

```sh
python -m pytest tests/test_splits.py tests/test_enriched_linear.py -q
```

The original lock file records a Windows/Python-specific environment. It is kept
for provenance, not presented as a portable installation guarantee.

The study data are not distributed. Full result reproduction requires the
original inputs and saved run artifacts. Model weights and geospatial outputs
are also excluded from this code-only package.

Repository: [GDietzsch/raster-fusion-experiments](https://github.com/GDietzsch/raster-fusion-experiments).
No distribution license has been specified by the authors.
