# Packaging and provenance

This package selects entry points for the V7 reported experiments and analyses,
then follows local Python imports to include supporting modules. It includes
existing synthetic tests for those models, gradients/graphs and spatial splitting.
No new model fitting was performed when assembling the package.

Main model configurations are taken from the exact run identities used by the
manuscript synthesis. `provenance/source_manifest.csv` maps packaged files to
their original relative location and SHA-256, with separate source and packaged
hashes. Four analysis scripts were changed only to replace absolute workspace
paths with paths relative to the package. `provenance/main_runs.csv` records
the twelve main run identities. Relevant files available in saved source snapshots
are preserved under `provenance/frozen_source/`; these are historical evidence,
not a second installation or a claim that each snapshot contains a complete runtime.
The working implementation may include later fixes; compare its hashes with the
historical snapshot when auditing a specific fit.

The manuscript itself, input datasets, original dataset manifests, coordinates,
weights, prediction maps, numerical result tables and local environment folders
are not copied. Paths and historical run IDs appearing in source code are metadata,
not bundled datasets. Figure-generation scripts retain their original output logic.

The package is prepared for public code review at
https://github.com/GDietzsch/raster-fusion-experiments.
The distribution license must be chosen by the authors; none has been assumed.
