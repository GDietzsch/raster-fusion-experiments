# 10B — U-Net: network depth

[Algorithm](../10A/enriched_unet_controls.py) · [Configuration](config.yaml) · [Installation and data format](../README.md)

Edit the input/output paths in `config.yaml` to use your own data. Paths are relative to this folder.

Requires a trained 10 reference (`source_run`). 10B and 10C reuse the 10A controls.

From the repository root:

```sh
python 10B/run.py --check-only
python 10B/run.py
```

Use `python 10B/run.py --help` for the available options.
