# 08A — Local-only CNN

[Algorithm](../08A/enriched_cnn_local_control.py) · [Configuration](config.yaml) · [Installation and data format](../README.md)

Edit the input/output paths in `config.yaml` to use your own data. Paths are relative to this folder.

Set the reference/sample paths to the corresponding trained main model.

From the repository root:

```sh
python 08A/run.py --check-only
python 08A/run.py
```

Use `python 08A/run.py --help` for the available options.
