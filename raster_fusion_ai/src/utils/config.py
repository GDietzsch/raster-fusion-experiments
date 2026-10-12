"""Configuration and manifest contracts shared by every service."""
from pathlib import Path
import json
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("data", "split", "training", "output"):
        if section not in cfg:
            raise ValueError(f"Missing configuration section: {section}")
    cfg["_base"] = str(path.parent)
    fractions = [cfg["split"][k] for k in ("train", "validation", "test")]
    if any(v <= 0 for v in fractions) or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to 1")
    if cfg["training"].get("device", "auto") not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if cfg["split"].get("buffer_pixels", 0) < 0:
        raise ValueError("buffer_pixels must be nonnegative")
    training = cfg["training"]
    if training.get("epochs", 1) < 1 or training.get("batch_size", 1) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if training.get("loss", "mse") not in ("mse", "mae", "huber"):
        raise ValueError("loss must be mse, mae or huber")
    image = cfg.get("image_pipeline", {})
    if image.get("enabled", False):
        if not image.get("patch_sizes") or not 0 < image.get("stride_ratio", .5) <= 1:
            raise ValueError("Require patch_sizes and 0<stride_ratio<=1")
        for size in image["patch_sizes"]:
            if size not in (16, 32, 64, 128):
                raise ValueError("Supported patch sizes: 16, 32, 64, 128")
            if size // 2**image.get("depth", 2) < 2:
                raise ValueError("Patch is too small for U-Net depth; require at least 2 bottleneck pixels per axis")
    return cfg


def resolve(cfg: dict, path: str | Path) -> Path:
    return (Path(cfg["_base"]) / path).resolve()


def load_layers(path: Path) -> list[dict]:
    layers = yaml.safe_load(path.read_text(encoding="utf-8"))["layers"]
    if not layers or len({x["name"] for x in layers}) != len(layers):
        raise ValueError("Layer names must be nonempty and unique")
    if len({x["filename"] for x in layers}) != len(layers):
        raise ValueError("Layer filenames must be unique")
    for layer in layers:
        filename = Path(layer["filename"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError("Layer filenames must be relative paths inside input_dir")
        if not layer["name"] or any(c in layer["name"] for c in '/\\:'):
            raise ValueError("Layer names must be nonempty and cannot contain path separators")
        if layer["name"] in {"row", "col", "x_coordinate", "y_coordinate", "split", "target"}:
            raise ValueError(f"Reserved layer name: {layer['name']}")
        if layer["type"] not in ("continuous", "binary", "categorical"):
            raise ValueError(f"Unknown layer type: {layer}")
        if layer["resampling"] not in ("nearest", "bilinear", "cubic"):
            raise ValueError("Resampling must be explicit: nearest, bilinear, cubic")
        if layer["type"] != "continuous" and layer["resampling"] != "nearest":
            raise ValueError("Binary/categorical layers require nearest resampling")
        if layer.get("normalization", "zscore") not in ("none", "zscore", "minmax", "robust"):
            raise ValueError("Unknown normalization")
    return layers


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
