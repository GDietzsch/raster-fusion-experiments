"""Resolve the shared runtime and canonical algorithm sources."""
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
SOURCES = {'src/services/enriched_linear.py': '01/enriched_linear.py', 'src/services/enriched_ridge.py': '02/enriched_ridge.py', 'src/services/enriched_hgb.py': '03/enriched_hgb.py', 'src/services/enriched_xgboost.py': '04/enriched_xgboost.py', 'src/services/enriched_random_forest.py': '05/enriched_random_forest.py', 'src/services/enriched_extra_trees.py': '06/enriched_extra_trees.py', 'src/services/enriched_mlp.py': '07/enriched_mlp.py', 'src/services/enriched_cnn.py': '08/enriched_cnn.py', 'src/services/enriched_cnn_local_control.py': '08A/enriched_cnn_local_control.py', 'src/services/enriched_unet.py': '10/enriched_unet.py', 'src/services/enriched_unet_controls.py': '10A/enriched_unet_controls.py', 'src/services/enriched_resunet.py': '11/enriched_resunet.py', 'src/services/enriched_inception.py': '12/enriched_inception.py', 'src/services/enriched_resnet_model.py': '09/enriched_resnet_model.py'}

def project_root():
    return ROOT

def source_path(relative):
    key = str(relative).replace('\\', '/')
    return ROOT / SOURCES[key] if key in SOURCES else PROJECT / key
