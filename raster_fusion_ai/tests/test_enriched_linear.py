"""Small end-to-end audit of the enriched linear experiment protocol."""
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml

from src.services.enriched_linear import PHENOMENA, PRODUCTS, check_only, metrics, run, select_phenomena


def _write_tif(path: Path, values: np.ndarray):
    path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,'w',driver='GTiff',height=values.shape[0],width=values.shape[1],
                       count=1,dtype='float32',crs='EPSG:3857',transform=from_origin(0,3000,250,250),
                       nodata=-9999.0) as ds:
        ds.write(values.astype(np.float32),1)


def _synthetic_dataset(tmp_path: Path) -> Path:
    root = tmp_path/'data'
    yy,xx = np.mgrid[:12,:12]
    recorded = {}
    for region in ('treinamento','teste_final'):
        for year in (2019,2020,2021,2022):
            for p,phenomenon in enumerate(PHENOMENA):
                for k,product in enumerate(PRODUCTS):
                    values = (1+p)*xx + (1+k)*yy + (year-2019)*0.1 + (region=='teste_final')*0.5
                    if product == 'suporte':
                        values = np.ones_like(xx,dtype=float)
                    folder = 'auxiliares' if product == 'suporte' else 'inputs'
                    path = root/region/str(year)/folder/phenomenon/f'{product}.tif'
                    _write_tif(path,values)
                    rel = path.relative_to(root).as_posix()
                    recorded[rel] = dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    evaluation_targets = {}
    for region in ('treinamento','teste_final'):
        for year in (2020,2021,2022):
            values = (xx+2*yy+(year-2020)*0.1+(region=='teste_final')*0.2).astype(float)
            values[:2,:2] = 0
            if region == 'treinamento':
                path = root/region/str(year)/'targets'/'target.tif'
                _write_tif(path,values)
                recorded[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size,
                                                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            else:
                path = root/'external_targets'/str(year)/'target.tif'
                _write_tif(path,values)
                evaluation_targets[f'{region}/{year}'] = str(path)
    (root/'manifest.json').write_text(json.dumps(dict(status='complete',config=dict(evaluation_targets=evaluation_targets),
                                                   files=recorded)),encoding='utf-8')
    config = dict(data_root='data',output_root='runs',reference_target='treinamento/2020/targets/target.tif',
                  split=dict(method='spatial_block',train=0.70,validation=0.15,test=0.15,
                             block_size=4,buffer_pixels=0,seed=42),
                  sampling=dict(per_stage_year=20,seed=4201,window_size=5),
                  evaluation=dict(target_resampling='nearest',high_target_positive_quantile=0.99,
                                  primary_metric='rmse',tie_breakers=['mae','fewer_features'],
                                  final_h0_target_years=[2020,2021,2022],final_h1_target_years=[2022]),
                  runtime=dict(cpu_threads=1))
    config_path = tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(config),encoding='utf-8')
    return config_path


def test_metrics_distinguish_zero_and_no_bias():
    y = np.array([0.,0.,1.,2.])
    p = np.array([0.,1.,1.,3.])
    result = metrics(y,p,high_threshold=2.)
    assert result['target_zero'] == 2
    assert result['rmse'] > 0
    assert result['subsets']['zero']['n'] == 2
    assert result['p90_absolute_error'] == 1


def test_group_selection_never_uses_extra_columns():
    x = np.zeros((30,24),dtype=float)
    x[:,0] = np.arange(30)
    x[:,1:] += np.random.default_rng(4).normal(size=(30,23))
    y = 3*x[:,0]
    chosen,records = select_phenomena('originais_filtradas',x[:20],y[:20],x[20:],y[20:])
    assert 'Garimpos' in chosen
    assert len(records) == 15
    assert all(all(feature.endswith('__original') for feature in record['features']) for record in records)


def test_full_synthetic_experiment(tmp_path):
    config = _synthetic_dataset(tmp_path)
    check_only(config)
    # A complete tiny run verifies training, feature selection, external metrics,
    # georeferenced predictions, provenance and report assembly together.
    run(config)
    output = next((tmp_path/'runs').iterdir())
    status = json.loads((output/'run.json').read_text(encoding='utf-8'))
    assert status['status'] == 'completed'
    assert status['final_external_test_used']
    assert (output/'RELATORIO.md').exists()
    assert (output/'horizon_0'/'external_prediction_target_2022.tif').exists()
    assert (output/'horizon_1'/'external_prediction_target_2022.tif').exists()
    for horizon in (0,1):
        winner = json.loads((output/f'horizon_{horizon}'/'winner.json').read_text(encoding='utf-8'))
        assert winner['arm'] in ('originais','originais_filtradas','gpa','grafos','todas','todas_filtradas')
