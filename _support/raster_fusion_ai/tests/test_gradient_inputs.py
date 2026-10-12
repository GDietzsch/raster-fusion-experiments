import numpy as np
import rasterio
from rasterio.transform import from_origin
from generate_gradient_inputs import gradients, process, PRODUCTS, NODATA


def test_gradient_conventions():
    y,x=np.mgrid[:7,:7]
    g=gradients(3*x-4*y,1,-1)
    np.testing.assert_allclose(g['magnitude'],5)
    np.testing.assert_allclose(g['direcao_seno'],.8)
    np.testing.assert_allclose(g['direcao_cosseno'],.6)
    np.testing.assert_allclose(g['direcao_graus'],np.degrees(np.arctan2(4,3)))
    flat=gradients(np.zeros((7,7)),250,-250)
    assert np.all(flat['magnitude']==0)
    assert np.all(flat['direcao_graus']==NODATA)
    assert np.all(flat['direcao_seno']==0) and np.all(flat['direcao_cosseno']==0)
    a=np.ones((7,7)); a[3,3]=np.nan
    g=gradients(a,250,-250)
    assert np.count_nonzero(g['magnitude']==NODATA)==5


def test_blocks_masks_and_grid(tmp_path):
    a=np.random.default_rng(7).normal(size=(25,31)).astype('float32')
    a[9,12]=np.nan
    source=tmp_path/'source.tif'
    transform=from_origin(500000,7000000,250,250)
    with rasterio.open(source,'w',driver='GTiff',height=25,width=31,count=1,
                       dtype='float32',crs='EPSG:31983',transform=transform,nodata=-9999) as ds:
        ds.write(np.where(np.isfinite(a),a,-9999),1)
    expected=gradients(np.pad(a.astype(float),1,constant_values=np.nan),250,-250)
    for block in (8,32):
        paths={k:tmp_path/f'{k}_{block}.tif' for k in PRODUCTS}
        process(source,paths,block)
        for k,p in paths.items():
            with rasterio.open(p) as ds:
                np.testing.assert_array_equal(ds.read(1),expected[k])
                assert ds.transform==transform and ds.shape==a.shape
