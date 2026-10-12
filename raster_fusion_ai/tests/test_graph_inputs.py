from argparse import Namespace
import csv
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS
from scipy.sparse import load_npz
from generate_graph_inputs import graph,export_graph,rasters,NODATA
from generate_graph_examples import surface


def test_graph_edges_and_strength():
    xy=np.array([[0.,0.],[3.,0.],[6.,0.],[100.,0.]])
    g=graph(xy,1,5)
    assert g[1].tolist()==[[0,1],[1,2]]
    assert g[4].tolist()==[1,2,1,0]
    np.testing.assert_allclose(g[5],[np.exp(-.6),2*np.exp(-.6),np.exp(-.6),0])
    assert g[7].tolist()==[3,3,3,1] and g[-1]==2
    assert graph(np.array([[0.,0.]]),8,50000)[4].tolist()==[0]
    assert graph(np.empty((0,2)),8,50000)[-1]==0


def test_raster_tiles_masks_and_graph_exports(tmp_path):
    xy=np.array([[500100.,6999900.],[500400.,6999900.],[510000.,6990000.]])
    g=graph(xy,1,500)
    profile=dict(driver='GTiff',width=13,height=11,count=1,dtype='float32',nodata=NODATA,
                 crs='EPSG:31983',transform=from_origin(500000,7000000,100,100))
    source=tmp_path/'source.tif'; data=np.zeros((11,13),'float32');data[3,3]=NODATA
    with rasterio.open(source,'w',**profile) as ds: ds.write(data,1)
    expected,support=surface(g[0],xy,g[5],profile['transform'],data.shape,500,1)
    expected[3,3]=NODATA;support[3,3]=NODATA
    for block in [4,20]:
        folder=tmp_path/str(block);folder.mkdir()
        rasters(source,folder,xy,g,Namespace(block=block,k=1,radius_km=.5))
        for name,a in [('conectividade',expected),('suporte',support)]:
            with rasterio.open(folder/(name+'.tif')) as ds:
                np.testing.assert_array_equal(ds.read(1),a)
                assert ds.transform==profile['transform'] and ds.shape==data.shape
    export_graph(tmp_path,xy,np.ones(3,int),np.arange(1,4),np.arange(3),CRS.from_epsg(31983),g)
    m=load_npz(tmp_path/'adjacencia.npz')
    assert (m!=m.T).nnz==0
    np.testing.assert_allclose(np.asarray(m.sum(axis=1)).ravel(),g[5])
    assert len(json.loads((tmp_path/'nos.geojson').read_text())['features'])==3
    assert len(json.loads((tmp_path/'arestas.geojson').read_text())['features'])==1


def test_empty_graph_export_and_zero_surface(tmp_path):
    xy=np.empty((0,2));g=graph(xy,8,50000)
    export_graph(tmp_path,xy,np.array([],int),np.array([],int),np.array([],int),CRS.from_epsg(31983),g)
    assert load_npz(tmp_path/'adjacencia.npz').shape==(0,0)
    assert json.loads((tmp_path/'nos.geojson').read_text())['features']==[]
    source=tmp_path/'source.tif'
    with rasterio.open(source,'w',driver='GTiff',width=4,height=3,count=1,dtype='float32',
                       crs='EPSG:31983',transform=from_origin(500000,7000000,100,100),nodata=NODATA) as ds:
        ds.write(np.zeros((3,4),'float32'),1)
    rasters(source,tmp_path,xy,g,Namespace(block=2,k=8,radius_km=50))
    for name in ['conectividade','suporte']:
        with rasterio.open(tmp_path/(name+'.tif')) as ds: assert np.all(ds.read(1)==0)
