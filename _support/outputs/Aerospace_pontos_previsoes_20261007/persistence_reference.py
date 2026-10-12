"""Historical-target reference on exactly the same native valid support."""
from pathlib import Path
import json
import numpy as np,rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from analyze_hotspots import OUT,ROOT,inventory,analyze

for region in ['internal','external']:
 for year in [2021,2022]:
  ref=next(r for r in inventory(region) if r['key']=='linear' and r['horizon']==0 and r['target_year']==year)
  dest=OUT/f'rasters/persistence/{region}_{year}.tif';dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():
   with rasterio.open(ref['prediction']) as src:
    profile=src.profile.copy();valid=np.isfinite(src.read(1))
    previous=ROOT/f'data/{"regiao_treino" if region=="internal" else "regiao_teste_final"}/{year-1}/target.tif'
    with rasterio.open(previous) as target,WarpedVRT(target,crs=src.crs,transform=src.transform,width=src.width,height=src.height,resampling=Resampling.nearest,dtype='float32',nodata=np.nan) as vrt:
     a=vrt.read(1);assert np.isfinite(a[valid]).all();a[~valid]=np.nan
    profile.update(dtype='float32',nodata=np.nan,compress='deflate',predictor=3,tiled=True,blockxsize=256,blockysize=256)
    with rasterio.open(dest,'w',**profile) as dst:dst.write(a,1);dst.update_tags(reference='Previous-year observed target, not fitted',source=str(previous))
  analyze(dict(region=region,key='persistence',model='Persistence reference',horizon=1,target_year=year,prediction=str(dest),expected_valid=ref['expected_valid']))
