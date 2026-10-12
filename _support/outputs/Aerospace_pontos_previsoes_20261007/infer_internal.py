"""Native-grid inference from the 24 frozen winners; never train or select."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import argparse, gc, hashlib, json, sys, time
import numpy as np
import rasterio
from rasterio.windows import Window
import joblib
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]/'raster_fusion_ai'
sys.path.insert(0,str(ROOT))
from src.services import enriched_cnn as cnn, enriched_unet as unet, enriched_linear as common

SPECS=[('Linear','linear','20260925T141835755712Z'),('Ridge','ridge','20260925T191737660876Z'),
 ('HGB','hgb','20260925T213151313641Z'),('XGBoost','xgboost','20260926T150535837633Z'),
 ('Random Forest','random_forest','20260926T171032572709Z'),('Extra Trees','extra_trees','20260926T191430559698Z'),
 ('MLP','mlp','20260926T204853996228Z'),('CNN','cnn','20260926T223539611207Z'),
 ('Compact ResNet','resnet','20260927T140640055671Z'),('U-Net','unet','20260927T185613221209Z'),
 ('ResUNet','resunet','20260928T112749380518Z'),('Dense Inception','inception','20260928T200135452961Z')]
CACHE=ROOT/'runs/enriched_cnn/20260926T223539611207Z/aligned_inputs'
DATA=ROOT/'data/conjuntos_enriquecidos/20260925T122022935585Z'
PAIRS=[(0,2020),(0,2021),(0,2022),(1,2021),(1,2022)]

class FrozenCache(unet.InputCache):
 def __init__(self,array,profile):self.array=array;self.profile=profile
 def year(self,year):return self.array

def run(group):
 torch.set_num_threads(4)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 with rasterio.open(DATA/'treinamento/2020/targets/target.tif') as src:profile=src.profile.copy()
 H,W=profile['height'],profile['width']
 folder=OUT/'rasters/internal';folder.mkdir(parents=True,exist_ok=True)
 output_profile=profile.copy();output_profile.update(driver='GTiff',count=1,dtype='float32',nodata=np.nan,tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=3,BIGTIFF='IF_SAFER')
 neural={'cnn','resnet','unet','resunet','inception'}
 for name,key,stamp in SPECS:
  if (key in neural)!=(group=='neural'):continue
  for horizon,year in PAIRS:
   dest=folder/f'{key}_H{horizon}_{year}.tif';meta=dest.with_suffix('.json')
   if dest.exists() and meta.exists():print(f'SKIP {dest.name}',flush=True);continue
   winner=json.loads((ROOT/f'runs/enriched_{key}/{stamp}/horizon_{horizon}/winner.json').read_text(encoding='utf-8'))
   model_path=Path(winner['model_path']);start=time.perf_counter()
   print(f'START {dest.name} device={device if key in neural else "cpu"}',flush=True)
   array=np.load(CACHE/f'treinamento_{year-horizon}.npy',mmap_mode='r')
   assert array.shape==(24,H,W)
   if key in neural:
    artifact=torch.load(model_path,map_location='cpu',weights_only=True)
    model,columns,scale=(cnn if key in {'cnn','resnet'} else unet)._restore(artifact,device)
   else:
    artifact=joblib.load(model_path);model=artifact['model'];columns=tuple(artifact['columns'])
    if hasattr(model,'n_jobs'):model.n_jobs=4
   assert list(artifact['feature_names'])==list(winner['feature_names'])
   tmp=dest.with_suffix('.partial.tif')
   valid_count=0
   with rasterio.open(tmp,'w',**output_profile) as target:
    if key not in neural:
     for r in range(0,H,64):
      height=min(64,H-r);raw=array[:,r:r+height,:]
      valid=np.isfinite(raw).all(axis=0);pred=np.full((height,W),np.nan,dtype=np.float32)
      x=raw.reshape(24,-1).T
      if valid.any():pred[valid]=common._predict(model,x[valid.ravel()],columns)
      valid_count+=int(valid.sum());target.write(pred,1,window=Window(0,r,W,height))
      if r%1024==0:print(f'  {dest.name} rows={r+height}/{H} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    else:
     tile_size=512;cache=FrozenCache(array,profile);i=0
     for r in range(0,H,tile_size):
      for c in range(0,W,tile_size):
       height,width=min(tile_size,H-r),min(tile_size,W-c)
       valid=np.isfinite(array[:,r:r+height,c:c+width]).all(axis=0)
       pred=np.full((height,width),np.nan,dtype=np.float32)
       if valid.any():
        if key in {'cnn','resnet'}:
         radius=cnn.WIDE_RADIUS if artifact.get('wide_context',True) else cnn.RADIUS
         r0,c0=max(0,r-radius),max(0,c-radius);r1,c1=min(H,r+height+radius),min(W,c+width+radius)
         raw=np.full((len(columns),height+2*radius,width+2*radius),np.nan,dtype=np.float32)
         raw[:,r0-r+radius:r1-r+radius,c0-c+radius:c1-c+radius]=array[list(columns),r0:r1,c0:c1]
         x=(raw-scale['input_mean'][:,None,None])/scale['input_scale'][:,None,None]
         np.nan_to_num(x,copy=False,nan=0,posinf=0,neginf=0)
         with torch.inference_mode():
          p=model.forward_dense(torch.from_numpy(np.ascontiguousarray(x[None])).to(device),height,width).squeeze(0).cpu().numpy()
         p=p*scale['target_scale']+scale['target_mean']
        else:p=unet._predict_tile(model,cache,(year-horizon,r//512,c//512),columns,scale,device)[:height,:width]
        assert np.isfinite(p[valid]).all();pred[valid]=p[valid]
       valid_count+=int(valid.sum());target.write(pred,1,window=Window(c,r,width,height));i+=1
       if i%40==0:print(f'  {dest.name} tiles={i}/220 elapsed={time.perf_counter()-start:.1f}s',flush=True)
    target.update_tags(model=name,horizon=horizon,input_year=year-horizon,target_year=year,model_source=str(model_path),evaluation='Descriptive internal region; includes fitting locations',inference='Frozen model, native 250 m grid, no retraining')
   tmp.replace(dest)
   meta.write_text(json.dumps(dict(model=name,key=key,horizon=horizon,input_year=year-horizon,target_year=year,model_path=str(model_path),model_sha256=common._sha256(model_path),features=winner['feature_names'],prediction=str(dest),valid_cells=valid_count,seconds=time.perf_counter()-start,new_training=False),indent=2),encoding='utf-8')
   print(f'DONE {dest.name} {time.perf_counter()-start:.1f}s valid={valid_count}',flush=True)
   del model,artifact,array;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('group',choices=['tabular','neural']);run(parser.parse_args().group)
