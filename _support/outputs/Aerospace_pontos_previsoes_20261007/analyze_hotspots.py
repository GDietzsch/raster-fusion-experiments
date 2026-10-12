"""Point-level intensity sampling and prespecified P90 hotspot coverage."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import rowcol,xy
from audit_points import write_csv,digest

OUT=Path(__file__).resolve().parent;BASE=OUT.parents[1];ROOT=BASE/'raster_fusion_ai'
KEYS={'Linear':'linear','Ridge':'ridge','HGB':'hgb','XGBoost':'xgboost','Random Forest':'random_forest','Extra Trees':'extra_trees','MLP':'mlp','CNN':'cnn','Compact ResNet':'resnet','ResNet':'resnet','U-Net':'unet','ResUNet':'resunet','Dense Inception':'inception','Inception':'inception'}
NAMES={v:k for k,v in KEYS.items()};NAMES.update(resnet='Compact ResNet',inception='Dense Inception')
def read_csv(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def inventory(region):
 records=[]
 if region=='external':
  for r in read_csv(BASE/'outputs/mdpi_acrescimos_20261004/supplement/S2_all_60_external_evaluations.csv'):
   records.append(dict(region=region,key=KEYS[r['Model']],model=NAMES[KEYS[r['Model']]],horizon=int(r['Horizon']),target_year=int(r['Evaluation pair'][-4:]),prediction=r['Prediction GeoTIFF'],expected_valid=int(r['Common pixels'])))
 else:
  for p in sorted((OUT/'rasters/internal').glob('*.json')):
   r=json.loads(p.read_text());r.update(region=region,expected_valid=r['valid_cells']);records.append(r)
 return records

def analyze(record):
 region,key,h,year=record['region'],record['key'],record['horizon'],record['target_year']
 stem=f'{region}_{key}_H{h}_{year}';dest=OUT/f'data/metrics/{stem}.json'
 if dest.exists():return json.loads(dest.read_text())
 start=time.perf_counter();prediction=Path(record['prediction'])
 points=read_csv(OUT/f'data/points_{region}_{year}.csv')
 target=ROOT/f'data/{"regiao_treino" if region=="internal" else "regiao_teste_final"}/{year}/target.tif'
 with rasterio.open(prediction) as src:
  profile=src.profile.copy();a=src.read(1);valid=np.isfinite(a)
  if src.nodata is not None and np.isfinite(src.nodata):valid &= a!=src.nodata
  assert int(valid.sum())==record['expected_valid']
  values=a[valid];minimum,maximum=float(values.min()),float(values.max())
  thresholds=np.quantile(values,[.8,.9,.95],method='linear',overwrite_input=True);del values
  x=np.array([float(p['x']) for p in points]);y=np.array([float(p['y']) for p in points])
  rr,cc=rowcol(src.transform,x,y);rr,cc=np.array(rr),np.array(cc)
  inside=(rr>=0)&(rr<src.height)&(cc>=0)&(cc<src.width)
  pv=np.full(len(points),np.nan);pv[inside]=a[rr[inside],cc[inside]]
  eligible=inside&np.isfinite(pv);annual=np.array([int(p['record_year'])==year for p in points])
  zone=np.zeros(len(points),dtype=int)
  if region=='internal':
   with rasterio.open(ROOT/'runs/enriched_cnn/20260926T223539611207Z/spatial_zones.tif') as zs:
    assert zs.shape==src.shape and zs.transform==src.transform
    zone[inside]=np.array([s[0] for s in zs.sample(np.column_stack([x[inside],y[inside]]))])
  observed=np.full(len(points),np.nan)
  with rasterio.open(target) as ts, WarpedVRT(ts,crs=src.crs,transform=src.transform,width=src.width,height=src.height,resampling=Resampling.nearest,dtype='float32',nodata=np.nan) as vrt:
   assert ts.crs==src.crs
   px,py=xy(src.transform,rr[inside],cc[inside])
   observed[inside]=np.array([s[0] for s in vrt.sample(np.column_stack([px,py]))])
  stats=[]
  for percentile,threshold in zip([80,90,95],thresholds):
   nhot=int(np.count_nonzero(valid&(a>=threshold)));point_hot=eligible&(pv>=threshold)
   def summarize(mask):
    n=int(np.count_nonzero(eligible&mask));hits=int(np.count_nonzero(point_hot&mask))
    return dict(n=n,hits=hits,hit_percent=100*hits/n if n else None)
   s=dict(percentile=percentile,threshold=float(threshold),hot_cells=nhot,hot_area_percent=100*nhot/int(valid.sum()),cells_equal_threshold=int(np.count_nonzero(valid&(a==threshold))),cumulative=summarize(np.ones(len(points),bool)),current_year=summarize(annual))
   s['lift_vs_area']=s['cumulative']['hit_percent']/s['hot_area_percent'] if s['hot_area_percent'] else None
   if region=='internal':s['zone3_descriptive']=summarize(zone==3);s['zone3_current_year']=summarize((zone==3)&annual)
   stats.append(s)
  point_rows=[]
  for i,p in enumerate(points):
   row=dict(region=region,model=record['model'],horizon=h,input_year=year-h,target_year=year,**p,raster_row=int(rr[i]),raster_col=int(cc[i]),eligible=int(eligible[i]),exclusion='' if eligible[i] else ('outside_extent' if not inside[i] else 'invalid_prediction'),prediction_intensity=float(pv[i]) if eligible[i] else '',observed_target_intensity=float(observed[i]) if np.isfinite(observed[i]) else '',partition_zone=int(zone[i]) if region=='internal' else '',current_year_record=int(annual[i]))
   for percentile,threshold in zip([80,90,95],thresholds):row[f'threshold_p{percentile}']=float(threshold);row[f'hit_p{percentile}']=int(pv[i]>=threshold) if eligible[i] else ''
   point_rows.append(row)
  write_csv(OUT/f'data/point_intensities/{stem}.csv',point_rows)
  hot=np.full(a.shape,255,dtype=np.uint8);hot[valid]=(a[valid]>=thresholds[1]).astype(np.uint8)
  hot_path=OUT/f'rasters/hotspots/{stem}_P90.tif';hot_path.parent.mkdir(parents=True,exist_ok=True)
  profile.update(dtype='uint8',nodata=255,compress='deflate',predictor=1,tiled=True,blockxsize=256,blockysize=256)
  with rasterio.open(hot_path,'w',**profile) as dst:
   dst.write(hot,1);dst.update_tags(threshold=float(thresholds[1]),percentile=90,rule='prediction >= P90; all threshold ties included',values='0=below threshold, 1=hotspot, 255=no valid prediction')
  metrics=dict(record,stem=stem,input_year=year-h,source_points=len(points),eligible_points=int(eligible.sum()),excluded_outside=int((~inside).sum()),excluded_invalid=int((inside&~eligible).sum()),source_current_year_points=int(annual.sum()),valid_cells=int(valid.sum()),prediction_min=minimum,prediction_max=maximum,negative_cells=int(np.count_nonzero(valid&(a<0))),point_unit='Detection record, not unique flight',quantile_method='numpy.quantile linear; include all ties',thresholds=stats,hotspot=str(hot_path),point_csv=str(OUT/f'data/point_intensities/{stem}.csv'),prediction_sha256=digest(prediction),seconds=time.perf_counter()-start)
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
 print(f'{stem}: P90={thresholds[1]:.5g}, hits={stats[1]["cumulative"]["hits"]}/{int(eligible.sum())} ({stats[1]["cumulative"]["hit_percent"]:.2f}%), area={stats[1]["hot_area_percent"]:.2f}%, {time.perf_counter()-start:.1f}s',flush=True)
 return metrics

def main(region):
 for r in inventory(region):analyze(r)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('region',choices=['internal','external']);main(p.parse_args().region)
