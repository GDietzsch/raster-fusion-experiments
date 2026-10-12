"""Audit point provenance, dates, CRS and compatibility with the evaluated targets."""
from pathlib import Path
import csv, hashlib, json, struct, sys
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from scipy.spatial.distance import cdist
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1];ROOT=BASE/'raster_fusion_ai'
sys.path.insert(0,str(ROOT))
from generate_graph_examples import read_points

def dbf(path):
 b=path.read_bytes();count,hl,rl=struct.unpack_from('<IHH',b,4)
 fields=[];offset=1
 for pos in range(32,hl-1,32):
  name=b[pos:pos+11].split(b'\0')[0].decode('ascii');length=b[pos+16]
  fields.append((name,offset,length));offset+=length
 assert offset==rl
 return [{name:b[hl+i*rl+off:hl+i*rl+off+length].decode('latin1').strip() for name,off,length in fields} for i in range(count)]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 (OUT/'data').mkdir(exist_ok=True);audits=[]
 for region in ['internal','external']:
  cumulative=[]
  for year in [2020,2021,2022]:
   source=(BASE/f'Projeto/Treinamento/Targets/Desconhecido_{year}.shp' if region=='internal' else BASE/f'Temporal/Test/Desconhecido_ate_{year}.shp')
   xy,ids,count=read_points(source);records=dbf(source.with_suffix('.dbf'))
   assert len(xy)==len(ids)==count and len(set(ids))==count
   target=ROOT/f'data/{"regiao_treino" if region=="internal" else "regiao_teste_final"}/{year}/target.tif'
   with rasterio.open(target) as src:
    transformer=Transformer.from_crs(CRS.from_wkt(source.with_suffix('.prj').read_text()),src.crs,always_xy=True)
    x,y=transformer.transform(xy[:,0],xy[:,1])
    points=[dict(point_id=f'{region}_{year}_{fid}' if region=='internal' else f'external_{records[fid-1]["FID_ORIG"]}',record_year=year if region=='internal' else int(records[fid-1]['ANO_REF']),x=float(xx),y=float(yy),source_file=str(source),source_record=int(fid)) for xx,yy,fid in zip(x,y,ids)]
    if region=='internal':cumulative+=points
    else:cumulative=points
    # Repeated coordinates remain separate detection records, never unique flights.
    write_csv(OUT/f'data/points_{region}_{year}.csv',cumulative)
    pts=np.array([[p['x'],p['y']] for p in cumulative]);rr,cc=rasterio.transform.rowcol(src.transform,pts[:,0],pts[:,1]);rr,cc=np.array(rr),np.array(cc)
    inside=(rr>=0)&(rr<src.height)&(cc>=0)&(cc<src.width)
    chosen=np.random.default_rng(20261007).choice(np.flatnonzero(inside),min(1000,int(inside.sum())),replace=False)
    xs,ys=rasterio.transform.xy(src.transform,rr[chosen],cc[chosen]);coords=np.column_stack([xs,ys])
    observed=np.array([v[0] for v in src.sample(coords)])
    dist=cdist(coords,pts);reconstructed=np.maximum(1-(dist/41000.)**2,0)**2;reconstructed=reconstructed.sum(axis=1)
    correlation=float(np.corrcoef(observed,reconstructed)[0,1]);rmse=float(np.sqrt(np.mean((observed-reconstructed)**2)))
    assert correlation>.999
    audits.append(dict(region=region,target_year=year,source=str(source),source_sha256=digest(source),target=str(target),target_sha256=digest(target),source_count=count,cumulative_points=len(cumulative),current_year_points=sum(p['record_year']==year for p in cumulative),inside_target_extent=int(inside.sum()),unique_coordinates=len(np.unique(pts,axis=0)),crs=src.crs.to_wkt(),kde_check_sample=len(chosen),kde_check_radius_m=41000,kde_check_rmse=rmse,kde_check_correlation=correlation,kde_check_interpretation='Numerical compatibility with cumulative unweighted quartic KDE; not a bitwise reproduction of QGIS rasterization'))
    print(region,year,'cumulative',len(cumulative),'annual',audits[-1]['current_year_points'],'KDE correlation',correlation,flush=True)
 (OUT/'data/point_source_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')

if __name__=='__main__':main()
