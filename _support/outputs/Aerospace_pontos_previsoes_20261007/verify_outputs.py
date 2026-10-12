"""Independent round-trip checks of point values, hotspot flags and saved models."""
from pathlib import Path
import csv,json
import numpy as np,rasterio
from rasterio.transform import rowcol
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]/'raster_fusion_ai'

def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
 results=[]
 paths=sorted((OUT/'data/metrics').glob('*.json'))
 for path in paths:
  m=json.loads(path.read_text());points=read_csv(Path(m['point_csv']))
  with rasterio.open(m['prediction']) as src:
   a=src.read(1);valid=np.isfinite(a);assert valid.sum()==m['valid_cells']
   rr,cc=rowcol(src.transform,[float(p['x']) for p in points],[float(p['y']) for p in points]);rr,cc=np.array(rr),np.array(cc)
   inside=(rr>=0)&(rr<src.height)&(cc>=0)&(cc<src.width)
   eligible=np.array([p['eligible']=='1' for p in points]);assert np.count_nonzero(~inside)==m['excluded_outside']
   assert all(int(p['raster_row'])==r and int(p['raster_col'])==c for p,r,c in zip(points,rr,cc))
   got=a[rr[eligible],cc[eligible]].astype(np.float64)
   stored=np.array([float(p['prediction_intensity']) for p in points if p['eligible']=='1'])
   assert np.array_equal(got,stored)
   with rasterio.open(m['hotspot']) as hs:
    assert hs.crs==src.crs and hs.transform==src.transform and hs.shape==src.shape
    hot=hs.read(1);assert np.array_equal(hot==255,~valid)
    assert np.count_nonzero(hot==1)==m['thresholds'][1]['hot_cells']
    assert np.array_equal(hot[rr[eligible],cc[eligible]]==1,got>=m['thresholds'][1]['threshold'])
   for s in m['thresholds']:
    q=s['percentile'];flags=got>=s['threshold'];saved=np.array([p[f'hit_p{q}']=='1' for p in points if p['eligible']=='1'])
    assert np.array_equal(flags,saved)
    assert int(flags.sum())==s['cumulative']['hits']
    annual=np.array([int(p['record_year'])==m['target_year'] for p in points if p['eligible']=='1'])
    assert int((flags&annual).sum())==s['current_year']['hits'] and int(annual.sum())==s['current_year']['n']
    assert np.isclose(100*flags.mean(),s['cumulative']['hit_percent'],rtol=0,atol=1e-10)
   check=dict(stem=m['stem'],point_values_match_native_raster=True,hotspot_flags_match_point_values=True,counts_reconciled=True)
   if m['region']=='internal' and m['key']!='persistence':
    olddir=ROOT/f'reports/treinamento_target_previsoes_{m["target_year"]}'
    z=np.load(olddir/'target_e_zonas.npz');old=np.load(olddir/f'H{m["horizon"]}_{m["key"]}.npy');new=a[np.ix_(z['rows'],z['cols'])]
    assert np.array_equal(np.isfinite(new),np.isfinite(old))
    mask=np.isfinite(old);diff=np.abs(new[mask].astype(np.float64)-old[mask].astype(np.float64))
    scale=float(np.max(np.abs(old[mask])));maxdiff=float(diff.max());rmse=float(np.sqrt(np.mean(diff**2)))
    check.update(prior_preview_points=int(mask.sum()),max_abs_difference_from_prior_preview=maxdiff,rmse_difference_from_prior_preview=rmse,max_difference_relative_to_peak=maxdiff/max(scale,1e-9),comparison_note='Compact CNN/ResNet use dense convolution rather than the earlier sampled-patch batch; floating-point kernels can differ numerically.')
    assert maxdiff<=max(1e-4,scale*.002)
    sources=json.loads((olddir/'fontes.json').read_text(encoding='utf-8'))
    previous=next(s for s in sources if s['model_path']==m['model_path'] and s['horizonte']==m['horizon'])
    assert previous['sha256']==m['model_sha256'];check['model_hash_matches_previous_report']=True
   results.append(check)
   print('PASS',m['stem'],flush=True)
 (OUT/'VERIFICACAO.json').write_text(json.dumps(dict(checked_evaluations=len(results),main_evaluations=sum('persistence' not in r['stem'] for r in results),all_checks_passed=True,results=results),indent=2),encoding='utf-8')

if __name__=='__main__':main()
