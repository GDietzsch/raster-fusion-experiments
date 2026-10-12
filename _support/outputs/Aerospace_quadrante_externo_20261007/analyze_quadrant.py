"""Post-hoc spatial diagnosis using frozen predictions and their actual inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import csv,json,hashlib
import numpy as np
import rasterio
from scipy.spatial import cKDTree
import joblib

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]/'raster_fusion_ai'
PREV=OUT.parent/'Aerospace_pontos_previsoes_20261007'
CACHE=ROOT/'runs/enriched_cnn/20260926T223539611207Z/aligned_inputs'
GRAPH=ROOT/'outputs/graph_inputs/20260925T000025837870Z'
THEMES=['Garimpos','Ocorrencias_Minerais','Pistas_Regulares','Pistas_Irregulares']
NAMES=['Mining sites','Mineral occurrences','Regular airstrips','Irregular airstrips']
KINDS=['Original KDE','Gradient magnitude','Gradient sine','Gradient cosine','Graph connectivity','Graph support']

def csvread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def savecsv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def stats(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return {'n':len(a),'nonzero':int(np.count_nonzero(a)), 'min':float(a.min()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'mean':float(a.mean()) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,'max':float(a.max()) if len(a) else None}

def main():
    with rasterio.open(ROOT/'data/regiao_teste_final/2022/target.tif') as src:
        bounds=list(src.bounds);transform=src.transform;shape=src.shape
    left,bottom,right,top=bounds;xm=(left+right)/2;ym=(bottom+top)/2
    rows=np.arange(shape[0]);cols=np.arange(shape[1]);xx=left+(cols+.5)*250;yy=top-(rows+.5)*250
    qgrid=(yy[:,None]>=ym)&(xx[None,:]<xm)
    points=csvread(PREV/'data/point_intensities/external_hgb_H1_2022.csv')
    rr=np.array([int(p['raster_row']) for p in points]);cc=np.array([int(p['raster_col']) for p in points])
    xy=np.array([[float(p['x']),float(p['y'])] for p in points]);years=np.array([int(p['record_year']) for p in points])
    nw=(xy[:,0]<xm)&(xy[:,1]>=ym)
    report=dict(definition={'x_less_than_m':xm,'y_greater_equal_m':ym,'bounds':bounds,'crs':'ESRI:102015','selection':'Upper-left halves of full external raster extent, selected after viewing errors; descriptive diagnosis'},points={'NW':int(nw.sum()),'remainder':int((~nw).sum()),'NW_by_record_year':{str(y):int(np.sum(nw&(years==y))) for y in sorted(set(years))}},years={},neighborhoods={},training={},graph_parameter_evidence=[])
    point_export=[];feature_rows=[];distance_rows=[]
    for i,p in enumerate(points):point_export.append({k:p[k] for k in ['point_id','record_year','x','y','raster_row','raster_col']}|{'NW':int(nw[i])})
    for year in [2020,2021,2022]:
        array=np.load(CACHE/f'teste_final_{year}.npy',mmap_mode='r')
        valid=np.isfinite(array).all(axis=0);allzero=np.all(array==0,axis=0);v=np.asarray(array[:,rr,cc]).T
        originalzero=np.all(v[:,[0,6,12,18]]==0,axis=1)
        yreport={}
        for area,sel,grid in [('NW',nw,qgrid),('remainder',~nw,~qgrid)]:
            g=grid&valid
            yreport[area]={'records':int(sel.sum()),'all24_zero_records':int(np.sum(np.all(v[sel]==0,axis=1))),'all_original_zero_records':int(np.sum(originalzero[sel])),'valid_grid_cells':int(g.sum()),'all24_zero_grid_percent':float(100*allzero[g].mean()),'invalid_grid_cells':int((grid&~valid).sum())}
            for j,(theme,name) in enumerate(zip(THEMES,NAMES)):
                for k,kind in enumerate(KINDS):
                    s=stats(v[sel,j*6+k]);feature_rows.append(dict(Input_year=year,Area=area,Theme=name,Channel=kind,**s,positive_grid_percent=float(100*np.mean(array[j*6+k][g]>0))))
        report['years'][str(year)]=yreport
        for i in range(len(points)):
            for j,theme in enumerate(THEMES):
                for k,kind in enumerate(['original','magnitude','direction_sine','direction_cosine','connectivity','support']):point_export[i][f'{year}_{theme}_{kind}']=float(v[i,j*6+k])
        if year==2021:
            for radius,label in [(2,'local_5x5'),(164,'square_41km'),(224,'square_56km')]:
                flags=[]
                for r,c in zip(rr,cc):
                    patch=np.asarray(array[:,max(0,r-radius):min(shape[0],r+radius+1),max(0,c-radius):min(shape[1],c+radius+1)])
                    flags.append(bool(np.all(np.nan_to_num(patch)==0)))
                flags=np.array(flags)
                report['neighborhoods'][label]={'NW_all_channels_zero':int(np.sum(nw&flags)),'remainder_all_channels_zero':int(np.sum(~nw&flags)),'note':'Square support diagnostic; not asserted to be the effective receptive field of a dense network.'}
                for i in range(len(points)):point_export[i][f'2021_{label}_all24zero']=int(flags[i])
            wide=[]
            for r,c in zip(rr,cc):
                rrr=r+np.array([-164,-82,0,82,164]);ccc=c+np.array([-164,-82,0,82,164]);rrr=rrr[(rrr>=0)&(rrr<shape[0])];ccc=ccc[(ccc>=0)&(ccc<shape[1])]
                wide.append(bool(np.all(np.nan_to_num(array[:,rrr[:,None],ccc[None,:]])==0)))
            wide=np.array(wide);report['neighborhoods']['CNN_sparse_wide_5x5']={'NW_all24zero':int(np.sum(nw&wide)),'remainder_all24zero':int(np.sum(~nw&wide))}
            for i in range(len(points)):point_export[i]['2021_sparse_wide_all24zero']=int(wide[i])
        del array
        print('INPUT',year,yreport,flush=True)
    for j,(theme,name) in enumerate(zip(THEMES,NAMES)):
        nodes=csvread(GRAPH/f'regiao_teste_final/2021/{theme}/nos.csv')
        nodexy=np.array([[float(n['x_m']),float(n['y_m'])] for n in nodes]);dist,_=cKDTree(nodexy).query(xy)
        for i in range(len(points)):point_export[i][f'nearest_2021_graph_node_{theme}_km']=float(dist[i]/1000)
        for area,sel in [('NW',nw),('remainder',~nw)]:distance_rows.append(dict(Area=area,Theme=name,Node_count=len(nodes),Beyond_50km=int(np.sum(dist[sel]>50000)),**stats(dist[sel]/1000)))
        meta=json.loads((CACHE/'teste_final_2021.json').read_text())
        with rasterio.open(meta['paths'][6*j+4]) as src:
            report['graph_parameter_evidence'].append({'theme':theme,'file':meta['paths'][6*j+4],'tags':src.tags()})
    performance=[]
    for path in sorted((PREV/'data/metrics').glob('external_*.json')):
        m=json.loads(path.read_text());p=csvread(m['point_csv']);pxy=np.array([[float(x['x']),float(x['y'])] for x in p]);inside=(pxy[:,0]<xm)&(pxy[:,1]>=ym)
        for area,sel in [('NW',inside),('remainder',~inside)]:
            n=int(sel.sum());selpoints=[x for x,z in zip(p,sel) if z];s=m['thresholds'][1];hits=sum(x['hit_p90']=='1' for x in selpoints)
            record=dict(Model=m['model'],Key=m['key'],Horizon=m['horizon'],Input_year=m['input_year'],Target_year=m['target_year'],Area=area,Records=n,Hits_P90=hits,Coverage_P90=100*hits/n if n else None,P90=s['threshold'],Prediction_median=float(np.median([float(x['prediction_intensity']) for x in selpoints])) if n else None,Record_years=';'.join(map(str,sorted({int(x['record_year']) for x in selpoints}))))
            for q in [80,95]:record[f'Hits_P{q}']=sum(x[f'hit_p{q}']=='1' for x in selpoints)
            performance.append(record)
        if m['horizon']==1 and m['target_year']==2022:
            with rasterio.open(m['prediction']) as s:pred=s.read(1,masked=True).filled(np.nan)
            valid=np.isfinite(pred)
            for rec in performance[-2:]:
                grid=qgrid if rec['Area']=='NW' else ~qgrid;g=grid&valid
                rec['Hot_area_within_subregion_percent']=float(100*np.mean(pred[g]>=m['thresholds'][1]['threshold']))
            for i,row in enumerate(points):
                byid={r['point_id']:r for r in p};r=byid[row['point_id']]
                point_export[i][f'{m["key"]}_H1_2022_prediction']=float(r['prediction_intensity']);point_export[i][f'{m["key"]}_H1_2022_hitP90']=int(r['hit_p90'])
    # Actual HGB fitting samples: initial stage and channel-selection stage, used in refit.
    run=ROOT/'runs/enriched_hgb/20260925T213151313641Z/horizon_1'
    with np.load(run/'samples.npz') as z:
        x=np.concatenate([z['ajuste_x'],z['selecao_camadas_x']]);y=np.concatenate([z['ajuste_y'],z['selecao_camadas_y']])
    zero=np.all(x==0,axis=1);report['training']['HGB_H1']={'refit_records':len(y),'all24zero_records':int(zero.sum()),'all24zero_target':stats(y[zero]),'all24zero_target_positive':int(np.sum(y[zero]>0)),'other_target':stats(y[~zero])}
    artifact=joblib.load(run/'todas/model.joblib');score=float(artifact['model'].predict(np.zeros((1,24),dtype=np.float32))[0]);report['training']['HGB_H1']['zero_vector_prediction']=score
    evalzero=np.array([all(p[f'2021_{t}_{k}']==0 for t in THEMES for k in ['original','magnitude','direction_sine','direction_cosine','connectivity','support']) for p in point_export])
    actual=np.array([p['hgb_H1_2022_prediction'] for p in point_export]);assert np.allclose(actual[evalzero],score,atol=1e-6)
    report['training']['HGB_H1']['zero_prediction_matches_all_external_zero_points']=True
    # Source rasters sampled at detection coordinates: distinguish valid zero from NoData.
    sourceaudit=[];meta=json.loads((CACHE/'teste_final_2021.json').read_text())
    for j in [0,6,12,18]:
        with rasterio.open(meta['paths'][j]) as src:
            vals=np.ma.vstack(list(src.sample(xy,masked=True)))[:,0]
            sourceaudit.append({'theme':THEMES[j//6],'nodata':str(src.nodata),'NW_masked_source_points':int(np.ma.getmaskarray(vals)[nw].sum()),'NW_zero_source_points':int(np.sum(np.asarray(vals)[nw]==0))})
    report['source_original_values']=sourceaudit
    savecsv('POINT_INPUTS_AND_PREDICTIONS.csv',point_export);savecsv('CHANNEL_SUMMARY.csv',feature_rows);savecsv('GRAPH_NODE_DISTANCES.csv',distance_rows)
    allkeys=list(dict.fromkeys(k for r in performance for k in r));performance=[{k:r.get(k) for k in allkeys} for r in performance];savecsv('QUADRANT_MODEL_COVERAGE.csv',performance)
    (OUT/'DIAGNOSTIC.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('DIAGNOSIS',json.dumps(report['points']),json.dumps(report['neighborhoods']),json.dumps(report['training']),flush=True)

if __name__=='__main__':main()
