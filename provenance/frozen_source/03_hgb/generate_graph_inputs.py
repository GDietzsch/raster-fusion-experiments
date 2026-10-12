"""Grafos independentes e rasters de conectividade para 32 inputs, sem Target."""
import argparse
from contextlib import ExitStack
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import shutil
import struct
import time

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, save_npz
import scipy
import matplotlib.pyplot as plt

from generate_graph_examples import read_points, make_graph, surface, draw
from generate_gradient_inputs import digest, save, ROOT, NODATA

LAYERS=('Garimpos','Ocorrencias_Minerais','Pistas_Irregulares','Pistas_Regulares')
REGIONS=('regiao_treino','regiao_teste_final')
SCRIPTS=('generate_graph_inputs.py','generate_graph_examples.py','generate_gradient_inputs.py')


def temporal_audit(shp,layer,year,record_ids):
    """Fail closed: todos os registros com pontos devem ter ano <= corte."""
    b=shp.with_suffix('.dbf').read_bytes()
    count,hl,rl=struct.unpack_from('<IHH',b,4)
    fields={}; offset=1
    for pos in range(32,hl-1,32):
        name=b[pos:pos+11].split(b'\0')[0].decode('ascii')
        length=b[pos+16]; fields[name]=(offset,length); offset+=length
    if offset!=rl or len(b)<hl+count*rl: raise ValueError(f'DBF inválido: {shp}')
    def get(i,name):
        off,length=fields[name]
        return b[hl+i*rl+off:hl+i*rl+off+length].decode('latin1').strip()
    native={'Garimpos':'data_img','Ocorrencias_Minerais':'data_cadas','Pistas_Regulares':'efetivacao'}
    field='DATA_TXT' if 'DATA_TXT' in fields else native.get(layer,'imagem')
    if field not in fields: raise ValueError(f'Campo temporal ausente: {shp}: {field}')
    years=[]; invalid=[]; future=[]
    for rec in np.unique(record_ids):
        i=int(rec)-1; raw=get(i,field)
        try:
            if field=='imagem':
                match=re.search(r'_(\d{6})(?:\.?(?:jpe?g|png))?$',raw,re.I)
                if not match: raise ValueError('sem sufixo DDMMYY')
                s=match[1]; d=date(2000+int(s[4:]),int(s[2:4]),int(s[:2]))
            elif len(raw)==8 and raw.isdigit():
                d=datetime.strptime(raw,'%Y%m%d').date()
            elif len(raw)==7:
                d=date.fromisoformat(raw+'-01')
            else: d=date.fromisoformat(raw[:10].replace('/','-'))
            if d.year>year: future.append(int(rec))
            if 'ANO_REF' in fields and int(get(i,'ANO_REF'))!=d.year: raise ValueError('Ano inconsistente')
            years.append(d.year)
        except (ValueError,TypeError): invalid.append(int(rec))
    if invalid or future:
        raise ValueError(f'Auditoria temporal falhou em {shp.name}: sem data válida={invalid[:10]}, futuros={future[:10]}')
    return dict(field=field,records=len(years),min_year=min(years) if years else None,
                max_year=max(years) if years else None,future=0,invalid=0,
                date_limitation='Pistas irregulares: ano inferido do sufixo da imagem, não confirmação de operação' if layer=='Pistas_Irregulares' else 'Data do cadastro/observação, não necessariamente início de atividade')


def cases(data,workspace):
    result=[]
    for region in REGIONS:
        for year in range(2019,2023):
            for layer in LAYERS:
                raster=data/region/str(year)/'inputs'/f'{layer}.tif'
                shp=(workspace/'Projeto/Treinamento/Inputs'/f'{layer}_{year}.shp' if region=='regiao_treino'
                     else workspace/'Temporal/Test'/f'{layer}_ate_{year}.shp')
                paths=[raster]+[shp.with_suffix(ext) for ext in ('.shp','.shx','.dbf','.prj')]
                for path in paths:
                    if not path.is_file(): raise FileNotFoundError(path)
                result.append(dict(region=region,year=year,layer=layer,raster=raster,shp=shp))
    return result


def load(case):
    raw,record_ids,records=read_points(case['shp'])
    audit=temporal_audit(case['shp'],case['layer'],case['year'],record_ids)
    with rasterio.open(case['raster']) as ds:
        if not ds.crs or not ds.crs.is_projected or ds.crs.linear_units_factor[1]!=1 or ds.count!=1 or ds.transform.b or ds.transform.d:
            raise ValueError(f'Exige raster monobanda em projeção métrica sem rotação: {ds.name}')
        crs=CRS.from_wkt(ds.crs.to_wkt())
        converter=Transformer.from_crs(CRS.from_wkt(case['shp'].with_suffix('.prj').read_text()),crs,always_xy=True)
        if len(raw):
            xx,yy=converter.transform(raw[:,0],raw[:,1]); raw=np.column_stack((xx,yy))
            if not np.isfinite(raw).all(): raise ValueError('Projeção produziu coordenadas inválidas')
        xy,inverse,counts=np.unique(raw,axis=0,return_inverse=True,return_counts=True)
        bounds=ds.bounds
        outside=int(np.count_nonzero((xy[:,0]<bounds.left)|(xy[:,0]>bounds.right)|(xy[:,1]<bounds.bottom)|(xy[:,1]>bounds.top)))
    audit.update(source_records=records,points=len(raw),unique_nodes=len(xy),nodes_outside_raster_bounds=outside)
    return xy,inverse,counts,record_ids,crs,audit


def graph(xy,k,radius):
    if len(xy): return make_graph(xy,k,radius)
    return (None,np.empty((0,2),int),np.array([]),np.array([]),np.array([],int),
            np.array([]),np.array([],int),np.array([],int),0)


def export_graph(folder,xy,counts,record_ids,inverse,crs,g):
    tree,edges,lengths,weights,degree,strength,labels,sizes,nc=g
    tables=[('nos.csv',['id','x_m','y_m','registros_coincidentes','grau','forca','componente','tamanho_componente'],
             ([i,*p,counts[i],degree[i],strength[i],labels[i],sizes[i]] for i,p in enumerate(xy))),
            ('arestas.csv',['origem','destino','distancia_m','peso'],([*e,d,w] for e,d,w in zip(edges,lengths,weights))),
            ('origem_nos.csv',['registro_shp','id_no'],zip(record_ids,inverse))]
    for filename,header,rows in tables:
        with (folder/filename).open('w',encoding='utf8',newline='') as f:
            writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
    matrix=coo_matrix((np.repeat(weights,2),(edges.ravel(),edges[:,::-1].ravel())),shape=(len(xy),len(xy))).tocsr()
    save_npz(folder/'adjacencia.npz',matrix)
    (folder/'crs.wkt').write_text(crs.to_wkt(),encoding='utf8')
    # GeoJSON em WGS84 (RFC7946), pronto para abrir em um SIG.
    converter=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
    ll=np.column_stack(converter.transform(xy[:,0],xy[:,1])) if len(xy) else xy
    for name in ['nos','arestas']:
        with (folder/(name+'.geojson')).open('w',encoding='utf8') as f:
            f.write('{"type":"FeatureCollection","features":[')
            for i in range(len(xy) if name=='nos' else len(edges)):
                if name=='nos':
                    geom=dict(type='Point',coordinates=ll[i].tolist())
                    props=dict(id=i,grau=int(degree[i]),forca=float(strength[i]),componente=int(labels[i]))
                else:
                    a,b=edges[i]; geom=dict(type='LineString',coordinates=ll[[a,b]].tolist())
                    props=dict(origem=int(a),destino=int(b),distancia_m=float(lengths[i]),peso=float(weights[i]))
                if i: f.write(',')
                f.write(json.dumps(dict(type='Feature',geometry=geom,properties=props),separators=(',',':')))
            f.write(']}')


def rasters(source,folder,xy,g,args):
    tree,edges,lengths,weights,degree,strength,labels,sizes,nc=g
    paths={key:folder/(key+'.tif') for key in ('conectividade','suporte')}
    stats=dict(valid=0,supported=0,sum=0.,minimum=None,maximum=None)
    with rasterio.open(source) as src,ExitStack() as stack:
        profile=src.profile.copy()
        profile.update(count=1,dtype='float32',nodata=NODATA,compress='deflate',predictor=3,
                       tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER')
        writers={k:stack.enter_context(rasterio.open(p.with_suffix('.partial.tif'),'w',**profile)) for k,p in paths.items()}
        for k,dst in writers.items():
            dst.set_band_description(1,k)
            dst.update_tags(k=args.k,radius_m=args.radius_km*1000,method='knn_strength_local_mean_v1',source=str(source))
        last=time.perf_counter()
        for row in range(0,src.height,args.block):
            for col in range(0,src.width,args.block):
                win=Window(col,row,min(args.block,src.width-col),min(args.block,src.height-row))
                original=src.read(1,window=win,masked=True)
                invalid=np.ma.getmaskarray(original)|~np.isfinite(original.data)
                a,s=(surface(tree,xy,strength,src.window_transform(win),original.shape,args.radius_km*1000,args.k)
                     if tree is not None else (np.zeros(original.shape,'float32'),np.zeros(original.shape,'float32')))
                valid=a[~invalid]
                if valid.size:
                    stats['valid']+=int(valid.size);stats['supported']+=int(s[~invalid].sum())
                    stats['sum']+=float(valid.sum(dtype='float64'))
                    stats['minimum']=min(stats['minimum'] if stats['minimum'] is not None else float('inf'),float(valid.min()))
                    stats['maximum']=max(stats['maximum'] if stats['maximum'] is not None else -float('inf'),float(valid.max()))
                for key,v in [('conectividade',a),('suporte',s)]:
                    v[invalid]=NODATA;writers[key].write(v,1,window=win)
            if time.perf_counter()-last>20:
                print(f'  raster: {min(row+args.block,src.height)}/{src.height} linhas',flush=True);last=time.perf_counter()
    for p in paths.values(): p.with_suffix('.partial.tif').replace(p)
    stats['mean']=stats.pop('sum')/stats['valid'] if stats['valid'] else None
    return stats


def figures(case,folder,xy,g,dpi):
    tree,edges,lengths,weights,degree,strength,labels,sizes,nc=g
    windows=[]
    with rasterio.open(case['raster']) as ds:
        windows.append(Window(0,0,ds.width,ds.height))
        r,c=ds.index(*xy[int(np.argmax(strength))]) if len(xy) else (ds.height//2,ds.width//2)
        h,w=min(1024,ds.height),min(1024,ds.width)
        windows.append(Window(min(max(0,c-w//2),ds.width-w),min(max(0,r-h//2),ds.height-h),w,h))
    for name,win in zip(['comparacao','detalhe'],windows):
        factor=min(1,1800/max(win.width,win.height))
        shape=(max(1,round(win.height*factor)),max(1,round(win.width*factor)))
        with rasterio.open(case['raster']) as ds:
            original=np.ma.masked_invalid(ds.read(1,window=win,out_shape=shape,masked=True))
            b=rasterio.windows.bounds(win,ds.transform)
        with rasterio.open(folder/'conectividade.tif') as ds:
            a=ds.read(1,window=win,out_shape=shape,masked=True)
        visible=(xy[:,0]>=b[0])&(xy[:,0]<=b[2])&(xy[:,1]>=b[1])&(xy[:,1]<=b[3])
        shown=edges[visible[edges].any(axis=1)]
        fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
        draw(axes,(case['layer'],original,a,xy,shown,strength,visible,(b[0]/1000,b[2]/1000,b[1]/1000,b[3]/1000)))
        fig.suptitle(f"{case['region']} · {case['year']} · {case['layer']}\nNós: {len(xy)} · arestas: {len(edges)} · conexões espaciais hipotéticas",fontsize=12)
        fig.savefig(folder/(name+'.png'),dpi=dpi);plt.close(fig)
    return [list(w.flatten()) for w in windows]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--workspace',type=Path,default=ROOT.parent)
    p.add_argument('--output',type=Path)
    p.add_argument('--k',type=int,default=8)
    p.add_argument('--radius-km',type=float,default=50)
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--block',type=int,default=256)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.k<1 or not np.isfinite(args.radius_km) or args.radius_km<=0 or args.block<1 or args.dpi<72:
        p.error('Parâmetros fora dos limites')
    all_cases=cases(args.data.resolve(),args.workspace.resolve())
    audits={}
    for case in all_cases:
        *_,audit=load(case); key=f"{case['region']}/{case['year']}/{case['layer']}"
        audits[key]=audit
        print(f"{key}: {audit['unique_nodes']} nós; ano máximo {audit['max_year']}",flush=True)
    if args.check_only:
        print('Pré-verificação concluída: 32 fontes, grades e cortes temporais válidos.',flush=True);return
    config=dict(k=args.k,radius_km=args.radius_km,dpi=args.dpi,scipy=scipy.__version__,
        data=str(args.data.resolve()),workspace=str(args.workspace.resolve()),
        code_hashes={name:digest(ROOT/name) for name in SCRIPTS})
    out=(args.output or ROOT/'outputs/graph_inputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    out.mkdir(parents=True,exist_ok=True); mf=out/'manifest.json'
    if mf.exists():
        manifest=json.loads(mf.read_text(encoding='utf8'))
        if not args.resume or manifest['config']!=config: p.error('Use --resume com mesmo código e parâmetros')
    else:
        if any(out.iterdir()): p.error('Use pasta de saída vazia')
        manifest=dict(status='running',config=config,audits=audits,results={},
                      provenance_limitation='Feições anuais documentadas; equivalência exata com fontes históricas dos KDEs não demonstrada')
        code=out/'source_code';code.mkdir()
        for name in SCRIPTS: shutil.copy2(ROOT/name,code/name)
        save(mf,manifest)
    for i,case in enumerate(all_cases,1):
        key=f"{case['region']}/{case['year']}/{case['layer']}"
        sources=[case['raster']]+[case['shp'].with_suffix(ext) for ext in ('.shp','.shx','.dbf','.prj')]
        fingerprints={str(f):digest(f) for f in sources}
        folder=out/key;folder.mkdir(parents=True,exist_ok=True)
        old=manifest['results'].get(key)
        if old:
            if old['source_hashes']!=fingerprints: raise ValueError(f'Fonte alterada: {key}')
            if all((folder/name).exists() and digest(folder/name)==h for name,h in old['output_hashes'].items()):
                print(f'[{i}/{len(all_cases)}] Já concluído: {key}',flush=True);continue
        print(f'[{i}/{len(all_cases)}] Gerando {key}',flush=True);start=time.perf_counter()
        xy,inverse,counts,ids,crs,audit=load(case)
        g=graph(xy,args.k,args.radius_km*1000)
        export_graph(folder,xy,counts,ids,inverse,crs,g)
        stats=rasters(case['raster'],folder,xy,g,args)
        wins=figures(case,folder,xy,g,args.dpi)
        manifest['results'][key]=dict(source_hashes=fingerprints,nodes=len(xy),edges=len(g[1]),
            components=int(g[-1]),isolated=int(np.count_nonzero(g[4]==0)),raster=stats,
            figure_windows=wins,seconds=time.perf_counter()-start,audit=audit,
            output_hashes={f.name:digest(f) for f in folder.iterdir() if f.is_file()})
        save(mf,manifest)
    manifest['status']='complete';save(mf,manifest)
    print(f'Concluído: {out}',flush=True)


if __name__=='__main__': main()
