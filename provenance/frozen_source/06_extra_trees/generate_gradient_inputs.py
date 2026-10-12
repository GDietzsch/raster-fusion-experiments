"""Magnitude e direção do gradiente em todos os inputs de treino e teste."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent
NODATA = -9999.
PRODUCTS = ('magnitude', 'direcao_graus', 'direcao_seno', 'direcao_cosseno')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    tmp.replace(path)


def gradients(a, dx, dy):
    """Halo de 1 pixel. dx/dy são passos assinados em metros da transformação."""
    gx = (a[1:-1, 2:] - a[1:-1, :-2])/(2*dx)
    gy = (a[2:, 1:-1] - a[:-2, 1:-1])/(2*dy)
    valid = np.isfinite(a[1:-1,1:-1]) & np.isfinite(gx) & np.isfinite(gy)
    mag = np.hypot(gx,gy)
    active = valid & (mag > 0)
    angle = np.full(mag.shape, NODATA)
    angle[active] = np.degrees(np.arctan2(gy[active],gx[active]))
    sine = np.divide(gy,mag,out=np.zeros_like(mag),where=active)
    cosine = np.divide(gx,mag,out=np.zeros_like(mag),where=active)
    result = [mag,angle,sine,cosine]
    for v in result: v[~valid] = NODATA
    return {k:v.astype('float32') for k,v in zip(PRODUCTS,result)}


def validate(ds):
    t = ds.transform
    if ds.count != 1 or t.b or t.d or not t.a or not t.e or min(ds.shape)<3:
        raise ValueError(f'Exige banda única e grade sem rotação: {ds.name}')
    if not ds.crs or not ds.crs.is_projected:
        raise ValueError(f'Exige CRS projetado: {ds.name}')
    return ds.crs.linear_units_factor[1]


def process(source, paths, block):
    started=time.perf_counter()
    with rasterio.open(source) as src, ExitStack() as stack:
        factor=validate(src)
        profile=src.profile.copy()
        profile.update(dtype='float32',nodata=NODATA,count=1,compress='deflate',predictor=3,
                       tiled=True,blockxsize=256,blockysize=256,BIGTIFF='IF_SAFER')
        writers={k:stack.enter_context(rasterio.open(p.with_suffix('.partial.tif'),'w',**profile)) for k,p in paths.items()}
        for k,ds in writers.items():
            ds.set_band_description(1,k)
            ds.update_tags(method='central_difference_physical_v1',source=str(source),
                           flat_direction='degrees_nodata_sin_cos_zero',direction='0 east, 90 north in projected axes')
        last=time.perf_counter()
        for row in range(0,src.height,block):
            for col in range(0,src.width,block):
                h,w=min(block,src.height-row),min(block,src.width-col)
                a=np.full((h+2,w+2),np.nan)
                r0,c0=max(0,row-1),max(0,col-1)
                r1,c1=min(src.height,row+h+1),min(src.width,col+w+1)
                v=src.read(1,window=Window(c0,r0,c1-c0,r1-r0),masked=True,out_dtype='float64').filled(np.nan)
                ar,ac=r0-row+1,c0-col+1
                a[ar:ar+r1-r0,ac:ac+c1-c0]=v
                results=gradients(a,src.transform.a*factor,src.transform.e*factor)
                for k,ds in writers.items(): ds.write(results[k],1,window=Window(col,row,w,h))
            if time.perf_counter()-last>20:
                print(f'  {min(row+block,src.height)}/{src.height} linhas',flush=True)
                last=time.perf_counter()
    for p in paths.values(): p.with_suffix('.partial.tif').replace(p)
    return time.perf_counter()-started


def panel(source, paths, output, dpi, window=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
    settings=[(source,'Original','viridis',None,None,'Intensidade original'),
              (paths['magnitude'],'Magnitude','viridis',0,None,'Intensidade / metro'),
              (paths['direcao_graus'],'Direção','twilight',-180,180,'Graus: 0 leste · 90 norte')]
    display={}
    for ax,(path,title,cmap,lo,hi,label) in zip(axes,settings):
        with rasterio.open(path) as ds:
            win=window or Window(0,0,ds.width,ds.height)
            scale=min(1,2000/max(win.width,win.height))
            shape=(max(1,round(win.height*scale)),max(1,round(win.width*scale)))
            a=np.ma.masked_invalid(ds.read(1,window=win,out_shape=shape,masked=True,resampling=Resampling.nearest))
            b=rasterio.windows.bounds(win,ds.transform)
            factor=ds.crs.linear_units_factor[1]/1000
            values=a.compressed()
            if lo is None: lo=float(values.min()) if values.size else 0
            if hi is None: hi=float(values.max()) if values.size else 1
            if hi<=lo: hi=lo+1
            im=ax.imshow(a,extent=(b[0]*factor,b[2]*factor,b[1]*factor,b[3]*factor),
                         cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
            ax.set(title=title,xlabel='X (km)',ylabel='Y (km)',facecolor='#dddddd')
            fig.colorbar(im,ax=ax,orientation='horizontal',pad=.1,label=label)
            display[title]=dict(vmin=lo,vmax=hi,preview_shape=list(shape))
    fig.suptitle(' / '.join(source.parts[-4:])+'\nCinza: sem dado ou direção indefinida',fontsize=12)
    fig.savefig(output,dpi=dpi)
    plt.close(fig)
    return display


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path)
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--block',type=int,default=512)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.block<1 or args.dpi<72: p.error('Bloco >=1 e dpi >=72')
    data=args.data.resolve()
    files=sorted(data.glob('regiao_*/*/inputs/*.tif'))
    if not files: p.error('Nenhum input em regiao_*/ANO/inputs/*.tif')
    for f in files:
        with rasterio.open(f) as ds: validate(ds)
        print(f.relative_to(data),flush=True)
    print(f'{len(files)} inputs: magnitude + direção (graus, seno, cosseno)',flush=True)
    if args.dry_run: return
    out=(args.output or ROOT/'outputs/gradient_inputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    config=dict(data=str(data),inputs=[str(f) for f in files],dpi=args.dpi,
                script_sha256=digest(Path(__file__)),method='central_difference_physical_v1')
    mf=out/'manifest.json'
    if mf.exists():
        manifest=json.loads(mf.read_text(encoding='utf8'))
        if not args.resume or manifest['config']!=config: p.error('Use --resume com mesmos dados, parâmetros e script')
    else:
        if any(out.iterdir()): p.error('Use uma pasta de saída vazia')
        manifest=dict(status='running',config=config,results={})
        shutil.copy2(__file__,out/Path(__file__).name)
        save(mf,manifest)
    for i,source in enumerate(files,1):
        rel=source.relative_to(data)
        key=rel.as_posix()
        folder=out/rel.parent
        folder.mkdir(parents=True,exist_ok=True)
        paths={k:folder/(source.stem+'_'+k+'.tif') for k in PRODUCTS}
        pngs=[folder/(source.stem+'_comparacao.png'),folder/(source.stem+'_detalhe.png')]
        products=list(paths.values())+pngs
        h=digest(source)
        if key in manifest['results']:
            record=manifest['results'][key]
            if record['input_sha256']!=h: raise ValueError(f'Input alterado: {source}')
            if all(f.exists() and digest(f)==record['hashes'].get(f.name) for f in products):
                print(f'[{i}/{len(files)}] Já concluído: {key}',flush=True)
                continue
        print(f'[{i}/{len(files)}] Processando {key}',flush=True)
        seconds=process(source,paths,args.block)
        display=panel(source,paths,pngs[0],args.dpi)
        with rasterio.open(source) as ds:
            shape=(min(512,ds.height),min(512,ds.width))
            preview=np.ma.masked_invalid(ds.read(1,out_shape=shape,masked=True))
            iy,ix=np.unravel_index(np.argmax(preview.filled(-np.inf)),shape)
            wh,ww=min(512,ds.height),min(512,ds.width)
            r=min(max(0,int((iy+.5)*ds.height/shape[0])-wh//2),ds.height-wh)
            c=min(max(0,int((ix+.5)*ds.width/shape[1])-ww//2),ds.width-ww)
        zoom=panel(source,paths,pngs[1],args.dpi,Window(c,r,ww,wh))
        manifest['results'][key]=dict(input_sha256=h,seconds=seconds,display=display,
            zoom_display=zoom,zoom_window=[c,r,ww,wh],hashes={f.name:digest(f) for f in products})
        save(mf,manifest)
    manifest['status']='complete'
    save(mf,manifest)
    print(f'Concluído: {out}',flush=True)


if __name__=='__main__': main()
