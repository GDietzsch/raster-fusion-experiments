"""English scientific maps: original predicted intensity and point-level P90 hits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,csv,json,gc
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator,ScalarFormatter

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]/'raster_fusion_ai'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'svg.fonttype':'none','savefig.facecolor':'white'})
VMAX={}
FORCE=False
def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def vmax(region,year):
 key=(region,year)
 if key not in VMAX:
  with rasterio.open(ROOT/f'data/{"regiao_treino" if region=="internal" else "regiao_teste_final"}/{year}/target.tif') as d:VMAX[key]=float(d.read(1,masked=True).max())
 return VMAX[key]
def axes_format(ax,extent):
 ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal')
 ax.set_xlabel('Projected easting (km)');ax.set_ylabel('Projected northing (km)')
 ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
 fmt=ScalarFormatter(useOffset=False);fmt.set_scientific(False);ax.xaxis.set_major_formatter(fmt)
 fmt=ScalarFormatter(useOffset=False);fmt.set_scientific(False);ax.yaxis.set_major_formatter(fmt)
 ax.tick_params(labelsize=8)

def plot_model(m):
 stem=m['stem'];folder=OUT/'figures'/m['region'];folder.mkdir(parents=True,exist_ok=True)
 done=folder/f'{stem}.done.json'
 if done.exists():return
 stats=m['thresholds'][1];points=read_csv(Path(m['point_csv']))
 points=[p for p in points if p['eligible']=='1']
 x=np.array([float(p['x'])/1000 for p in points]);y=np.array([float(p['y'])/1000 for p in points]);hits=np.array([p['hit_p90']=='1' for p in points])
 with rasterio.open(m['prediction']) as src:
  # Display raster sampled to <=2400 columns for publication panels; all calculations
  # and delivered GeoTIFFs retain native 250 m cells. Point positions are never moved.
  shape=(round(src.height*min(1,2400/src.width)),min(src.width,2400))
  a=src.read(1,out_shape=shape,masked=True,resampling=rasterio.enums.Resampling.nearest)
  b=src.bounds;extent=[b.left/1000,b.right/1000,b.bottom/1000,b.top/1000]
 with rasterio.open(m['hotspot']) as src:hot=src.read(1,out_shape=shape,masked=True,resampling=rasterio.enums.Resampling.nearest)
 ratio=shape[0]/shape[1];height=4.8 if ratio<.65 else 6.7
 fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.6,height))
 fig.subplots_adjust(left=.065,right=.945,bottom=.23,top=.81,wspace=.23)
 cmap=plt.get_cmap('viridis').copy();cmap.set_bad('white');cmap.set_under('#4575b4');cmap.set_over('#c51b7d')
 im=ax1.imshow(a,extent=extent,cmap=cmap,norm=Normalize(0,vmax(m['region'],m['target_year'])),interpolation='nearest')
 ax1.scatter(x,y,s=7,facecolors='none',edgecolors='white',linewidths=.35,zorder=3)
 ax1.set_title('(a) Predicted intensity and detection records',pad=9)
 ax2.imshow(hot,extent=extent,cmap=ListedColormap(['#edf0f2','#f4b942']),vmin=0,vmax=1,interpolation='nearest')
 ax2.scatter(x[~hits],y[~hits],s=11,c='#1769a8',marker='x',linewidths=.7,zorder=3)
 ax2.scatter(x[hits],y[hits],s=10,facecolors='none',edgecolors='#171717',linewidths=.6,zorder=4)
 ax2.set_title(f'(b) P90 hotspots: {stats["cumulative"]["hits"]:,}/{m["eligible_points"]:,} records ({stats["cumulative"]["hit_percent"]:.2f}%)',pad=9)
 for ax in (ax1,ax2):axes_format(ax,extent)
 cax=fig.add_axes([.953,.31,.012,.34]);bar=fig.colorbar(im,cax=cax,extend='both');bar.set_label('Predicted KDE intensity (raw scale)',fontsize=8);bar.ax.tick_params(labelsize=7)
 handles=[Patch(facecolor='#f4b942',label=f'Hotspot: intensity >= {stats["threshold"]:.4g}'),Line2D([],[],color='black',marker='o',markerfacecolor='none',linestyle='none',markersize=5,label='Detection inside hotspot'),Line2D([],[],color='#1769a8',marker='x',linestyle='none',markersize=5,label='Detection outside hotspot')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),ncol=3,frameon=False,fontsize=8)
 region='Internal region (descriptive)' if m['region']=='internal' else 'External region'
 fig.suptitle(f'{m["model"]} | {region} | H{m["horizon"]}: {m["input_year"]} to {m["target_year"]}',fontsize=15,y=.98)
 yearstat=stats['current_year'];fig.text(.5,.89,f'Cumulative records through {m["target_year"]}; P90 hotspot area = {stats["hot_area_percent"]:.2f}% | Current-year coverage: {yearstat["hits"]}/{yearstat["n"]} ({yearstat["hit_percent"]:.2f}%)',ha='center',fontsize=9)
 foot=('Internal maps include fitting locations. ' if m['region']=='internal' else '')+f'Excluded records: {m["source_points"]-m["eligible_points"]}. Records are not unique flights. All threshold ties are included.'
 fig.text(.5,.048,foot,ha='center',fontsize=8)
 fig.text(.5,.016,'Scores and thresholds use native 250 m cells; display uses nearest-neighbour sampling. CRS: South America Lambert Conformal Conic / SAD69.',ha='center',fontsize=7)
 for ext in ['png','svg']:fig.savefig(folder/f'{stem}.{ext}',dpi=600)
 plt.close(fig);done.write_text(json.dumps({'png_dpi':600,'display_shape':shape,'point_count':len(points),'native_calculation':True}),encoding='utf-8')
 print('PLOT',stem,flush=True);gc.collect()

def plot_target(region,year):
 folder=OUT/'figures/source_points';folder.mkdir(parents=True,exist_ok=True);stem=f'{region}_target_points_{year}'
 if (folder/f'{stem}.svg').exists() and not FORCE:return
 points=read_csv(OUT/f'data/points_{region}_{year}.csv')
 x=np.array([float(p['x'])/1000 for p in points]);y=np.array([float(p['y'])/1000 for p in points]);annual=np.array([int(p['record_year'])==year for p in points])
 target=ROOT/f'data/{"regiao_treino" if region=="internal" else "regiao_teste_final"}/{year}/target.tif'
 with rasterio.open(target) as src:
  shape=(round(src.height*min(1,3600/src.width)),min(src.width,3600));a=src.read(1,out_shape=shape,masked=True,resampling=rasterio.enums.Resampling.nearest)
  b=src.bounds;extent=[b.left/1000,b.right/1000,b.bottom/1000,b.top/1000]
 fig,ax=plt.subplots(figsize=(11,7.5 if region=='internal' else 10));fig.subplots_adjust(left=.12,right=.87,bottom=.17,top=.84)
 cmap=plt.get_cmap('viridis').copy();cmap.set_bad('white')
 im=ax.imshow(a,extent=extent,cmap=cmap,vmin=0,vmax=vmax(region,year),interpolation='nearest')
 ax.scatter(x[~annual],y[~annual],s=10,facecolors='#454545',edgecolors='white',linewidths=.45,label=f'Earlier records (n = {int((~annual).sum()):,})')
 ax.scatter(x[annual],y[annual],s=15,facecolors='none',edgecolors='#f47c38',linewidths=.65,label=f'{year} records (n = {int(annual.sum()):,})')
 display_extent=[min(extent[0],float(x.min())),max(extent[1],float(x.max())),min(extent[2],float(y.min())),max(extent[3],float(y.max()))]
 dx=(display_extent[1]-display_extent[0])*.01;dy=(display_extent[3]-display_extent[2])*.01
 display_extent=[display_extent[0]-dx,display_extent[1]+dx,display_extent[2]-dy,display_extent[3]+dy]
 axes_format(ax,display_extent);fig.colorbar(im,ax=ax,fraction=.025,pad=.035,label='Observed KDE intensity (raw scale)')
 fig.suptitle(f'{region.title()} region | Observed target and source detections | {year}',fontsize=14,y=.97)
 fig.text(.5,.90,f'{len(points):,} cumulative point records | Quartic KDE, 41 km radius | Native 250 m grid',ha='center',fontsize=10)
 ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False,fontsize=9)
 fig.text(.5,.025,'Point locations and record multiplicities are preserved. Markers may overlap. Records are not unique flights.\nDisplay uses nearest-neighbour sampling; source rasters and point coordinates remain unchanged.',ha='center',fontsize=8)
 for ext in ['png','svg']:fig.savefig(folder/f'{stem}.{ext}',dpi=600)
 plt.close(fig);print('PLOT',stem,flush=True);gc.collect()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('region',choices=['internal','external','sources']);p.add_argument('--limit',type=int);p.add_argument('--force',action='store_true');args=p.parse_args();FORCE=args.force
 if args.region=='sources':
  for reg in ['internal','external']:
   for year in [2020,2021,2022]:plot_target(reg,year)
 else:
  paths=sorted((OUT/'data/metrics').glob(args.region+'_*.json'))
  if args.limit:paths=paths[:args.limit]
  for path in paths:plot_model(json.loads(path.read_text()))
