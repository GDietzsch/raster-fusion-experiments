"""Scientific input-support maps for a retrospective quadrant diagnosis."""
from pathlib import Path
import csv,json
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm,SymLogNorm
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]/'raster_fusion_ai';PREV=OUT.parent/'Aerospace_pontos_previsoes_20261007'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
cache=ROOT/'runs/enriched_cnn/20260926T223539611207Z/aligned_inputs'
arr=np.load(cache/'teste_final_2021.npy',mmap_mode='r')
d=json.loads((OUT/'DIAGNOSTIC.json').read_text());l,b,r,t=d['definition']['bounds'];xm=d['definition']['x_less_than_m'];ym=d['definition']['y_greater_equal_m'];extent=[l/1000,r/1000,b/1000,t/1000]
with (OUT/'POINT_INPUTS_AND_PREDICTIONS.csv').open(encoding='utf-8-sig',newline='') as f:points=list(csv.DictReader(f))
x=np.array([float(p['x'])/1000 for p in points]);y=np.array([float(p['y'])/1000 for p in points]);nw=np.array([p['NW']=='1' for p in points]);valid=np.isfinite(arr).all(axis=0)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})

def axes(ax,points_on=True):
    ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=8)
    ax.set_xlabel('Projected easting (km)',fontsize=9);ax.set_ylabel('Projected northing (km)',fontsize=9)
    ax.add_patch(Rectangle((l/1000,ym/1000),(xm-l)/1000,(t-ym)/1000,fill=False,edgecolor='#c51b4c',linewidth=1.8,zorder=5))
    if points_on:
        ax.scatter(x[~nw],y[~nw],s=7,facecolors='none',edgecolors='#333333',linewidths=.3,alpha=.5,zorder=3)
        ax.scatter(x[nw],y[nw],s=24,c='#c51b4c',marker='x',linewidths=1,zorder=4)

def finish(fig,name):
    for ext in ['png','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=600,facecolor='white')
    plt.close(fig)

def overview():
    fig,axs=plt.subplots(2,2,figsize=(11.8,11.6));fig.subplots_adjust(left=.08,right=.945,top=.91,bottom=.145,hspace=.30,wspace=.30)
    count=np.sum(arr!=0,axis=0);count=np.ma.array(count,mask=~valid)
    cm=ListedColormap(['#fafafa','#d0e6e8','#8fc4c9','#4596a0','#145a6b']);norm=BoundaryNorm([-.5,.5,6.5,12.5,18.5,24.5],5)
    im=axs[0,0].imshow(count,extent=extent,cmap=cm,norm=norm,interpolation='nearest');axes(axs[0,0]);axs[0,0].set_title('(a) Nonzero input channels | 2021',fontsize=11)
    cb=fig.colorbar(im,ax=axs[0,0],fraction=.035,pad=.025,ticks=[0,3.5,9.5,15.5,21.5]);cb.ax.set_yticklabels(['0','1-6','7-12','13-18','19-24']);cb.set_label('Nonzero channels (of 24)',fontsize=8);cb.ax.tick_params(labelsize=8)
    support=np.ma.array(np.sum(arr[[5,11,17,23]]>0,axis=0),mask=~valid)
    im=axs[0,1].imshow(support,extent=extent,cmap=cm,vmin=-.5,vmax=4.5,interpolation='nearest');axes(axs[0,1]);axs[0,1].set_title('(b) Themes with graph support | 2021',fontsize=11)
    cb=fig.colorbar(im,ax=axs[0,1],fraction=.035,pad=.025,ticks=range(5));cb.set_label('Themes with a node within 50 km',fontsize=8);cb.ax.tick_params(labelsize=8)
    for ax,key,panel in [(axs[1,0],'hgb','c'),(axs[1,1],'persistence','d')]:
        m=json.loads((PREV/f'data/metrics/external_{key}_H1_2022.json').read_text())
        with rasterio.open(m['hotspot']) as src:a=src.read(1,masked=True)
        ax.imshow(a,extent=extent,cmap=ListedColormap(['#eef1f3','#f4b942']),vmin=0,vmax=1,interpolation='nearest');axes(ax,False)
        hits=np.array([p[f'{key}_H1_2022_hitP90']=='1' for p in points])
        ax.scatter(x[~nw],y[~nw],s=7,facecolors='none',edgecolors='#555555',linewidths=.3,alpha=.45)
        ax.scatter(x[nw&~hits],y[nw&~hits],s=25,c='#c51b4c',marker='x',linewidths=1)
        ax.scatter(x[nw&hits],y[nw&hits],s=25,facecolors='none',edgecolors='#151515',linewidths=1)
        name='HGB' if key=='hgb' else 'Persistence (2021 target)'
        ax.set_title(f'({panel}) {name} | NW hits: {np.sum(nw&hits)}/31',fontsize=11)
    fig.suptitle('Why was the northwestern external quadrant poorly prioritized?',fontsize=15,y=.983)
    fig.text(.5,.952,'H1: 2021 inputs to 2022 cumulative target | All 31 northwestern records are dated in 2021',ha='center',fontsize=10)
    handles=[Patch(facecolor='white',edgecolor='#c51b4c',label='Diagnosed quadrant'),Line2D([],[],color='#c51b4c',marker='x',linestyle='none',label='NW record / outside P90 hotspot'),Line2D([],[],color='black',marker='o',markerfacecolor='none',linestyle='none',label='NW record inside P90 hotspot'),Patch(facecolor='#f4b942',label='P90 hotspot')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.018,'Valid zero values are retained. Raw records and native 250 m cells are used; overlapping markers do not remove records.',ha='center',fontsize=9)
    finish(fig,'NW_INPUT_SUPPORT_AND_HOTSPOTS')

def atlas():
    names=['Mining sites','Mineral occurrences','Regular airstrips','Irregular airstrips']
    fig,axs=plt.subplots(2,4,figsize=(17,9.4));fig.subplots_adjust(left=.055,right=.965,top=.87,bottom=.12,wspace=.45,hspace=.42)
    for j,name in enumerate(names):
        a=np.ma.array(arr[j*6],mask=~valid)
        cm=plt.get_cmap('viridis').copy();cm.set_bad('white')
        im=axs[0,j].imshow(a,extent=extent,cmap=cm,norm=SymLogNorm(linthresh=.01,vmin=0,vmax=float(a.max()),base=10),interpolation='nearest');axes(axs[0,j]);axs[0,j].set_title(name+'\nOriginal KDE',fontsize=11)
        cb=fig.colorbar(im,ax=axs[0,j],fraction=.035,pad=.025);cb.ax.tick_params(labelsize=7);cb.set_label('Raw KDE intensity (symlog scale)',fontsize=7)
        a=np.ma.array(arr[j*6+5],mask=~valid)
        axs[1,j].imshow(a,extent=extent,cmap=ListedColormap(['#f4f5f6','#448b92']),vmin=0,vmax=1,interpolation='nearest');axes(axs[1,j]);axs[1,j].set_title(name+'\nGraph support within 50 km',fontsize=11)
    fig.suptitle('External contextual inputs used for H1 2021-2022',fontsize=17,y=.985)
    fig.text(.5,.939,'2021 input layers | The red rectangle identifies the northwest quadrant; red crosses mark its 31 detection records',ha='center',fontsize=11)
    handles=[Patch(facecolor='#f4f5f6',label='No graph node within 50 km'),Patch(facecolor='#448b92',label='At least one graph node within 50 km')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False,fontsize=10)
    fig.text(.5,.026,'Top panels use separate intensity scales. Gradients and graph connectivity were also inspected numerically; all 24 point values are provided in the diagnostic CSV.',ha='center',fontsize=10)
    finish(fig,'NW_ORIGINAL_AND_GRAPH_INPUTS')

if __name__=='__main__':
    overview();atlas();print('Two diagnostic figures saved: PNG 600 dpi and SVG.',flush=True)
