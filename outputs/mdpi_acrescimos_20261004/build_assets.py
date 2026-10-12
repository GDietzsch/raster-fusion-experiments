from pathlib import Path
import importlib.util, json, hashlib, re, collections, shutil, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, SymLogNorm
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw

OUT=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[2] / 'raster_fusion_ai')
for name in ('figures','tables','supplement','qa'):(OUT/name).mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('source',ROOT/'build_final_interpretation.py')
src=importlib.util.module_from_spec(spec);spec.loader.exec_module(src)
d,e,inventory,ch,configs=src.collect()
c,ce,u,ue=src.control_data()
w=d[d.vencedor].copy()
assert (len(d),len(e),len(w),len(c),len(ce),len(u),len(ue))==(148,60,24,4,10,24,57)
assert not e.duplicated(['modelo','par']).any()
assert e.pixels.nunique()==1 and e.pixels.iloc[0]==5868615
for p,g in e.groupby('par'):
    assert g.target_sha256.nunique()==1
    assert g.zero_rmse.max()-g.zero_rmse.min()<1e-10
for row in e.itertuples():
    m=json.loads((ROOT/row.fonte).read_text(encoding='utf-8-sig'))
    if m.get('reference_persistence'):
        assert m['reference_persistence']['common_pixels']==row.pixels
    assert m['uncertainty']['occupied_blocks']==42
    assert m['uncertainty']['replicates']==1000
assert (e.rmse<e.zero_rmse).sum()==3
assert (e.rmse<e.media_rmse).sum()==1
assert (e.rmse<e.persistencia_rmse).sum()==0
assert (e.r2>0).sum()==0
# Compare direct JSON extraction with the earlier consolidated CSVs.
for new,old,cols in [(d,'todos_148_ajustes.csv',['validacao_rmse','treino_rmse']),
                     (e,'externo_60_pares.csv',['rmse','mae','capture_10pct'])]:
    saved=pd.read_csv(ROOT/'reports/interpretacao_geral_20260929/tabelas'/old)
    for col in cols:assert np.allclose(new[col],saved[col],rtol=1e-12,atol=1e-12)

arms={'originais':'Original','originais_filtradas':'Selected original','gpa':'Gradient',
      'grafos':'Graph','todas':'All','todas_filtradas':'Selected all',
      'originais_gpa':'Original + gradient','originais_grafos':'Original + graph'}
modelnames={'Linear':'Linear','ResNet':'Compact ResNet','Inception':'Dense Inception'}
models=src.MODELS
labels=[modelnames.get(m,m) for m in models]
pairs=src.PAIRS
pairlabels=[p.replace('→',' → ') for p in pairs]
rename={'experimento':'Experiment','modelo':'Model','horizonte':'Horizon','conjunto':'Input set',
 'canais':'Channels','vencedor':'Selected winner','fonte':'Source JSON','nomes_canais':'Channel names',
 'epocas':'Final epochs','arvores_iteracoes':'Trees or iterations','lambda':'Lambda per sample',
 'minutos_run':'Run duration (min)','parametros':'Parameters','par':'Evaluation pair','pixels':'Common pixels',
 'target_sha256':'Target SHA256','prediction':'Prediction GeoTIFF','media_rmse':'Training-mean RMSE',
 'zero_rmse':'Zero RMSE','persistencia_rmse':'Persistence RMSE','persistencia_captura':'Persistence capture at 10%',
 'positivos_pct':'Positive target pixels (%)','referencia_rmse':'Two-scale RMSE',
 'variante':'Variant','centros':'Training centers','capture':'Capture at 10%','epocas':'Final epochs'}
phenomena={'Garimpos':'Mining sites','Ocorrencias_Minerais':'Mineral occurrences',
 'Pistas_Regulares':'Regular airstrips','Pistas_Irregulares':'Irregular airstrips'}
products={'original':'original','magnitude':'magnitude','direcao_seno':'direction sine',
 'direcao_cosseno':'direction cosine','conectividade':'connectivity','suporte':'support'}
def channel(s):return '; '.join(phenomena[x.split('__')[0]]+' / '+products[x.split('__')[1]] for x in s.split(';'))
def english(df):
    df=df.copy()
    if 'modelo' in df:df['modelo']=df.modelo.replace(modelnames)
    if 'conjunto' in df:df['conjunto']=df.conjunto.map(arms)
    if 'nomes_canais' in df:df['nomes_canais']=df.nomes_canais.map(channel)
    def rn(k):
        if k in rename:return rename[k]
        for old,new in [('treino_','Training '),('validacao_','Validation '),('selecao_','Selection ')]:
            if k.startswith(old):return new+k[len(old):]
        return k
    return df.rename(columns=rn)
def table(df,name,caption):
    df.to_csv(OUT/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig')
    lines=[caption,'','| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']
    def fmt(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x)
    lines+=['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    (OUT/'tables'/f'{name}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for df,name in [(d,'S1_all_148_final_fits'),(e,'S2_all_60_external_evaluations'),(ce,'S3_local_control_external'),(u,'S4_unet_24_fits'),(ue,'S5_unet_57_external')]:
    english(df).to_csv(OUT/'supplement'/f'{name}.csv',index=False,encoding='utf-8-sig')
table(english(w[['modelo','horizonte','conjunto','canais','validacao_rmse','validacao_mae','validacao_r2']]),'Table_2_internal_winners','Table 2. Internally selected configurations. H0 and H1 use different validation designs and are reported separately.')
base=e.drop_duplicates('par').set_index('par').loc[pairs]
b=base[['pixels','positivos_pct','zero_rmse','media_rmse','persistencia_rmse']].copy()
# Training-mean baselines vary by model normalization: use Linear as an explicit representative.
b=base[['pixels','positivos_pct','zero_rmse','persistencia_rmse']].copy()
b.insert(0,'Evaluation pair',pairlabels)
table(english(b.reset_index(drop=True)),'Table_3_external_targets','Table 3. External target support and reference errors. Training-mean baselines are model-specific and are retained in Supplementary Table S2. NA: no earlier target available.')
for metric,name,cap in [('rmse','Table_4_external_RMSE','Table 4. External RMSE of the 24 frozen internal winners across five input-target pairs. Lower is better.'),('capture_10pct','Table_S6_external_capture','Supplementary Table S6. Target mass captured in the top-ranked 10% of valid pixels (%). Higher is better.')]:
    z=e.pivot(index='modelo',columns='par',values=metric).reindex(index=models,columns=pairs)
    if metric.startswith('capture'):z*=100
    z.index=labels;z.index.name='Model';z.columns=pairlabels
    table(z.reset_index(),name,cap)
summary=u.groupby(['experimento','horizonte','variante'],sort=True).rmse.agg(['mean','std']).reset_index()
summary['Seeds']=3
table(english(summary).rename(columns={'mean':'Mean validation RMSE','std':'Sample SD'}),'Table_S7_unet_controls','Supplementary Table S7. Mean and sample standard deviation across three seeds; these are not confidence intervals or ensemble scores.')
english(w[['modelo','horizonte','conjunto','nomes_canais','epocas','arvores_iteracoes','lambda','parametros']]).to_csv(OUT/'supplement/S8_winner_channels_parameters.csv',index=False,encoding='utf-8-sig')
for _,r in inventory.iterrows():
    folder=OUT/'supplement'/'frozen_configurations'/f'{int(r.experimento):02d}'
    folder.mkdir(parents=True,exist_ok=True)
    for name in ('config.yaml','environment.json'):
        if (ROOT/r.run/name).exists():shutil.copyfile(ROOT/r.run/name,folder/name)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
figmeta=json.loads((OUT/'AUDIT.json').read_text(encoding='utf-8'))['figures'] if (OUT/'AUDIT.json').exists() else {}
def save(name):
    plt.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight')
    plt.savefig(OUT/'figures'/f'{name}.png',dpi=600,bbox_inches='tight')
    plt.close()
    im=Image.open(OUT/'figures'/f'{name}.png')
    figmeta[name]={'pixels':list(im.size),'dpi':im.info.get('dpi'),'vector':'SVG'}
    im.thumbnail((1400,1000));im.convert('RGB').save(OUT/'qa'/f'{name}.jpg')

f,ax=plt.subplots(figsize=(12,7));ax.set_xlim(0,12);ax.set_ylim(0,7);ax.axis('off')
def box(x,y,w,h,text,color='#e6f0f2'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#42616a'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#42616a','lw':1.5})
box(.2,5.7,11.6,1,'Four contextual themes | Annual inputs: 2019–2022\nOriginal KDE: 4 channels + gradient: 12 + graph: 8 = 24 candidate channels')
box(.2,3.35,5.5,1.75,'H0: same-year estimation\nTargets: 2020, 2021, 2022\nZone 1: fit → Zone 2: select → Zone 3: validate\nFinal fit: 600,000 | Validation: 300,000 centers')
box(6.3,3.35,5.5,1.75,'H1: next-year estimation\n2019→2020: fit | 2020→2021: select\n2021→2022: validate | All stages in zone 1\nFinal fit: 200,000 | Validation: 100,000 centers')
arrow((3,5.7),(3,5.1));arrow((9,5.7),(9,5.1))
box(.8,1.85,10.4,.9,'Refit each input set after internal selection; select by validation RMSE\n12 model families × 2 horizons → 24 frozen winners', '#f1eee5')
arrow((3,3.35),(3,2.75));arrow((9,3.35),(9,2.75))
box(.8,.25,10.4,.95,'One external region | 5 input-target pairs | 60 main evaluations\n5,868,615 common pixels per evaluation; controls reported separately','#edf0e7')
arrow((6,1.85),(6,1.2));save('Figure_1_protocol')

f,axes=plt.subplots(1,2,figsize=(14,7))
for h,ax in enumerate(axes):
    z=d[d.horizonte==h].pivot(index='modelo',columns='conjunto',values='validacao_rmse').reindex(index=models,columns=src.ARMS)
    vals=100*(1-z.div(z.originais,axis=0))
    im=ax.imshow(vals,cmap='RdBu',vmin=-100,vmax=100,aspect='auto')
    ax.set_xticks(range(8),[arms[a] for a in src.ARMS],rotation=50,ha='right',fontsize=9)
    ax.set_yticks(range(12),labels);ax.set_title(f'({chr(97+h)}) H{h}')
    for (i,j),v in np.ndenumerate(vals.to_numpy()):ax.text(j,i,'NA' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>50 else 'black')
f.subplots_adjust(bottom=.23,wspace=.42,right=.9)
f.colorbar(im,cax=f.add_axes([.93,.27,.014,.58]),label='Validation RMSE reduction vs original (%)')
save('Figure_2_input_sets')

f,ax=plt.subplots(figsize=(10,7.5))
z=e.pivot(index='modelo',columns='par',values='rmse').reindex(index=models,columns=pairs)
rat=e.pivot(index='modelo',columns='par',values='ratio_zero').reindex(index=models,columns=pairs)
im=ax.imshow(np.log2(rat),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=4),aspect='auto')
ax.set_xticks(range(5),pairlabels,rotation=20,ha='right');ax.set_yticks(range(12),labels)
for (i,j),v in np.ndenumerate(z.to_numpy()):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if np.log2(rat.iloc[i,j])>2.3 else 'black')
cb=f.colorbar(im,ax=ax,label='Model RMSE / zero-reference RMSE',ticks=[-1,0,1,2,3,4]);cb.ax.set_yticklabels(['0.5','1','2','4','8','16'])
ax.set_title('External evaluation | Cell values: RMSE in original target units');f.tight_layout();save('Figure_3_external_RMSE')

f,axes=plt.subplots(1,2,figsize=(14,6))
for h,ax in enumerate(axes):
    sub=e[(e.horizonte==h)&e.par.str.endswith('2022')].set_index('modelo').loc[models]
    ax.scatter(sub.rmse,sub.capture_10pct*100,c=range(12),cmap='tab20',s=55)
    offsets={(0,'CNN'):(5,-13),(0,'XGBoost'):(5,9),(0,'U-Net'):(-38,15),(0,'Inception'):(12,9),(0,'Ridge'):(8,-1),(0,'Linear'):(8,-10),(1,'Linear'):(5,-13),(1,'Ridge'):(5,8),(1,'Inception'):(5,-12)}
    for name,r in sub.iterrows():ax.annotate(modelnames.get(name,name),(r.rmse,r.capture_10pct*100),xytext=offsets.get((h,name),(5,5)),textcoords='offset points',fontsize=8)
    ax.axvline(sub.zero_rmse.iloc[0],ls=':',color='gray',label='Zero-reference RMSE')
    ax.scatter(sub.persistencia_rmse.iloc[0],sub.persistencia_captura.iloc[0]*100,marker='*',s=150,c='black',label='Persistence (historical target)')
    ax.axhline(10,color='gray',lw=.8,ls='--',label='Random-selection expectation')
    ax.set(xlabel='External RMSE',ylabel='Target mass captured in top 10% (%)',title=f'({chr(97+h)}) H{h} | External target: 2022',ylim=(7,48));ax.grid(alpha=.15)
    ax.legend(fontsize=7,loc='upper right')
f.tight_layout();save('Figure_4_error_and_prioritization')

f,axes=plt.subplots(1,2,figsize=(13,5.5))
for h,ax in enumerate(axes):
    groups=list(u[u.horizonte==h].groupby(['experimento','variante'],sort=True))
    for i,((exp,var),g) in enumerate(groups):
        ax.scatter(i+np.array([-.07,0,.07]),g.rmse,color='#246878',s=35)
        ax.errorbar(i,g.rmse.mean(),yerr=g.rmse.std(ddof=1),fmt='D',color='#b36e27',capsize=5)
    ax.set_xticks(range(len(groups)),[f'{k[0]}\n{k[1]}' for k,g in groups],rotation=20,ha='right')
    ax.set(title=f'({chr(97+h)}) H{h}',ylabel='Validation RMSE');ax.grid(axis='y',alpha=.15)
f.tight_layout();save('Figure_5_unet_controls')

f,axes=plt.subplots(1,2,figsize=(14,7))
for h,ax in enumerate(axes):
    sub=e[(e.horizonte==h)&e.par.str.endswith('2021')].set_index('modelo').loc[models]
    yy=np.arange(12);ax.barh(yy-.18,sub.subset_zero_mae,height=.35,label='Zero target',color='#bc833d');ax.barh(yy+.18,sub.subset_positive_mae,height=.35,label='Positive target',color='#246878')
    ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set(xlabel='External MAE',title=f'H{h} | External target: 2021');ax.legend()
f.tight_layout();save('Figure_S1_zero_positive_errors')
f,axes=plt.subplots(1,2,figsize=(14,7))
for h,ax in enumerate(axes):
    sub=e[(e.horizonte==h)&e.par.str.endswith('2022')].set_index('modelo').loc[models]
    ax.errorbar(sub.rmse,np.arange(12),xerr=[sub.rmse-sub.ci_low,sub.ci_high-sub.rmse],fmt='o',color='#246878',capsize=3)
    ax.axvline(sub.zero_rmse.iloc[0],ls='--',color='#bc833d',label='Zero');ax.axvline(sub.persistencia_rmse.iloc[0],ls=':',color='black',label='Persistence')
    ax.set_yticks(range(12),labels);ax.invert_yaxis();ax.set(xlabel='RMSE with exploratory 95% spatial-block interval',title=f'H{h} | External target: 2022');ax.legend()
f.tight_layout();save('Figure_S2_spatial_uncertainty')

# All twelve models, native raster cells embedded in SVG. PNG is a print rendition.
import rasterio
for h in (() if '--charts-only' in sys.argv else (0,1)):
    sub=e[(e.horizonte==h)&e.par.str.endswith('2021')].set_index('modelo').loc[models]
    paths=[ROOT/'data/regiao_teste_final/2021/target.tif']+[Path(p) for p in sub.prediction]
    mask=None;vmax=0
    for p in paths:
        with rasterio.open(p) as ds:a=ds.read(1,masked=True)
        invalid=np.ma.getmaskarray(a)|~np.isfinite(a.data)
        mask=invalid.copy() if mask is None else mask|invalid
    assert (~mask).sum()==5868615
    for p in paths:
        with rasterio.open(p) as ds:a=ds.read(1)
        vmax=max(vmax,float(np.max(np.abs(a[~mask]))))
    f,axes=plt.subplots(4,4,figsize=(18,17),layout='constrained')
    norm=SymLogNorm(linthresh=.1,vmin=-vmax,vmax=vmax,base=10)
    titles=['Observed target 2021']+[f'{modelnames.get(m,m)} | RMSE {sub.loc[m,"rmse"]:.2f}' for m in models]
    for ax,p,title in zip(axes.flat,paths,titles):
        with rasterio.open(p) as ds:a=ds.read(1)
        im=ax.imshow(np.ma.array(a,mask=mask),cmap='RdBu_r',norm=norm,interpolation='none')
        ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    for ax in list(axes.flat)[13:]:ax.set_visible(False)
    f.colorbar(im,ax=list(axes.flat)[:13],shrink=.55,label='Original target intensity | Shared symmetric-log scale (linear threshold: 0.1)')
    f.suptitle(f'H{h} | External target: 2021 | Native grid: 2593 × 2268 cells',fontsize=15)
    save(f'Figure_S{3+h}_external_maps_H{h}')

# Bibliographic audit; do not change the supplied .bib file.
bib=(OUT/'original/bib.bib').read_text(encoding='utf-8-sig')
keys=re.findall(r'@\w+\s*\{\s*([^,]+)',bib)
counts=collections.Counter(keys)
man=(OUT/'original/template.tex').read_text(encoding='utf-8-sig')
cites=[x.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',man) for x in group.split(',')]
audit={'main_fits':len(d),'internal_winners':len(w),'main_external':len(e),'local_fits':len(c),'unet_fits':len(u),'total_external':len(e)+len(ce)+len(ue),
 'common_pixels':5868615,'direct_json_matches_consolidated_csv':True,'native_map_common_mask_checked':True,
 'full_raster_metrics_recomputed':False,'new_training':False,'main_positive_r2':int((e.r2>0).sum()),
 'main_wins_vs_zero':int((e.rmse<e.zero_rmse).sum()),'main_wins_vs_training_mean':int((e.rmse<e.media_rmse).sum()),
 'main_wins_vs_persistence':int((e.rmse<e.persistencia_rmse).sum()),'figures':figmeta,
 'duplicate_bib_keys':{k:v for k,v in counts.items() if v>1},'missing_existing_citations':sorted(set(cites)-set(keys)),
 'source_hashes':src.sources}
(OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k not in ('source_hashes','figures')},indent=2,ensure_ascii=False))
print('Figures:',len(figmeta))
