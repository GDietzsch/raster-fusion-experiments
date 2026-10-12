"""Read frozen experiment artifacts; generate synthesis without fitting any model."""
from pathlib import Path
import csv, hashlib, html, json, re, sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/interpretacao_geral_20260929'
SPECS = [
 ('01','Linear','linear','20260925T141835755712Z'),
 ('02','Ridge','ridge','20260925T191737660876Z'),
 ('03','HGB','hgb','20260925T213151313641Z'),
 ('04','XGBoost','xgboost','20260926T150535837633Z'),
 ('05','Random Forest','random_forest','20260926T171032572709Z'),
 ('06','Extra Trees','extra_trees','20260926T191430559698Z'),
 ('07','MLP','mlp','20260926T204853996228Z'),
 ('08','CNN','cnn','20260926T223539611207Z'),
 ('09','ResNet','resnet','20260927T140640055671Z'),
 ('10','U-Net','unet','20260927T185613221209Z'),
 ('11','ResUNet','resunet','20260928T112749380518Z'),
 ('12','Inception','inception','20260928T200135452961Z')]
ARMS = ['originais','originais_filtradas','gpa','grafos','todas','todas_filtradas','originais_gpa','originais_grafos']
LABELS = ['Originais','Orig. filtradas','GPA','Grafos','Todas','Todas filtradas','Orig. + GPA','Orig. + grafos']
MODELS = [s[1] for s in SPECS]
PAIRS = ['H0 2020→2020','H0 2021→2021','H0 2022→2022','H1 2021→2022','H1 2020→2021']
sources={}
def read(path):
    raw=path.read_bytes(); sources[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))
def csvout(df,name):
    df.to_csv(OUT/'tabelas'/f'{name}.csv',index=False,encoding='utf-8-sig')
def number(x,digits=4):
    if x is None or (isinstance(x,float) and np.isnan(x)): return '—'
    return f'{x:.{digits}f}'.replace('.',',') if isinstance(x,(float,np.floating)) else str(x)
def table(df):
    rows=['| '+' | '.join(map(str,df.columns))+' |','|'+'|'.join(['---']*len(df.columns))+'|']
    rows += ['| '+' | '.join(number(v).replace('|','/') for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)
def savefig(name):
    plt.savefig(OUT/'figuras'/f'{name}.png',dpi=450,bbox_inches='tight')
    plt.savefig(OUT/'figuras'/f'{name}.svg',bbox_inches='tight')
    plt.close()
def fig(name,caption): return f'![{caption}](figuras/{name}.png)\n\n{caption}'

def collect():
    rows=[]; external=[]; inventory=[]; channels=[]; configs=[]
    for exp,name,key,stamp in SPECS:
        p=ROOT/'runs'/('enriched_'+key)/stamp
        state=read(p/'run.json'); assert state['status']=='completed'
        for source in list(p.glob('*.md'))+[p/'config.yaml']:
            raw=source.read_bytes();sources[str(source.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
        duration=(datetime.fromisoformat(state['finished'])-datetime.fromisoformat(state['started'])).total_seconds()/60
        inventory.append(dict(experimento=exp,modelo=name,run=str(p.relative_to(ROOT)),minutos=duration))
        for h in (0,1):
            for arm in ARMS:
                f=p/f'horizon_{h}'/arm/'metrics.json'
                if not f.exists(): continue
                m=read(f); r=dict(experimento=exp,modelo=name,horizonte=h,conjunto=arm,canais=m['n_features'],vencedor=state['winners'][str(h)]==arm,fonte=str(f.relative_to(ROOT)))
                for stage,field in [('treino','training_metrics'),('validacao','validation_metrics'),('selecao','initial_selection_metrics')]:
                    for metric,val in m.get(field,{}).items():
                        if not isinstance(val,(list,dict)): r[f'{stage}_{metric}']=val
                r['epocas']=m.get('selected_epoch');r['arvores_iteracoes']=m.get('n_estimators',m.get('actual_iterations'))
                r['lambda']=m.get('lambda_per_sample');r['minutos_run']=duration
                r['nomes_canais']=';'.join(m['feature_names'])
                r['parametros']=json.dumps(m.get('hyperparameters',m.get('estimator_parameters',{})),ensure_ascii=False)
                rows.append(r)
                for channel in m['feature_names']:
                    channels.append(dict(modelo=name,horizonte=h,conjunto=arm,vencedor=r['vencedor'],canal=channel))
        files=list(p.glob('horizon_*/external_target_*_metrics.json'))+list(p.glob('supplemental_H1_2021/external_target_*_metrics.json'))
        if exp in ['01','02','03','04']:
            files += [ROOT/'reports/target_2021_frozen_models/20260926T162434415225Z'/('enriched_'+key)/'external_target_2021_metrics.json']
        assert len(files)==5,(name,len(files))
        for f in files:
            external.append(external_row(f,name,exp))
        configs.append(dict(modelo=name,config=str((p/'config.yaml').relative_to(ROOT))))
    return pd.DataFrame(rows),pd.DataFrame(external),pd.DataFrame(inventory),pd.DataFrame(channels),configs

def external_row(f,name,exp):
    m=read(f);h=m['horizon']; pair=f"H{h} {m['input_year']}→{m['target_year']}"
    if 'n_features' not in m:
        spec=next((s for s in SPECS if s[1]==name),None)
        run=(ROOT/'runs'/('enriched_'+spec[2])/spec[3]) if spec else f.parents[1]
        m['n_features']=read(run/f'horizon_{h}'/('winner.json' if spec else 'metrics.json'))['n_features']
    r=dict(experimento=exp,modelo=name,horizonte=h,par=pair,canais=m['n_features'],pixels=m['common_pixels'],target_sha256=m['target_sha256'],fonte=str(f.relative_to(ROOT)),prediction=m['prediction'])
    for key,val in m['metrics'].items():
        if not isinstance(val,(dict,list)): r[key]=val
    for k,field in [('zero','reference_zero'),('media','reference_training_mean')]: r[k+'_rmse']=m[field]['rmse']
    per=m.get('reference_persistence');r['persistencia_rmse']=per['metrics']['rmse'] if per else None
    r['persistencia_captura']=per['metrics'].get('capture_10pct') if per else None
    for sub,values in m['metrics'].get('subsets',{}).items():
        if values:
            for k in ('rmse','mae','bias','n'): r['subset_'+sub+'_'+k]=values.get(k)
    r['ci_low'],r['ci_high']=m['uncertainty']['rmse_ci95']
    r['skill_zero']=1-(r['rmse']/r['zero_rmse'])**2
    r['skill_media']=1-(r['rmse']/r['media_rmse'])**2
    r['ratio_zero']=r['rmse']/r['zero_rmse']
    r['positivos_pct']=100*(1-r['target_zero']/r['pixels'])
    return r

def control_data():
    rows=[]; ext=[]
    for exp,name,key,stamp in [('08A','CNN local','cnn_local_control','20260927T010521698427Z'),('09A','ResNet local','resnet_local_control','20260927T181122308509Z')]:
        p=ROOT/'runs'/('enriched_'+key)/stamp
        for h in (0,1):
            m=read(p/f'horizon_{h}'/'metrics.json'); rows.append(dict(experimento=exp,modelo=name,horizonte=h,rmse=m['validation_metrics']['rmse'],referencia_rmse=m['reference_validation_metrics']['rmse']))
        for f in list(p.glob('horizon_*/external_target_*_metrics.json'))+list(p.glob('supplemental_H1_2021/external_target_*_metrics.json')):
            ext.append(external_row(f,name,exp))
    p=ROOT/'runs/enriched_unet_controls/20260928T005339448925Z'
    unet=[]
    for f in p.glob('jobs/*/metrics.json'):
        m=read(f);unet.append(dict(id=m['id'],experimento=m['experiment'],horizonte=m['horizon'],variante=m['variant'],seed=m['seed'],rmse=m['validation_metrics']['rmse'],mae=m['validation_metrics']['mae'],capture=m['validation_metrics']['capture_10pct'],centros=m['actual_training_centers'],epocas=m['epochs']))
    ue=[]
    for f in p.glob('jobs/**/external_target_*_metrics.json'):
        m=read(f);ue.append(dict(job=f.relative_to(p).parts[1],horizonte=m['horizon'],input_year=m['input_year'],target_year=m['target_year'],rmse=m['metrics']['rmse'],zero_rmse=m['reference_zero']['rmse']))
    return pd.DataFrame(rows),pd.DataFrame(ext),pd.DataFrame(unet),pd.DataFrame(ue)

def charts(d,e,w,ch,c,ce,u):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    f,axes=plt.subplots(1,2,figsize=(14,7))
    for h,ax in enumerate(axes):
        sub=w[w.horizonte==h].set_index('modelo').loc[MODELS]
        ax.barh(MODELS,sub.validacao_rmse,color='#247c85'); ax.invert_yaxis()
        for y,v in enumerate(sub.validacao_rmse): ax.text(v+.08,y,f'{v:.2f}',va='center',fontsize=9)
        ax.set(xlabel='RMSE na validação interna',title=f'H{h} · vencedor de cada família',xlim=(0,sub.validacao_rmse.max()*1.15));ax.grid(axis='x',alpha=.15)
    f.tight_layout();savefig('01_validacao')
    f,axes=plt.subplots(1,2,figsize=(15,8))
    for h,ax in enumerate(axes):
        z=d[d.horizonte==h].pivot(index='modelo',columns='conjunto',values='validacao_rmse').reindex(index=MODELS,columns=ARMS)
        gains=100*(1-z.div(z['originais'],axis=0));a=gains.to_numpy()
        im=ax.imshow(a,cmap='RdBu',vmin=-100,vmax=100,aspect='auto')
        ax.set_xticks(range(8),LABELS,rotation=60,ha='right');ax.set_yticks(range(12),MODELS);ax.set_title(f'H{h} · redução de RMSE (%) versus originais')
        for (i,j),v in np.ndenumerate(a): ax.text(j,i,'—' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>50 else 'black')
    f.subplots_adjust(bottom=.25,wspace=.36,right=.88)
    cax=f.add_axes([.92,.29,.015,.52])
    f.colorbar(im,cax=cax,label='Redução de RMSE (%) · positivo = melhora')
    savefig('02_conjuntos')
    z=e.pivot(index='modelo',columns='par',values='rmse').reindex(index=MODELS,columns=PAIRS)
    ratios=e.pivot(index='modelo',columns='par',values='ratio_zero').reindex(index=MODELS,columns=PAIRS)
    f,ax=plt.subplots(figsize=(11,8));norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=4)
    im=ax.imshow(np.log2(ratios),cmap='RdBu_r',norm=norm,aspect='auto')
    ax.set_xticks(range(5),PAIRS,rotation=20,ha='right');ax.set_yticks(range(12),MODELS)
    for (i,j),v in np.ndenumerate(z.to_numpy()):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if norm(np.log2(ratios.iloc[i,j]))>.8 else 'black')
    ax.set_title('Área externa · números = RMSE; cor = razão em relação a prever zero')
    cb=f.colorbar(im,ax=ax,label='RMSE do modelo / RMSE zero',ticks=[-1,0,1,2,3,4]);cb.ax.set_yticklabels(['0,5×','1×','2×','4×','8×','16×'])
    f.tight_layout();savefig('03_externo')
    f,axes=plt.subplots(1,2,figsize=(14,6))
    for h,ax in enumerate(axes):
        sub=e[(e.horizonte==h)&(e.par.str.endswith('2022'))].set_index('modelo').loc[MODELS]
        ax.scatter(sub.rmse,sub.capture_10pct*100,c=range(12),cmap='tab20',s=65)
        offsets={(0,'CNN'):(5,-12),(0,'XGBoost'):(-8,8),(1,'Linear'):(5,-12),(1,'Ridge'):(5,8)}
        for name,r in sub.iterrows(): ax.annotate(name,(r.rmse,r.capture_10pct*100),xytext=offsets.get((h,name),(4,4)),textcoords='offset points',fontsize=8)
        ax.axvline(sub.zero_rmse.iloc[0],ls=':',color='gray',label='RMSE zero')
        ax.scatter(sub.persistencia_rmse.iloc[0],sub.persistencia_captura.iloc[0]*100,marker='*',s=180,c='black',label='Persistência')
        ax.axhline(10,color='gray',lw=.7,ls='--',label='10%: expectativa seleção aleatória')
        ax.set(xlabel='RMSE externo (menor é melhor)',ylabel='Massa do target capturada no top 10% (%)',title=f'H{h} · target externo 2022');ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper right')
    f.tight_layout();savefig('04_intensidade_priorizacao')
    f,axes=plt.subplots(1,2,figsize=(15,7))
    for h,ax in enumerate(axes):
        sub=e[(e.horizonte==h)&(e.par.str.endswith('2021'))].set_index('modelo').loc[MODELS]
        ys=np.arange(12);ax.barh(ys-.18,sub.subset_zero_mae,height=.35,label='Pixels target zero',color='#d28642');ax.barh(ys+.18,sub.subset_positive_mae,height=.35,label='Pixels target positivo',color='#247c85')
        ax.set_yticks(ys,MODELS);ax.invert_yaxis();ax.set(xlabel='MAE externo',title=f'H{h} · target 2021, cerca de 50% positivos');ax.legend(fontsize=8)
    f.tight_layout();savefig('05_zeros_positivos')
    features=list(dict.fromkeys(ch.canal))
    # Fixed scientific order, not selected by observed performance.
    features=[p+'__'+v for p in ['Garimpos','Ocorrencias_Minerais','Pistas_Regulares','Pistas_Irregulares'] for v in ['original','magnitude','direcao_seno','direcao_cosseno','conectividade','suporte']]
    f,axes=plt.subplots(1,2,figsize=(16,9))
    for h,ax in enumerate(axes):
        a=np.array([[int(((ch.modelo==m)&(ch.horizonte==h)&ch.vencedor&(ch.canal==feat)).any()) for feat in features] for m in MODELS])
        ax.imshow(a,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(12),MODELS)
        short=[f.split('__')[0].replace('Garimpos','G').replace('Ocorrencias_Minerais','M').replace('Pistas_Regulares','R').replace('Pistas_Irregulares','I')+' '+f.split('__')[1].replace('direcao_','').replace('conectividade','conect.').replace('magnitude','mag.').replace('original','orig.') for f in features]
        ax.set_xticks(range(24),short,rotation=90,fontsize=8);ax.set_title(f'H{h} · canais do vencedor (azul = presente)')
        for cut in [5.5,11.5,17.5]: ax.axvline(cut,color='gray',lw=.7)
    f.tight_layout();savefig('06_canais')
    f,axes=plt.subplots(1,2,figsize=(14,6))
    for h,ax in enumerate(axes):
        sub=u[u.horizonte==h];groups=list(sub.groupby(['experimento','variante'],sort=True))
        for i,((exp,variant),g) in enumerate(groups):
            ax.scatter(np.full(len(g),i),g.rmse,color='#247c85',s=42)
            ax.errorbar(i,g.rmse.mean(),yerr=g.rmse.std(ddof=1),color='#b46a28',fmt='D',capsize=6)
        translations={'fixed':'fixo','rotating':'rotativo','depth3':'três níveis','direction':'direção','magnitude':'magnitude'}
        ax.set_xticks(range(len(groups)),[f'{k[0]}\n{translations[k[1]]}' for k,g in groups],rotation=25,ha='right');ax.set(title=f'U-Net · H{h}',ylabel='RMSE de validação');ax.grid(axis='y',alpha=.2)
    f.tight_layout();savefig('07_controles_unet')
    f,ax=plt.subplots(figsize=(10,5));inv=w.groupby('modelo').minutos_run.first().reindex(MODELS)
    ax.bar(MODELS,inv,color='#247c85');ax.set_yscale('log');ax.set(ylabel='Minutos de execução completa (escala log)',title='Custo observado · inclui busca, reajustes, inferência e relatório');ax.tick_params(axis='x',rotation=45);f.tight_layout();savefig('08_custo')
    f,axes=plt.subplots(1,2,figsize=(15,7))
    for h,ax in enumerate(axes):
        sub=e[(e.horizonte==h)&e.par.str.endswith('2022')].set_index('modelo').loc[MODELS]
        ax.errorbar(sub.rmse,np.arange(12),xerr=[sub.rmse-sub.ci_low,sub.ci_high-sub.rmse],fmt='o',color='#247c85',capsize=3)
        ax.axvline(sub.zero_rmse.iloc[0],color='#b46a28',ls='--',label='Zero');ax.axvline(sub.persistencia_rmse.iloc[0],color='black',ls=':',label='Persistência')
        ax.set_yticks(range(12),MODELS);ax.invert_yaxis();ax.set(title=f'H{h} · target 2022',xlabel='RMSE e intervalo bootstrap espacial exploratório de 95%');ax.legend(fontsize=8)
    f.tight_layout();savefig('09_incerteza')

def maps(e):
    import rasterio
    from rasterio.enums import Resampling
    for h in (0,1):
        sub=e[(e.horizonte==h)&e.par.str.endswith('2021')].set_index('modelo').loc[MODELS]
        target=ROOT/'data/regiao_teste_final/2021/target.tif'
        paths=[target]+[Path(p) for p in sub.prediction]
        arrays=[]
        for p in paths:
            with rasterio.open(p) as src:
                shape=(round(src.height*min(1,500/max(src.shape))),round(src.width*min(1,500/max(src.shape))))
                arrays.append(src.read(1,out_shape=shape,masked=True,resampling=Resampling.nearest))
        mask=np.logical_or.reduce([np.ma.getmaskarray(a)|~np.isfinite(a.filled(np.nan)) for a in arrays])
        arrays=[np.ma.array(a,mask=mask) for a in arrays]
        vmax=max(float(np.max(np.abs(a.compressed()))) for a in arrays)
        f,axes=plt.subplots(4,4,figsize=(15,15),layout='constrained')
        titles=['Target 2021']+[f'{name} · RMSE {sub.loc[name,"rmse"]:.2f}' for name in MODELS]
        norm=SymLogNorm(linthresh=.1,vmin=-vmax,vmax=vmax,base=10)
        for ax,arr,title in zip(axes.flat,arrays,titles):
            im=ax.imshow(arr,cmap='RdBu_r',norm=norm);ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        for ax in list(axes.flat)[13:]:ax.set_visible(False)
        f.colorbar(im,ax=list(axes.flat)[:13],shrink=.55,label='Intensidade original · escala simétrica/logarítmica compartilhada')
        f.suptitle(f'H{h} · área externa, target 2021\nPrévia por vizinho mais próximo, lado maior ≤500 pixels; métricas na resolução nativa',fontsize=15)
        savefig(f'{10+h}_mapas_h{h}')

def render_markdown(text):
    # Small offline renderer for this report's paragraphs, tables, links and figures.
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',r'<img loading="lazy" alt="\1" src="\2">',s)
        s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
        s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
        s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
        return s
    body=[];toc=[];lines=text.splitlines();i=0
    while i<len(lines):
        l=lines[i]
        if not l.strip():i+=1;continue
        if l.startswith('#'):
            n=len(l)-len(l.lstrip('#'));title=l[n:].strip();id_=f'section-{len(toc)}'
            body.append(f'<h{n} id="{id_}">{inline(title)}</h{n}>');toc.append((n,title,id_));i+=1
        elif l.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[-: ]+',x) for x in cells):block.append(cells)
                i+=1
            body.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(v)+'</th>' for v in block[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(v)+'</td>' for v in row)+'</tr>' for row in block[1:])+'</tbody></table></div>')
        elif l.startswith('- '):
            body.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):body.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            body.append('</ul>')
        else:
            part=[]
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ')):part.append(lines[i]);i+=1
            body.append('<p>'+inline(' '.join(part))+'</p>')
    nav=''.join(f'<a class="l{n}" href="#{id_}">{html.escape(t)}</a>' for n,t,id_ in toc if n==2)
    return '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Interpretação geral dos experimentos</title><style>body{margin:0;background:#f2f5f4;color:#20333b;font:17px/1.7 Georgia,serif}main{max-width:1160px;margin:auto;background:white;padding:55px 65px}h1,h2,h3,nav,th{font-family:Arial,sans-serif}h1{font-size:42px;line-height:1.2;color:#12545f}h2{margin-top:65px;border-top:2px solid #d7e5e5;padding-top:22px;font-size:27px}h3{font-size:21px;margin-top:35px}a{color:#146e80}img{max-width:100%;height:auto;border:1px solid #e2e6e6}table{border-collapse:collapse;width:100%;font:13px/1.5 Arial,sans-serif}th,td{padding:9px 11px;border-bottom:1px solid #d8e1e1;text-align:left}th{background:#164f5a;color:white}tr:nth-child(even){background:#f1f6f6}.table-wrap{overflow-x:auto;margin:24px 0}nav{background:#eaf2f2;padding:20px;font-size:14px;columns:2}nav a{display:block;padding:3px}code{font-size:.85em;overflow-wrap:anywhere}@media(max-width:800px){main{padding:25px}nav{columns:1}h1{font-size:31px}}@media print{main{padding:0}nav{display:none}h2{break-before:page}img,table{break-inside:avoid}a{color:inherit}}</style><main><nav>'+nav+'</nav>'+''.join(body)+'</main></html>'

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (OUT/'tabelas').mkdir(parents=True,exist_ok=True);(OUT/'figuras').mkdir(exist_ok=True)
    d,e,inv,ch,configs=collect();w=d[d.vencedor].copy();c,ce,u,ue=control_data()
    assert len(d)==148 and len(e)==60 and len(w)==24
    assert len(c)==4 and len(ce)==10 and len(u)==24 and len(ue)==57
    assert e.groupby('par').target_sha256.nunique().max()==1 and e.pixels.nunique()==1
    for _,group in e.groupby('par'):
        for field in ['zero_rmse','media_rmse','persistencia_rmse']:
            values=group[field].dropna().to_numpy()
            if len(values): assert np.allclose(values,values[0],rtol=1e-10,atol=1e-10),(field,values)
    assert not e.duplicated(['modelo','par']).any()
    for name,df in [('todos_148_ajustes',d),('externo_60_pares',e),('vencedores_24',w),('inventario_execucoes',inv),('presenca_canais',ch),('controles_locais_validacao',c),('controles_locais_externo',ce),('controles_unet_24',u),('controles_unet_externo_57',ue)]:csvout(df,name)
    if '--text-only' not in sys.argv:
        charts(d,e,w,ch,c,ce,u)
        maps(e)
    print('WINNERS\n'+w[['modelo','horizonte','conjunto','canais','validacao_rmse','validacao_r2','epocas','arvores_iteracoes']].to_string(index=False))
    print('EXTERNAL\n'+e.pivot(index='modelo',columns='par',values='rmse').reindex(MODELS).to_string())
    print('CAPTURE\n'+e.pivot(index='modelo',columns='par',values='capture_10pct').reindex(MODELS).to_string())
    print('BASELINES counts',int((e.rmse<e.zero_rmse).sum()),int((e.rmse<e.media_rmse).sum()),int((e.rmse<e.persistencia_rmse).sum()),int((e.r2>0).sum()))
    print('CONTROLS',u.groupby(['experimento','horizonte','variante']).rmse.agg(['mean','std']).to_string())
    template=ROOT/'protocols/INTERPRETACAO_GERAL_TEXTO.md'
    if template.exists():
        text=template.read_text(encoding='utf-8')
        tokens=report_tokens(d,e,w,inv,ch,c,ce,u)
        for key,value in tokens.items():text=text.replace('{{'+key+'}}',value)
        assert '{{' not in text
        (OUT/'RELATORIO_FINAL.md').write_text(text,encoding='utf-8')
        (OUT/'RELATORIO_FINAL.html').write_text(render_markdown(text),encoding='utf-8')
    (OUT/'fontes_metricas_sha256.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False),encoding='utf-8')
    if template.exists():
        from PIL import Image
        local_links=re.findall(r'!?\[[^]]*\]\(([^)]+)\)',text)
        broken=[url for url in local_links if not url.startswith(('http:','https:','#')) and not (OUT/url).exists()]
        assert not broken,broken
        sizes={p.name:Image.open(p).size for p in (OUT/'figuras').glob('*.png')}
        assert len(sizes)==11
        assert int((e.rmse<e.zero_rmse).sum())==3
        assert int((e.rmse<e.media_rmse).sum())==1
        assert int((e.r2>0).sum())==0
        for (_,h),g in d.groupby(['modelo','horizonte']):
            assert len(g[g.vencedor])==1
            assert np.isclose(g[g.vencedor].validacao_rmse.iloc[0],g.validacao_rmse.min(),atol=1e-10)
        audit={'date':'2026-09-29','main_models':12,'main_final_fits':len(d),'main_external_pairs':len(e),
               'local_controls':len(c),'unet_controls':len(u),'all_external_pairs':len(e)+len(ce)+len(ue),
               'unique_target_hashes':e.groupby('par').target_sha256.first().to_dict(),
               'external_common_pixels':int(e.pixels.iloc[0]),'baseline_tolerance':1e-10,
               'main_wins_vs_zero':int((e.rmse<e.zero_rmse).sum()),'main_wins_vs_training_mean':int((e.rmse<e.media_rmse).sum()),
               'main_wins_vs_persistence':int((e.rmse<e.persistencia_rmse).sum()),'main_positive_r2':int((e.r2>0).sum()),
               'broken_local_links':broken,'figure_dimensions':sizes,'png_dpi':450,'vector_format':'SVG',
               'native_predictions_recomputed':False,'new_training':False,
               'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'text_source_sha256':hashlib.sha256(template.read_bytes()).hexdigest()}
        (OUT/'AUDITORIA.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print('OUTPUT',OUT)

def report_tokens(d,e,w,inv,ch,c,ce,u):
    t={}
    for h in (0,1):
        z=d[d.horizonte==h].pivot(index='modelo',columns='conjunto',values='validacao_rmse').reindex(index=MODELS,columns=ARMS)
        z.columns=LABELS;t[f'ARMS_H{h}']=table(z.reset_index().rename(columns={'modelo':'Modelo'}))
    t['WINNERS']=table(w[['modelo','horizonte','conjunto','canais','validacao_rmse','validacao_mae','validacao_r2']].rename(columns={'modelo':'Modelo','horizonte':'H','conjunto':'Conjunto','canais':'Canais','validacao_rmse':'RMSE','validacao_mae':'MAE','validacao_r2':'R²'}))
    for metric,token in [('rmse','EXTERNAL'),('mae','EXTERNAL_MAE'),('capture_10pct','CAPTURE'),('bias','BIAS')]:
        z=e.pivot(index='modelo',columns='par',values=metric).reindex(index=MODELS,columns=PAIRS)
        if metric=='capture_10pct':z*=100
        t[token]=table(z.reset_index().rename(columns={'modelo':'Modelo'}))
    t['INVENTORY']=table(inv[['experimento','modelo','minutos','run']].rename(columns={'experimento':'Exp.','modelo':'Modelo','minutos':'Minutos','run':'Execução'}))
    t['LOCAL']=table(c.rename(columns={'experimento':'Exp.','modelo':'Modelo','horizonte':'H','rmse':'RMSE local','referencia_rmse':'RMSE duas escalas'}))
    us=u.groupby(['experimento','horizonte','variante']).rmse.agg(['mean','std']).reset_index();t['UNET_CONTROLS']=table(us.rename(columns={'experimento':'Controle','horizonte':'H','variante':'Condição','mean':'Média RMSE','std':'DP entre 3 sementes'}))
    t['WINNER_CHANNELS']=table(w[['modelo','horizonte','nomes_canais']].rename(columns={'modelo':'Modelo','horizonte':'H','nomes_canais':'Canais exatos'}))
    t['PARAMS']=table(w[['modelo','horizonte','epocas','arvores_iteracoes','lambda','parametros']].rename(columns={'modelo':'Modelo','horizonte':'H','epocas':'Épocas','arvores_iteracoes':'Árvores/iterações','lambda':'λ/amostra','parametros':'Hiperparâmetros registrados'}))
    gain=[]
    for (name,h),g in d.groupby(['modelo','horizonte'],sort=False):
        s=g.set_index('conjunto').validacao_rmse;gain.append(dict(Modelo=name,H=h,GPA=100*(1-s['gpa']/s['originais']),Grafos=100*(1-s['grafos']/s['originais']),Todas=100*(1-s['todas']/s['originais'])))
    gd=pd.DataFrame(gain);csvout(gd,'ganhos_vs_originais');t['GAINS']=table(gd)
    target=e[e.modelo=='Linear'].drop_duplicates('par')[['par','pixels','positivos_pct','zero_rmse','media_rmse','persistencia_rmse']]
    t['TARGET']=table(target.rename(columns={'par':'Par','pixels':'Pixels','positivos_pct':'Positivos %','zero_rmse':'RMSE zero','media_rmse':'RMSE média treino','persistencia_rmse':'RMSE persistência'}))
    t['SOURCES']='\n'.join(f'- Experimento {exp}, {name}: [relatório](../../runs/enriched_{key}/{stamp}/RELATORIO.md).' for exp,name,key,stamp in SPECS)
    t['MAPS']=fig('10_mapas_h0','Figura 10. Comparação H0 para target 2021: as cores têm a mesma escala entre todos os modelos desta figura.')+'\n\n'+fig('11_mapas_h1','Figura 11. Comparação H1 complementar para target 2021. Escala comum dentro desta figura; não comparar cores diretamente com a Figura 10.')
    captions={
      '01_validacao':'Figura 1. RMSE dos vencedores internos; H0 e H1 são problemas distintos.',
      '02_conjuntos':'Figura 2. Ganho relativo de cada conjunto contra originais. Positivo indica redução de RMSE; células ausentes não foram treinadas.',
      '03_externo':'Figura 3. Todos os 60 testes externos principais. A cor compara cada erro com a referência zero do mesmo target; os números são RMSE nas unidades originais.',
      '04_intensidade_priorizacao':'Figura 4. Intensidade e priorização respondem a perguntas diferentes. O ponto ideal tem erro baixo e captura alta; persistência dispõe de target histórico.',
      '05_zeros_positivos':'Figura 5. MAE nos pixels sem registro e com registro em 2021. A divisão não transforma zero em ausência real do fenômeno.',
      '06_canais':'Figura 6. Presença de canais nos vencedores; não é importância causal. G: garimpos; M: minerais; R: pistas regulares; I: pistas irregulares.',
      '07_controles_unet':'Figura 7. Controles U-Net: pontos são sementes individuais; losango e barras mostram média ± desvio padrão, não intervalo de confiança.',
      '08_custo':'Figura 8. Duração observada das execuções. Busca, hardware, orçamento e número de modelos diferem; não representa velocidade intrínseca de cada algoritmo.',
      '09_incerteza':'Figura 9. Intervalos bootstrap espacial já registrados, com 42 blocos de aproximadamente 96 km. Não são testes pareados de superioridade entre modelos.'}
    for key,caption in captions.items():t['FIG_'+key[:2]]=fig(key,caption)
    return t

if __name__=='__main__': main()
