from pathlib import Path
import csv, html, json, re, zipfile, hashlib

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'latex_tables'
OUT.mkdir(exist_ok=True)
def read(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def esc(s):
    text=str(s)
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}','→':r'$\rightarrow$','–':'--','±':r'$\pm$','²':r'$^2$'}.get(c,c) for c in text)
def num(v,d=4,pct=False,integer=False):
    if v in ('',None,'NA','nan'):return r'\textemdash{}'
    x=float(v)*(100 if pct else 1)
    if integer:return f'{int(round(x)):,}'.replace(',',r'\,')
    if abs(x)<0.5*10**-d:x=0
    return f'{x:.{d}f}'
arms={'Original':'O','Selected original':r'O$_s$','Gradient':'D','Graph':'G','All':'A','Selected all':r'A$_s$','Original + gradient':'O+D','Original + graph':'O+G'}
models={'Linear':'LIN','Ridge':'RID','HGB':'HGB','XGBoost':'XGB','Random Forest':'RF','Extra Trees':'ET','MLP':'MLP','CNN':'CNN','Compact ResNet':'RES','U-Net':'UN','ResUNet':'RUN','Dense Inception':'INC','CNN local':'CNN-L','ResNet local':'RES-L'}
modelnote='Model codes: LIN, linear regression; RID, Ridge; HGB, histogram-based gradient boosting; XGB, XGBoost; RF, Random Forest; ET, Extra Trees; MLP, multilayer perceptron; CNN, two-scale CNN; RES, Compact ResNet; UN, U-Net; RUN, ResUNet; INC, Dense Inception. CNN-L and RES-L denote the local-only controls.'
armnote=r'Input sets: O, original; O$_s$, selected original; D, gradient; G, graph; A, all; A$_s$, selected all; O+D and O+G, addition controls. H0 and H1 denote temporal horizons, not statistical hypotheses.'
metricnote=r'RMSE, root-mean-square error; MAE, mean absolute error; bias = prediction minus target; P90, 90th percentile of absolute error; NMAD, normalized median absolute deviation. C5, C10 and C20 are percentages of target mass captured in the top 5\%, 10\% and 20\% of valid cells. A dash indicates unavailable or inapplicable data, not zero.'

PREAMBLE=r'''% Add once to the preamble of the supplied MDPI template.
% The MDPI class already loads booktabs, tabularx, array, float and changepage.
\usepackage{longtable}
\newlength{\rasterTableBodyWidth}
% No external data files, resizing commands or extra fonts are required.
'''
(OUT/'00_PREAMBULO.tex').write_text(PREAMBLE,encoding='utf-8')
blocks={};counts={}
def table(key,caption,label,headers,rows,weights,align=None,note='',long=False,supp=None,panel=None):
    assert all(len(r)==len(headers) for r in rows),key
    align=align or ['l']+['r']*(len(headers)-1)
    total=sum(weights);weights=[w/total for w in weights]
    cols=[]
    for w,a in zip(weights,align):
        command={'l':r'\raggedright','r':r'\raggedleft','c':r'\centering'}[a]
        cols.append('>{'+command+r'\arraybackslash}p{'+f'{w:.6f}'+r'\rasterTableBodyWidth}')
    col=''.join(cols)
    prefix=[r'\begingroup',r'\small',r'\renewcommand{\arraystretch}{1.16}',r'\setlength{\tabcolsep}{4pt}',
            r'\setlength{\rasterTableBodyWidth}{\dimexpr\fulllength-'+str(2*len(headers))+r'\tabcolsep\relax}']
    if supp:prefix.append(r'\renewcommand{\thetable}{'+supp+'}')
    head=' & '.join(r'\textbf{'+h+'}' for h in headers)+r' \\'
    caption=esc(caption)
    if long:
        lines=prefix+[r'\setlength{\LTleft}{-\extralength}',r'\setlength{\LTright}{0pt}',r'\setlength{\LTcapwidth}{\fulllength}',
          r'\begin{longtable}{'+col+'}',r'\caption{'+caption+r'\label{'+label+r'}}\\',r'\toprule',head,r'\midrule',r'\endfirsthead',
          r'\caption[]{'+caption+r' (continued).}\\',r'\toprule',head,r'\midrule',r'\endhead',
          r'\midrule',r'\multicolumn{'+str(len(headers))+r'}{r}{\footnotesize Continued on the next page.}\\',r'\endfoot',r'\bottomrule']
        if note:lines.append(r'\multicolumn{'+str(len(headers))+r'}{@{}p{\fulllength}@{}}{\footnotesize '+note+r'}\\')
        lines.append(r'\endlastfoot')
    else:
        lines=prefix+[r'\begin{table}[H]',r'\caption{'+caption+r'\label{'+label+'}}',r'\begin{adjustwidth}{-\extralength}{0cm}',
          r'\begin{tabular}{'+col+'}',r'\toprule',head,r'\midrule']
    for row in rows:lines.append(' & '.join(row)+r' \\')
    if long:lines+=[r'\end{longtable}']
    else:
        lines+=[r'\bottomrule',r'\end{tabular}']
        if note:lines+=[r'\par\vspace{3pt}',r'\noindent{\footnotesize '+note+'}']
        lines+=[r'\end{adjustwidth}',r'\end{table}']
    lines+=[r'\endgroup','']
    text='\n'.join(lines)
    blocks.setdefault(key,[]).append(text);counts.setdefault(key,[]).append(len(rows))

t1=read('tables/Table_1_input_sets.csv')
table('01_Table_1','Input representations evaluated in the raster experiments.','tab:raster_inputs',
      ['Input set','Contents','Channels','Model families'],[[esc(r[c]) for c in ['Input set','Contents','Channels','Model families']] for r in t1],
      [2.1,4.5,1,1.7],['l','l','c','l'],note=armnote+' Direction sine/cosine and graph connectivity/support were retained as pairs during channel selection.')
t2=read('tables/Table_2_internal_winners.csv')
table('02_Table_2','Internally selected configurations and validation performance.','tab:raster_internal',
      ['Model','Horizon','Input set','Channels','RMSE','MAE',r'$R^2$'],
      [[esc(r['Model']),'H'+r['Horizon'],arms[r['Input set']],r['Channels'],num(r['Validation RMSE']),num(r['Validation MAE']),num(r['Validation R2'])] for r in t2],
      [2.4,.85,1.05,.85,1.15,1.15,1.15],['l','c','c','r','r','r','r'],note=armnote+' H0 and H1 use different validation samples. Each row is one frozen winner, not an average across seeds.')
t3=read('tables/Table_3_external_targets.csv')
table('03_Table_3','External target support and reference-predictor RMSE.','tab:raster_external_support',
      ['Input--target pair','Valid cells',r'Positive cells (\%)','Zero RMSE','Persistence RMSE'],
      [[esc(r['Evaluation pair']),num(r['Common pixels'],integer=True),num(r['Positive target pixels (%)']),num(r['Zero RMSE']),num(r['Persistence RMSE'])] for r in t3],
      [2.4,1.6,1.5,1.5,1.7],note='Persistence uses additional historical target information. A dash indicates that the earlier target was unavailable. Model-specific training-mean errors are reported in Supplementary Table S2.')
for key,file,caption,label,supp in [('04_Table_4','Table_4_external_RMSE','External RMSE of internally selected frozen models.','tab:raster_external_rmse',None),
 ('10_Table_S6','Table_S6_external_capture','External target mass captured in the highest-scoring 10% of valid cells (%).','tab:raster_s6','S6')]:
    data=read('tables/'+file+'.csv');columns=list(data[0])[1:]
    headers=['Model']+[r'\shortstack{'+c[:2]+r'\\'+esc(c[3:])+'}' for c in columns]
    table(key,caption,label,headers,[[esc(r['Model'])]+[num(r[c]) for c in columns] for r in data],
          [2.4]+[1.45]*5,note='H1 2020--2021 is the complementary external pair. All comparisons use 5,868,615 common valid cells. '+('Lower RMSE is better.' if not supp else 'Capture is a ranking metric, not classification accuracy; higher values indicate more observed target mass in the selected cells.'),supp=supp)

# S1: all 148 refits, full quantitative metrics for all three recorded stages.
s1=read('supplement/S1_all_148_final_fits.csv')
stages=[('Validation','Validation RMSE','Validation MAE','Validation R2'),('Training','Training rmse','Training mae','Training r2'),('Selection','Selection rmse','Selection mae','Selection r2')]
for stage,rmse,mae,r2 in stages:
    identity=lambda r:[models[r['Model']],'H'+r['Horizon'],arms[r['Input set']]]
    rows=[identity(r)+[r['Channels'],num(r[stage+' n'],integer=True),num(r[rmse]),num(r[mae]),num(r[r2]),num(r[stage+' bias'])] for r in s1]
    table('05_Table_S1',f'All 148 final configurations: {stage.lower()} sample size and regression errors.',f'tab:raster_s1_{stage.lower()}_errors',
      ['Model','Horizon','Set','Channels',r'$n$','RMSE','MAE',r'$R^2$','Bias'],rows,[.8,.8,.7,.8,1.15,1.05,1.05,1.05,1.05],['l','c','c']+['r']*6,
      note=esc(modelnote)+' '+armnote+' Selection metrics refer to the earlier fitting/selection stage; training metrics refer to the final refit. They are not an epoch curve.',long=True,supp='S1')
    rows=[identity(r)+[num(r[stage+' '+k],pct=k.startswith('capture')) for k in ['median_error','p90_absolute_error','nmad','capture_5pct','capture_10pct','capture_20pct']]+[num(r[stage+' negative_predictions'],integer=True),num(r[stage+' target_zero'],integer=True)] for r in s1]
    table('05_Table_S1',f'All 148 final configurations: {stage.lower()} residual diagnostics and spatial capture.',f'tab:raster_s1_{stage.lower()}_diagnostics',
      ['Model','Horizon','Set','Median error','P90','NMAD','C5','C10','C20','Negative predictions','Zero target'],rows,
      [.7,.7,.65,1.1,.85,.85,.85,.85,.85,1.2,1.2],['l','c','c']+['r']*8,note=metricnote,long=True,supp='S1')

def external(key,data,supp,label,control=False):
    ids=[]
    for r in data:
        if control:
            match=re.match(r'(10[A-C])_H(\d)_(.*?)_seed(\d+)',r['Job']);assert match,r['Job']
            ex,h,v,seed=match.groups()
            pair=f"{r['Input year']}--{r['Target year']}"
            ids.append([ex,'H'+h,esc(v),seed,pair])
        else:
            pair=r['Evaluation pair'];ids.append([models[r['Model']],pair[:2],esc(pair[3:])])
    heads=['Control','Horizon','Condition','Seed','Pair'] if control else ['Model','Horizon','Pair']
    widths=[.7,.7,1.15,.75,1.35] if control else [.8,.75,1.4]
    defs=[('errors',['rmse','mae','r2','bias','nmad','p90_absolute_error'],['RMSE','MAE',r'$R^2$','Bias','NMAD','P90']),
          ('capture',['capture_5pct','capture_10pct','capture_20pct','Zero RMSE',('Training mean RMSE' if control else 'Training-mean RMSE'),'Persistence RMSE',('Persistence capture at 10%' if control else 'Persistence capture at 10%')],['C5','C10','C20','Zero RMSE','Mean RMSE','Persistence RMSE','Persistence C10']),
          ('counts',['median_error','negative_predictions','target_zero','n'],['Median error','Negative predictions','Zero target','Valid cells'])]
    if not control:defs.append(('subsets',['subset_zero_n','subset_zero_rmse','subset_zero_mae','subset_zero_bias','subset_positive_n','subset_positive_rmse','subset_positive_mae','subset_positive_bias'],[r'$n_0$','RMSE zero cells','MAE zero cells','Bias zero cells',r'$n_+$','RMSE positive','MAE positive','Bias positive']))
    if not control:defs.append(('uncertainty',['ci_low','ci_high','skill_zero','skill_training_mean','ratio_zero','Positive target pixels (%)'],[r'RMSE lower 95\%',r'RMSE upper 95\%','Skill vs zero','Skill vs mean','RMSE ratio vs zero',r'Positive cells (\%)']))
    for panel,cols,titles in defs:
        rows=[]
        for ident,r in zip(ids,data):
            vals=[]
            for k in cols:vals.append(num(r[k],pct=k.startswith('capture_') or k=='Persistence capture at 10%',integer=k in ['negative_predictions','target_zero','n','subset_zero_n','subset_positive_n']))
            rows.append(ident+vals)
        note=metricnote+' All evaluations use 5,868,615 common cells. Mean RMSE refers to the training-mean predictor; persistence uses historical target information.'
        if panel=='uncertainty':note=r'Exploratory RMSE intervals use 1,000 spatial-block bootstrap replicates (42 occupied blocks; approximately 96 km). Skill = $1-(\mathrm{RMSE}_{model}/\mathrm{RMSE}_{reference})^2$. These are not paired intervals for model differences.'
        if panel=='subsets':note=r'$n_0$ and $n_+$ denote counts in zero-target and positive-target subsets. Zero cells indicate no recorded intensity, not verified absence of the phenomenon. Empty high-intensity subsets are not assigned zero error.'
        if not control and panel=='errors':note=esc(modelnote)+' '+note
        table(key,f'External evaluation records: {panel.replace("_"," ")} ({len(data)} model--pair evaluations).',label+'_'+panel,heads+titles,rows,widths+[1.05]*len(cols),
          ['l','c','l','c','c']+['r']*len(cols) if control else ['l','c','c']+['r']*len(cols),note=note,long=True,supp=supp)

external('06_Table_S2',read('supplement/S2_all_60_external_evaluations.csv'),'S2','tab:raster_s2')
external('07_Table_S3',read('supplement/S3_local_control_external.csv'),'S3','tab:raster_s3')
s4=read('supplement/S4_unet_24_fits.csv')
table('08_Table_S4','Individual U-Net control fits and validation results.','tab:raster_s4',
 ['Control','Horizon','Condition','Seed','Epochs','Training centres','RMSE','MAE','C10'],
 [[r['Experiment'],'H'+r['Horizon'],esc(r['Variant']),r['seed'],r['Final epochs'],num(r['Training centers'],integer=True),num(r['rmse']),num(r['mae']),num(r['Capture at 10%'],pct=True)] for r in s4],
 [.8,.8,1.3,.85,.7,1.4,1.1,1.1,1.1],['l','c','l']+['r']*6,long=True,supp='S4',note='Each row is one fitted model. Fixed and rotating refer to core exposure; depth3 denotes three downsampling stages. Direction and magnitude conditions were tested only under fixed-core H1 training. C10 is reported as a percentage.')
external('09_Table_S5',read('supplement/S5_unet_57_external.csv'),'S5','tab:raster_s5',True)
s7=read('tables/Table_S7_unet_controls.csv')
table('11_Table_S7','U-Net control validation RMSE across three random seeds.','tab:raster_s7',
 ['Control','Horizon','Condition','Seeds','Mean RMSE','Sample SD'],
 [[r['Experiment'],'H'+r['Horizon'],esc(r['Variant']),r['Seeds'],num(r['Mean validation RMSE']),num(r['Sample SD'])] for r in s7],
 [1,.9,2,.8,1.6,1.6],['l','c','l','r','r','r'],supp='S7',note='Means and sample standard deviations summarize individual-model RMSE values. They are neither ensemble metrics nor confidence intervals. All control models used eleven epochs.')
s8=read('supplement/S8_winner_channels_parameters.csv')
themes={'Mining sites':'M','Mineral occurrences':'V','Regular airstrips':'R','Irregular airstrips':'I'}
prods={'original':'O','magnitude':'M','direction sine':'S','direction cosine':'C','connectivity':'K','support':'B'}
def chans(s):return ', '.join(themes[p.split(' / ')[0]]+'-'+prods[p.split(' / ')[1]] for p in s.split('; '))
table('12_Table_S8','Channels retained by each internally selected winner.','tab:raster_s8_channels',
 ['Model','Horizon','Set','Retained channels'],[[esc(r['Model']),'H'+r['Horizon'],arms[r['Input set']],chans(r['Channel names'])] for r in s8],
 [2,.6,.7,7],['l','c','c','l'],long=True,supp='S8',note='Theme prefixes: M, mining sites; V, mineral occurrences; R, regular airstrips; I, irregular airstrips. Product suffixes: O, original; M, magnitude; S, direction sine; C, direction cosine; K, connectivity; B, support. '+armnote)
def params(s):
    d=json.loads(s)
    if not d:return 'No additional parameters recorded in this field.'
    # Break parameter names at underscores; never break numeric values.
    return '; '.join(esc(k).replace(r'\_',r'\_\allowbreak{}')+' = '+esc(json.dumps(v)) for k,v in d.items())
table('12_Table_S8','Recorded fitting budgets and hyperparameters of the 24 winners.','tab:raster_s8_parameters',
 ['Model','Horizon','Epochs','Trees / iterations',r'$\lambda$/sample','Recorded parameters'],
 [[esc(r['Model']),'H'+r['Horizon'],num(r['Final epochs'],integer=True),num(r['Trees or iterations'],integer=True),num(r['Lambda per sample']),params(r['Parameters'])] for r in s8],
 [1.6,.6,.6,1,.95,5.5],['l','c','r','r','r','l'],long=True,supp='S8',note='A dash denotes a field that is inapplicable or not recorded in this format; it does not denote zero. Full fixed settings and search grids are preserved in the per-run configuration files. Epoch counts refer to final refits, not the total search budget.')

main=[];supp=[]
for key,parts in sorted(blocks.items()):
    text='% '+key+' — paste the complete block.\n\n'+'\n'.join(parts)
    (OUT/(key+'.tex')).write_text(text,encoding='utf-8')
    (main if int(key[:2])<=4 else supp).append(text)
(OUT/'TABELAS_PRINCIPAIS.tex').write_text('\n\n'.join(main),encoding='utf-8')
(OUT/'TABELAS_SUPLEMENTARES.tex').write_text('\n\n'.join(supp),encoding='utf-8')
alltext='% PREAMBLE: copy 00_PREAMBULO.tex before begin{document}.\n\n'+'\n\n'.join(main+supp)
(OUT/'TODAS_AS_TABELAS.tex').write_text(alltext,encoding='utf-8')
preview=r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[left=5.88cm,right=1.27cm,top=2cm,bottom=2cm]{geometry}
\usepackage{booktabs,tabularx,array,float,changepage,longtable,amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\newlength{\extralength}\setlength{\extralength}{4.61cm}
\newlength{\fulllength}\setlength{\fulllength}{18.46cm}
\newlength{\rasterTableBodyWidth}
\begin{document}
\section*{Raster experiments: table proof}
This standalone proof checks tables at the supplied MDPI template's full table width. The final journal class remains in the original manuscript. Main tables and supplementary tables are separated below.
'''
preview+='\n'.join(main)+r'\clearpage\section*{Supplementary tables}'+'\n'.join(supp)+r'\end{document}'
(OUT/'PREVIA_TABELAS.tex').write_text(preview,encoding='utf-8')
assert [sum(counts[k]) for k in ['01_Table_1','02_Table_2','03_Table_3','04_Table_4']]==[8,24,5,12]
assert counts['05_Table_S1']==[148]*6
assert counts['06_Table_S2']==[60]*5
assert counts['07_Table_S3']==[10]*5
assert counts['08_Table_S4']==[24]
assert counts['09_Table_S5']==[57]*3
assert counts['10_Table_S6']==[12] and counts['11_Table_S7']==[8] and counts['12_Table_S8']==[24,24]
check={'logical_tables':12,'main_tables':4,'supplementary_tables':8,'panels':sum(len(x) for x in blocks.values()),'rows_by_panel':counts,'source_files_unchanged':True,'decimals':4}
(OUT/'VERIFICACAO.json').write_text(json.dumps(check,indent=2),encoding='utf-8')

readme='''# Tabelas LaTeX — Aerospace / MDPI

São 4 tabelas principais e 8 suplementares. Todas as linhas dos conjuntos descritos no roteiro foram preservadas. As matrizes extensas foram divididas em painéis com o mesmo número lógico (S1, S2 etc.), sem reduzir o texto a tamanho ilegível. Metadados de arquivo, caminhos e hashes permanecem nos CSVs de origem; não foram transformados em colunas de artigo.

1. Copie **00_PREAMBULO.tex** uma única vez antes de `\\begin{document}`. O único pacote adicional é `longtable`; os demais recursos já estão na classe MDPI enviada.
2. Copie os blocos de **TABELAS_PRINCIPAIS.tex** para os pontos indicados no roteiro. Alternativamente, abra os arquivos 01 a 04, um por tabela.
3. Use **TABELAS_SUPLEMENTARES.tex** no material suplementar. Os arquivos 05 a 12 também podem ser copiados individualmente. A designação S1–S8 já está definida em cada bloco.
4. **TODAS_AS_TABELAS.tex** reúne os 12 blocos, sem preâmbulo nem ambiente de documento. Não é necessário usar `\\input` nem enviar CSVs para compilar as tabelas.

O padrão segue os exemplos do template recebido: caption acima, label junto à caption, booktabs, ausência de linhas verticais, notas abaixo e largura estendida `\\fulllength`. A fonte é `\\small`, com espaçamento uniforme e quatro casas decimais. Os números são alinhados à direita; descrições à esquerda. Não há resizebox, screenshots nem tabelas convertidas em imagens. Para tabelas multipágina, longtable repete os cabeçalhos e identifica a continuação; não o envolva em table, minipage ou resizebox.

Os códigos compactos de modelos, canais e conjuntos estão definidos nas notas em inglês. Capturas são expressas em porcentagem; desvio padrão não é intervalo de confiança. Travessão indica não disponível/não aplicável, nunca zero. Não foram acrescentados negritos de “significância” ou novos testes estatísticos.

PREVIA_TABELAS.tex é uma prova autônoma com a largura do template MDPI, para checagem de composição. Não substitui a classe oficial, nem modifica o manuscrito. A numeração principal final depende da posição das tabelas no artigo; referências devem usar os labels fornecidos. As tabelas S1–S8 destinam-se ao suplemento.

Verificação: as contagens de linhas e a estrutura dos blocos foram conferidas. A compilação da prévia foi tentada no editor integrado, mas o ambiente retornou “Unable to find standard directories for platform”. Portanto, a paginação e eventuais transbordamentos ainda precisam ser conferidos na compilação do projeto MDPI. A precisão integral dos valores permanece nos CSVs; os indicadores nas tabelas são apresentados com quatro casas decimais.
'''
(OUT/'LEIA_ME.md').write_text(readme,encoding='utf-8')
css='body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:auto;padding:32px;color:#233640}h1,h2{line-height:1.2}section{margin:30px 0;border-top:1px solid #cad8df;padding-top:20px}pre{white-space:pre;overflow:auto;max-height:650px;padding:20px;background:#f1f5f7;font:13px/1.45 Consolas,monospace}button{background:#17667a;color:white;border:0;border-radius:5px;padding:10px 18px;cursor:pointer}a{color:#17667a}nav{display:flex;flex-wrap:wrap;gap:15px}'
sections=[]
parts=[('00_PREAMBULO','Preâmbulo: copiar uma vez',PREAMBLE)]+[(k,k.replace('_',' '),'\n'.join(v)) for k,v in sorted(blocks.items())]
for key,title,text in parts:
    sections.append(f'<section id="{key}"><h2>{title}</h2><button onclick="copyCode(this)">Copiar LaTeX</button> <a href="{key}.tex">Abrir .tex</a><pre><code>{html.escape(text)}</code></pre></section>')
js='''async function copyCode(b){const p=b.parentElement.querySelector('code');try{await navigator.clipboard.writeText(p.textContent);b.textContent='Copiado';}catch(e){const s=window.getSelection();const r=document.createRange();r.selectNodeContents(p);s.removeAllRanges();s.addRange(r);b.textContent='Selecionado: pressione Ctrl+C';}}'''
page='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tabelas LaTeX — MDPI Aerospace</title><style>'+css+'</style><body><h1>Tabelas LaTeX para copiar e colar</h1><p>4 tabelas principais e 8 suplementares, com legendas e notas em inglês. Copie primeiro o preâmbulo; depois os blocos desejados. As tabelas extensas usam cabeçalhos repetidos e continuação automática.</p><p><a href="TODAS_AS_TABELAS.tex">Todas as tabelas em um arquivo</a> · <a href="LEIA_ME.md">Instruções</a></p><nav>'+''.join(f'<a href="#{k}">{t}</a>' for k,t,_ in parts)+'</nav>'+''.join(sections)+'<script>'+js+'</script></body></html>'
(OUT/'COPIAR_TABELAS.html').write_text(page,encoding='utf-8')
print(json.dumps(check,indent=2))
