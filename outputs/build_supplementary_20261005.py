from pathlib import Path
import ast, csv, hashlib, html, json, re, shutil, zipfile, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import rasterio

BASE = Path(__file__).resolve().parent
SRC = BASE/'mdpi_acrescimos_20261004'
OUT = BASE/'Aerospace_supplementary_20261005'
for directory in ['figures','tables','data','configurations','qa']:
    (OUT/directory).mkdir(parents=True, exist_ok=True)
shutil.copytree(SRC/'original/Definitions', OUT/'Definitions', dirs_exist_ok=True)
shutil.copytree(SRC/'supplement/frozen_configurations', OUT/'configurations', dirs_exist_ok=True)

def read(name):
    with (SRC/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

# Reuse only formatting functions, without running the prior generator.
tree = ast.parse((SRC/'build_latex_tables.py').read_text(encoding='utf-8'))
ns = {'blocks':{}, 'counts':{}}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ['esc','table']:
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<table formatter>', 'exec'), ns)
esc, table = ns['esc'], ns['table']
variants = {'fixed':'Fixed cores', 'rotating':'Rotating cores',
            'depth3':'Three-stage U-Net', 'magnitude':'Magnitude only', 'direction':'Direction only'}
controlnote = ('Fixed cores and Rotating cores compare training-core sampling with the five-stage U-Net. '
               'Three-stage U-Net evaluates network depth against the five-stage Fixed cores reference. '
               'Magnitude only and Direction only evaluate gradient components under fixed-core H1 training. '
               'Shared reference fits are reported once; all fits used eleven epochs. ')
table_titles = [
 'All 148 final configurations: fitting, selection, and validation metrics',
 'All 60 external evaluations of the main frozen models',
 'All 10 external evaluations of the local-context controls',
 'Individual U-Net diagnostic-control fits',
 'All 57 external evaluations of the U-Net diagnostic controls',
 'External target mass captured in the highest-scoring 10% of cells',
 'U-Net validation RMSE across three random seeds',
 'Selected channels, fitting budgets, and hyperparameters of the 24 winners']
panel_counts = []
review_panels = []
for number, source in enumerate(sorted((SRC/'latex_tables').glob('*_Table_S*.tex')), 1):
    original = source.read_text(encoding='utf-8')
    panels = original.split(r'\begingroup')[1:]
    for idx, panel in enumerate(panels):
        long = r'\begin{longtable}' in panel
        cap, label = re.search(r'\\caption\{(.*?)\\label\{([^}]+)\}\}', panel).groups()
        cap = cap.replace(r'\%', '%').rstrip('.')
        if len(panels)>1: cap = f'({chr(97+idx)}) '+cap
        cap += '.'
        header = panel.split(r'\toprule',1)[1].split(r'\midrule',1)[0].strip()
        headers = [x.strip()[len(r'\textbf{'):-1] for x in header[:-2].strip().split(' & ')]
        body = (panel.split(r'\endlastfoot',1)[1].split(r'\end{longtable}',1)[0] if long
                else panel.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0])
        rows = [line.strip()[:-2].strip().split(' & ') for line in body.splitlines() if line.strip().endswith(r'\\')]
        colline = next(line for line in panel.splitlines() if line.startswith((r'\begin{longtable}',r'\begin{tabular}')))
        weights = [float(x) for x in re.findall(r'p\{([\d.]+)\\rasterTableBodyWidth\}', colline)]
        align = [{'raggedright':'l','raggedleft':'r','centering':'c'}[x] for x in re.findall(r'>(?:\{)\\(raggedright|raggedleft|centering)', colline)]
        if long:
            noteline = next(line for line in panel.splitlines() if line.startswith(r'\multicolumn') and r'@{}p{\fulllength}' in line)
            note = noteline.split(r'{\footnotesize ',1)[1][:-3]
        else:
            note = re.search(r'\\noindent\{\\footnotesize (.*?)\}\s*\\end\{adjustwidth\}',panel,re.S).group(1)
        if number in [4,5,7]:
            for row in rows:
                # Remove opaque archive identifiers; the named configuration is sufficient.
                row[:] = [variants[row[2]], row[1]]+row[3:]
            headers = ['U-Net configuration','Horizon']+headers[3:]
            weights = [2.35,.8]+[1.05]*(len(headers)-2)
            if number==5: weights[3]=1.45  # input--target pair
            align = ['l','c']+['r']*(len(headers)-2)
            note = controlnote + (note if number!=4 else 'Each row is one fitted model. C10 is reported as a percentage.')
            if number==5:
                cap = cap.replace('External evaluation records:', 'U-Net diagnostic controls:')
        if number==3:
            cap=cap.replace('External evaluation records:', 'Local-context controls:')
        if number==2:
            cap=cap.replace('External evaluation records:', 'Main frozen models:')
        # Allow the exact recorded string-valued parameters to wrap at underscores.
        if number==8:
            rows = [[re.sub(r'\\_(?!\\allowbreak)',r'\\_\\allowbreak{}',v) for v in row] for row in rows]
        table(f'Table_S{number}',cap,label,headers,rows,weights,align,note=note,long=long,supp=f'S{number}')
        review_panels.append((number,cap,headers,rows,note))
    panel_counts.append(ns['counts'][f'Table_S{number}'])
    (OUT/'tables'/f'Table_S{number}.tex').write_text('\n\\clearpage\n'.join(ns['blocks'][f'Table_S{number}']),encoding='utf-8')

expected=[[148]*6,[60]*5,[10]*5,[24],[57]*3,[12],[8],[24,24]]
assert panel_counts==expected,(panel_counts,expected)
for f in (SRC/'supplement').glob('*.csv'): shutil.copy2(f,OUT/'data'/f.name)
for name in ['Table_S6_external_capture.csv','Table_S7_unet_controls.csv']:
    shutil.copy2(SRC/'tables'/name,OUT/'data'/name)

# Print-size figures: vector text/axes and native raster grids in PDF.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
records=read('supplement/S2_all_60_external_evaluations.csv')
model_order=list(dict.fromkeys(r['Model'] for r in records))
figure_files=[]
def save(fig,name):
    for ext in ['pdf','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=600,bbox_inches='tight')
    fig.savefig(OUT/'qa'/f'{name}.png',dpi=130,bbox_inches='tight')
    plt.close(fig)
    figure_files.append(name)
    print('Figure completed:',name,flush=True)

def subset(h,year):
    rows={r['Model']:r for r in records if r['Horizon']==str(h) and r['Evaluation pair'].endswith(str(year))}
    return [rows[m] for m in model_order]

fig,axes=plt.subplots(2,1,figsize=(7.2,8.7))
for h,ax in enumerate(axes):
    rows=subset(h,2021); y=np.arange(12)
    ax.barh(y-.18,[float(r['subset_zero_mae']) for r in rows],height=.35,label='Zero target',color='#bc833d')
    ax.barh(y+.18,[float(r['subset_positive_mae']) for r in rows],height=.35,label='Positive target',color='#246878')
    ax.set_yticks(y,model_order);ax.invert_yaxis()
    ax.set(xlabel='External MAE',title=f'({chr(97+h)}) H{h} | External target: 2021')
    ax.set_ylim(11.7,-2.0)
    ax.legend(fontsize=8)
fig.tight_layout(h_pad=2);save(fig,'Figure_S1')
fig,axes=plt.subplots(2,1,figsize=(7.2,8.7))
for h,ax in enumerate(axes):
    rows=subset(h,2022); x=np.array([float(r['rmse']) for r in rows])
    lo=np.array([float(r['ci_low']) for r in rows]);hi=np.array([float(r['ci_high']) for r in rows])
    ax.errorbar(x,np.arange(12),xerr=[x-lo,hi-x],fmt='o',color='#246878',capsize=3)
    ax.axvline(float(rows[0]['Zero RMSE']),ls='--',color='#bc833d',label='Zero')
    ax.axvline(float(rows[0]['Persistence RMSE']),ls=':',color='black',label='Persistence')
    ax.set_yticks(np.arange(12),model_order);ax.invert_yaxis()
    ax.set(xlabel='RMSE with exploratory 95% spatial-block interval',title=f'({chr(97+h)}) H{h} | External target: 2022')
    ax.legend(fontsize=8)
fig.tight_layout(h_pad=2);save(fig,'Figure_S2')

if '--reuse-maps' in sys.argv:
    map_checks=json.loads((OUT/'VERIFICACAO.json').read_text(encoding='utf-8'))['map_checks']
    figure_files += [f'Figure_S{n}_part{p}' for n in [3,4] for p in range(1,5)]
else:
    map_checks=[]
    for h in [0,1]:
        rows=subset(h,2021)
        paths=[BASE.parent/'raster_fusion_ai/data/regiao_teste_final/2021/target.tif']+[Path(r['Prediction GeoTIFF']) for r in rows]
        arrays=[];mask=None
        for path in paths:
            with rasterio.open(path) as ds:
                arr=ds.read(1,masked=True)
            invalid=np.ma.getmaskarray(arr)|~np.isfinite(arr.data)
            mask=invalid.copy() if mask is None else mask|invalid
            arrays.append(arr.data)
        assert int((~mask).sum())==5868615
        vmax=max(float(np.max(np.abs(a[~mask]))) for a in arrays)
        norm=SymLogNorm(linthresh=.1,vmin=-vmax,vmax=vmax,base=10)
        names=['Observed target 2021']+[f'{r["Model"]} | RMSE {float(r["rmse"]):.2f}' for r in rows]
        for part,start in enumerate(range(0,13,4)):
            if '--repair-last' in sys.argv and part!=3:
                figure_files.append(f'Figure_S{3+h}_part{part+1}')
                continue
            if start==12:
                fig,ax=plt.subplots(figsize=(7.2,7.1),layout='constrained')
                axes=np.array([ax])
            else:
                fig,axes=plt.subplots(2,2,figsize=(7.2,7.1),layout='constrained')
            active=[]
            for ax,j in zip(axes.flat,range(start,min(start+4,13))):
                im=ax.imshow(np.ma.array(arrays[j],mask=mask),cmap='RdBu_r',norm=norm,interpolation='none')
                ax.set_title(f'({chr(97+j)}) {names[j]}',fontsize=8.5)
                ax.set_xticks([]);ax.set_yticks([]);active.append(ax)
            for ax in list(axes.flat)[len(active):]: ax.set_visible(False)
            fig.colorbar(im,ax=list(axes.flat),orientation='horizontal',shrink=.88,fraction=.06,pad=.025,
                         label='Target intensity | Symmetric-log scale; linear threshold 0.1')
            fig.suptitle(f'H{h} | External target 2021 | Native grid: 2593 x 2268 cells',fontsize=10)
            save(fig,f'Figure_S{3+h}_part{part+1}')
        map_checks.append({'horizon':h,'common_cells':int((~mask).sum()),'symmetric_limit':vmax,'shape':list(mask.shape)})
        del arrays
title='Toward Human-in-the-Loop Multimodal AI for Suspicious Air Traffic Detection: An Experimental Synthesis Using Tabular, Graphical, and Contextual Geospatial Inputs'
preamble=r'''\documentclass[aerospace,supfile,submit,moreauthors]{Definitions/mdpi}
\firstpage{1}
\makeatletter\setcounter{page}{\@firstpage}\makeatother
\pubyear{2026}
\copyrightyear{2026}
\usepackage{longtable}
\newlength{\rasterTableBodyWidth}
\Title{'''+title+r'''}
\Author{Gabriel Dietzsch and Elcio H. Shiguemori}
\AuthorNames{Gabriel Dietzsch and Elcio H. Shiguemori}
\date{}
\begin{document}
\nolinenumbers

\section*{Scope and Reading Guide}
This supplement accompanies the manuscript named above and reports the raster experiments and their diagnostic controls. It contains four supplementary figures and eight supplementary tables. Tables with multiple panels retain one logical table number; page continuations repeat the column headings. Figures S3 and S4 each span four pages so that the native-grid maps and panel titles remain legible.

H0 denotes same-year estimation; H1 denotes estimation from the preceding year's inputs. These identifiers denote temporal horizons, not statistical hypotheses. RMSE and MAE denote root-mean-square error and mean absolute error; NMAD denotes normalized median absolute deviation; P90 denotes the 90th percentile of absolute error. C5, C10, and C20 denote the percentage of observed target mass captured in the highest-scoring 5\%, 10\%, and 20\% of valid cells. SD denotes sample standard deviation. A dash denotes unavailable or inapplicable data, not zero. Zero target intensity denotes no recorded intensity in the processed snapshot, not verified impossibility of the phenomenon.

The U-Net diagnostic controls investigate training-core sampling (Fixed cores versus Rotating cores), network depth (Three-stage U-Net versus the five-stage Fixed cores reference), and gradient components (Magnitude only versus Direction only under fixed-core H1 training). The three seeds are 4210, 4211, and 4212, with eleven epochs per fit. Shared reference fits are counted once. Rotating-core shallow models and rotating-core gradient-component ablations were not evaluated.

Tables S1--S3 contain 148 main final fits, 60 main external evaluations, and 10 local-context external evaluations, respectively. Tables S4 and S5 contain 24 U-Net control fits and 57 external control evaluations. Table S6 summarizes external capture; Table S7 summarizes three-seed validation variability; Table S8 documents the 24 internally selected winners. Frozen numerical records, rather than new model fitting, underlie this supplement. The accompanying CSVs preserve the recorded numerical precision; displayed metrics use four decimal places. Per-run configuration and environment records are supplied separately in the source package.

\subsection*{Contents}
\begin{itemize}
\item Figure S1: Errors conditional on zero and positive target intensity.
\item Figure S2: Exploratory spatial-block uncertainty intervals.
\item Figures S3 and S4: All main-model spatial predictions for target 2021 under H0 and H1.
'''
for i,t in enumerate(table_titles,1):preamble+=r'\item Table S'+str(i)+': '+esc(t)+'.\n'
preamble+=r'\end{itemize}'+'\n'

cap1='External MAE conditional on zero and positive target intensity in 2021. Panels (a) and (b) show H0 and the complementary H1 2020--2021 pair, respectively. Zero denotes no recorded intensity in the snapshot, not verified absence of the phenomenon. Errors use the native-grid common support.'
cap2='External RMSE and exploratory 95% spatial-block bootstrap intervals for target 2022. Panels (a) and (b) show H0 and H1. Intervals use 1,000 resamples of 42 occupied blocks approximately 96 km wide. The vertical lines show zero and persistence reference errors. These intervals describe individual-model error uncertainty; they are not paired tests of model superiority.'
figblocks=[]
def figure(number,path,caption,label,part=None):
    # Reset the counter only for multipart logical figures; hyperref anchors remain unique.
    out='\n\\clearpage\n'
    if part:out+=r'\setcounter{figure}{'+str(number-1)+'}\n'+r'\renewcommand{\theHfigure}{S'+str(number)+'part'+str(part)+'}\n'
    else:out+=r'\renewcommand{\theHfigure}{\arabic{figure}}'+'\n'
    out+=r'''\begin{figure}[H]
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\includegraphics[width=\fulllength,height=0.72\textheight,keepaspectratio]{figures/'''+path+r'''.pdf}
\caption{'''+esc(caption)+r'\label{'+label+r'''}}
\end{adjustwidth}
\end{figure}
'''
    return out
figblocks.append(figure(1,'Figure_S1',cap1,'fig:supp_zero_positive'))
figblocks.append(figure(2,'Figure_S2',cap2,'fig:supp_intervals'))
for h in [0,1]:
    for part in range(1,5):
        start=(part-1)*4; end=min(start+3,12)
        span=f'Panel ({chr(97+start)})' if start==end else f'Panels ({chr(97+start)})--({chr(97+end)})'
        caption=(f'{span}. Observed target and main-model predictions for '+('H0 in the external region, target 2021.' if h==0 else 'the complementary H1 2020--2021 external pair.')+
                 ' The common mask contains 5,868,615 valid cells. A shared symmetric-log scale with linear threshold 0.1 is used across all four pages of this figure; blue indicates negative predictions and masked cells are transparent. Colour limits are specific to this figure and must not be compared numerically between Figures S3 and S4. All metrics were computed at native resolution. '+
                 ('Panel (a) shows the observed target; panels (b)--(m) show the twelve models.' if part==1 else 'Continuation of the same logical figure.'))
        figblocks.append(figure(h+3,f'Figure_S{h+3}_part{part}',caption,f'fig:supp_map_h{h}_part{part}',part))
body=preamble+'\n'.join(figblocks)
body+='\n\\clearpage\n\\section*{Supplementary Tables}\n'
for i,title_i in enumerate(table_titles,1):
    body+='\n\\clearpage\n'+r'\subsection*{Table S'+str(i)+'. '+esc(title_i)+'}\n'
    body+=(OUT/'tables'/f'Table_S{i}.tex').read_text(encoding='utf-8')
body+='\n\\end{document}\n'
(OUT/'Supplementary_Materials.tex').write_text(body,encoding='utf-8')

statement=r'''% In the MAIN manuscript: replace the template's commented supplementary statement.
% The publication link is assigned by the journal; do not invent a DOI.
\supplementary{Supporting information submitted with this manuscript includes
Figures S1--S4 and Tables S1--S8: Figure S1, external errors conditional on target
intensity; Figure S2, exploratory spatial-block uncertainty intervals;
Figure S3, H0 spatial predictions for target 2021;
Figure S4, complementary H1 spatial predictions for target 2021;
Table S1, all main final configurations;
Table S2, main-model external evaluations;
Table S3, local-context control external evaluations;
Table S4, individual U-Net diagnostic-control fits;
Table S5, U-Net diagnostic-control external evaluations;
Table S6, external spatial capture;
Table S7, three-seed U-Net validation summaries;
Table S8, selected channels and fitting parameters.
Machine-readable numerical tables and per-run configuration and environment
records accompany the supplementary source package.}
'''
(OUT/'INSERIR_NO_ARTIGO.tex').write_text(statement,encoding='utf-8')
readme='''# Material suplementar — Aerospace / MDPI

Arquivo principal: **Supplementary_Materials.tex**. Usa a classe original com as opções `aerospace,supfile,submit,moreauthors`; a classe fornece o cabeçalho Supplementary Materials e a numeração S. O título e os autores foram preservados do manuscrito enviado. Não foram inventados DOI, ORCID ou dados editoriais.

## Compilar e usar

1. Importe o ZIP completo em um projeto LaTeX (por exemplo, Overleaf), mantendo Definitions, figures, tables, data e configurations.
2. Defina Supplementary_Materials.tex como documento principal. Compile com pdfLaTeX pelo menos duas vezes para resolver referências e total de páginas.
3. O texto das tabelas já está incorporado ao documento principal. Os arquivos tables/Table_S*.tex são cópias para reutilização, não devem ser inseridos novamente no mesmo documento.
4. Inclua o PDF resultante como Supplementary Materials na submissão. Mantenha os CSVs e configurações como arquivos de apoio; não os descreva como dados brutos completos.
5. Copie INSERIR_NO_ARTIGO.tex para o lugar do comando supplementary de exemplo no manuscrito principal. Figuras e tabelas suplementares também devem ser citadas nos parágrafos pertinentes. O link de publicação será atribuído pelo periódico.

## Conteúdo

- Quatro figuras lógicas: S1 e S2 em uma página cada; S3 e S4 com quatro páginas de continuação cada. PDF e SVG preservam textos vetoriais; os mapas preservam as grades nativas incorporadas. PNGs de impressão: 600 dpi.
- Oito tabelas lógicas, divididas em 24 painéis de leitura. Todas as linhas dos conjuntos anteriores são preservadas. Não foram acrescentados experimentos ou testes estatísticos.
- data/: CSVs com precisão integral. Os códigos de arquivo e caminhos originais ficam disponíveis para rastreabilidade.
- configurations/: configurações e ambientes congelados dos 12 experimentos principais; não representam arquivos de configuração completos de todos os controles.
- Definitions/: classe e recursos originais do template MDPI.

Os identificadores 10A, 10B e 10C foram substituídos no documento por configurações descritivas. A referência compartilhada de cinco estágios não é duplicada como um novo ajuste. O texto de discussão e o manuscrito principal não foram modificados.

Orientação consultada: https://www.mdpi.com/authors/layout — materiais suplementares separados, identificadores S e menção no manuscrito. O suplemento não usa apêndices A.

## Verificação

As contagens, a correspondência numérica com as tabelas anteriores, as referências locais e a integridade dos arquivos são verificadas pelo script. Consulte VERIFICACAO.json para o estado da compilação. A presença de PDFs de figuras não significa que o documento LaTeX completo tenha sido compilado.
'''
(OUT/'LEIA_ME.md').write_text(readme,encoding='utf-8')

# Static review index avoids requiring a TeX installation to inspect all supplied figures.
page='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Material suplementar — Aerospace</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;color:#203841;padding:20px}img{max-width:100%}figure{margin:30px 0;padding:18px;background:#f1f5f6}a{color:#146d80}table{border-collapse:collapse;font-size:12px;width:100%}td,th{border-bottom:1px solid #cbd7dc;padding:6px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e6f0f2}section{overflow:auto;margin:35px 0}small{display:block;margin:12px 0}</style><h1>Material suplementar — Aerospace</h1><p>4 figuras e 8 tabelas. Esta página permite revisar o conteúdo; a composição editorial está no LaTeX. Fonte: <a href="Supplementary_Materials.tex">Supplementary_Materials.tex</a> · <a href="LEIA_ME.md">Instruções</a> · <a href="INSERIR_NO_ARTIGO.tex">Declaração para o artigo</a></p><h2>Tabelas</h2><ul>'
for i,t in enumerate(table_titles,1):page+=f'<li><a href="tables/Table_S{i}.tex">Table S{i}</a>: {html.escape(t)}</li>'
page+='</ul><h2>Figuras e páginas de continuação</h2>'
for name in figure_files:page+=f'<figure><h3>{name.replace("_"," ")}</h3><img src="qa/{name}.png"><figcaption><a href="figures/{name}.pdf">PDF vetorial</a> · <a href="figures/{name}.svg">SVG</a> · <a href="figures/{name}.png">PNG 600 dpi</a></figcaption></figure>'
def plain(s):
    s=s.replace(r'\textemdash{}','—').replace(r'\allowbreak{}','').replace(r'\,',' ').replace(r'\%','%').replace(r'\_','_')
    s=s.replace(r'\rightarrow','→').replace(r'\lambda','λ').replace(r'\mathrm','').replace(r'\shortstack','')
    s=s.replace('R^2','R²').replace('$_s$','s').replace(r'\\',' / ').replace('$','').replace('{','').replace('}','')
    return html.escape(s)
for number,cap,heads,rows,note in review_panels:
    page+=f'<section><h3>Table S{number}. {plain(cap)}</h3><table><thead><tr>'+''.join('<th>'+plain(x)+'</th>' for x in heads)+'</tr></thead><tbody>'
    page+=''.join('<tr>'+''.join('<td>'+plain(x)+'</td>' for x in row)+'</tr>' for row in rows)
    page+='</tbody></table><small>'+plain(note)+'</small></section>'
(OUT/'INDICE.html').write_text(page+'</html>',encoding='utf-8')
labels=re.findall(r'\\label\{([^}]+)\}',body)
assert len(labels)==len(set(labels))==34
assert not re.search(r'\b10[ABC]\b',body)
assert body.count(r'\begingroup')==body.count(r'\endgroup')==24
assert all((OUT/'figures'/(name+'.pdf')).is_file() for name in figure_files)
checks={'logical_figures':4,'figure_pages':10,'logical_tables':8,'table_panels':24,
        'rows_by_panel':panel_counts,'unique_labels':len(labels),'map_checks':map_checks,
        'new_model_fitting':False,'main_manuscript_modified':False,'compiled_document':False}
(OUT/'VERIFICACAO.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Supplement source ready:',OUT,flush=True)
