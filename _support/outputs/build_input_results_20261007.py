from pathlib import Path
import ast,csv,hashlib,html,json,math,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

BASE=Path(__file__).resolve().parent
OLD=BASE/'mdpi_acrescimos_20261004'
ROOT=BASE.parent/'raster_fusion_ai'
OUT=BASE/'Aerospace_resultados_camadas_20261007'
for name in ['figures','data','qa']: (OUT/name).mkdir(parents=True,exist_ok=True)
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(name,rows):
    with (OUT/'data'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=read(OLD/'supplement/S1_all_148_final_fits.csv')
models=list(dict.fromkeys(r['Model'] for r in rows))
sets=['Original','Gradient','Graph','All']
labels=['Original (4)','Gradient only / GPA (12)','Graph only (8)','All channels (24)']
neural=['MLP','CNN','Compact ResNet','U-Net','ResUNet','Dense Inception']
summary=[];curves={};audit=[];points=[]
for h in ['0','1']:
    for model in models:
        arms={r['Input set']:r for r in rows if r['Model']==model and r['Horizon']==h}
        base=float(arms['Original']['Validation RMSE'])
        item={'Model':model,'Horizon':'H'+h,'Original RMSE':base}
        for arm in sets[1:]:
            value=float(arms[arm]['Validation RMSE']);item[arm+' RMSE']=value
            item[arm+' reduction (%)']=100*(1-value/base)
        summary.append(item)
        if model not in neural:continue
        for arm in sets:
            r=arms[arm];path=ROOT/r['Source JSON'];d=json.loads(path.read_text(encoding='utf-8'))
            trials=read(path.parent.parent/'hyperparameter_trials.csv')
            params=d['hyperparameters']
            matches=[t for t in trials if t['base_arm']==d['arm'] and
                (int(t['candidate_id'])==d['search_candidate_id'] if model=='MLP' else json.loads(t['params'])==params)]
            assert len(matches)==1,(model,h,arm)
            trial=matches[0];cid=trial['candidate_id'];selected=int(d['selected_epoch'])
            historypath=path.parent.parent/'hyperparameter_epoch_history.csv'
            history=[t for t in read(historypath) if t['base_arm']==d['arm'] and t['candidate_id']==cid]
            history.sort(key=lambda t:int(t['epoch']))
            epochs=[int(t['epoch']) for t in history]
            assert epochs==list(range(1,len(history)+1))
            selectedrow=next(t for t in history if int(t['epoch'])==selected)
            for val in [trial['selection_rmse'],selectedrow['selection_rmse']]:
                assert math.isclose(float(val),float(d['initial_selection_metrics']['rmse']),rel_tol=1e-9,abs_tol=1e-8)
            assert all(float(t['selection_rmse'])>0 for t in history)
            curves[(model,h,arm)]=(history,selected)
            audit.append({'Model':model,'Horizon':'H'+h,'Input set':arm,'Candidate':cid,
                          'Selected epoch':selected,'Recorded epochs':len(history),
                          'Selected selection RMSE':float(selectedrow['selection_rmse']),
                          'Final validation RMSE':float(r['Validation RMSE']),
                          'Parameters':json.dumps(params),'Source metrics':str(path),
                          'Source history':str(historypath),
                          'History SHA256':hashlib.sha256(historypath.read_bytes()).hexdigest()})
            for t in history:
                points.append({'Model':model,'Horizon':'H'+h,'Input set':arm,'Candidate':cid,
                               'Epoch':int(t['epoch']),'Selection RMSE':float(t['selection_rmse']),
                               'Scaled training loss':float(t['train_loss_scaled']),
                               'Selected checkpoint':int(t['epoch'])==selected})
assert len(audit)==48 and len(summary)==24
assert [sum(r['All reduction (%)']>0 for r in summary if r['Horizon']=='H'+str(h)) for h in [0,1]]==[11,10]
writecsv('validation_representation_comparison.csv',summary)
writecsv('selected_candidate_audit.csv',audit)
writecsv('recorded_learning_curves.csv',points)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
colors=['#5b6168','#007c91','#d18518','#8e3d78']
styles=['--','-',':','-.']
names=['Figure_learning_curves_compact','Figure_S5_learning_curves_dense']
for group,name in zip([neural[:3],neural[3:]],names):
    fig,axes=plt.subplots(3,2,figsize=(7.25,7.4))
    handles=[]
    for i,model in enumerate(group):
        for h in [0,1]:
            ax=axes[i,h]
            for j,arm in enumerate(sets):
                history,chosen=curves[model,str(h),arm]
                xx=[int(t['epoch']) for t in history]; yy=[float(t['selection_rmse']) for t in history]
                line,=ax.plot(xx,yy,color=colors[j],ls=styles[j],lw=1.2,label=labels[j])
                ax.scatter([chosen],[yy[xx.index(chosen)]],s=25,color=colors[j],edgecolor='white',linewidth=.4,zorder=5)
                if i==0 and h==0:handles.append(line)
            ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Selection RMSE (log scale)')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
            ax.set_title(f'({chr(97+i*2+h)}) {model} | H{h}')
            ax.grid(True,which='major',alpha=.15)
    fig.legend(handles=handles,labels=labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.0))
    fig.tight_layout(rect=[0,0,1,.94],h_pad=1.8,w_pad=1.2)
    for ext in ['pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=600,bbox_inches='tight')
    plt.close(fig)

tree=ast.parse((OLD/'build_latex_tables.py').read_text(encoding='utf-8'))
ns={'blocks':{},'counts':{}}
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ['esc','table']:
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<formatter>','exec'),ns)
tabrows=[[r['Model'],r['Horizon']]+[f'{r[k]:.4f}' for k in ['Original RMSE','Gradient RMSE','Graph RMSE','All RMSE']]+[f'{r["All reduction (%)"]:+.2f}' for _ in [0]] for r in summary]
ns['table']('comparison','Validation RMSE for the four unfiltered input representations.','tab:representation_comparison',
            ['Model','Horizon','Original','Gradient only','Graph only','All channels',r'All reduction (\%)'],
            tabrows,[2.2,.75,1,1.1,1.1,1.1,1.2],['l','c']+['r']*5,
            note=r'Original, 4 KDE channels; Gradient only, 12 magnitude/direction channels (archive label GPA); Graph only, 8 connectivity/support channels; All channels, all 24 channels. Reduction = $100(1-\mathrm{RMSE}_{all}/\mathrm{RMSE}_{original})$. Negative reduction denotes deterioration. Comparisons are internal and descriptive, with input-specific fitting and selection; they do not establish an isolated causal effect or external generalization.')
(OUT/'TABLE_COMPARISON.tex').write_text('\n'.join(ns['blocks']['comparison']),encoding='utf-8')

method=r'''% ADD to Materials and Methods, at the end of Models, Fitting, and Internal Selection.
Epoch-wise learning diagnostics were retrieved from the recorded hyperparameter-search histories. For each unfiltered input representation, neural-network family, and horizon, the curve corresponds to the candidate selected by the original internal procedure. Candidate identity, hyperparameters, selected epoch, and the recorded RMSE at that epoch were checked against the retained trial summary and model metrics. Curves display internal-selection RMSE in the original response units, on a logarithmic vertical axis, without smoothing or interpolation; markers identify the selected checkpoints. These are search-stage histories, distinct from both the final-refit training histories and the subsequent validation results. Hyperparameters and stopping epochs may differ across representations, and the selection supports differ across model families and horizons. Epoch counts are therefore not a common computational budget for cross-model comparisons.
'''
result=r'''% ADD in Results / Input Representations and Selected Channels,
% AFTER the first paragraph and BEFORE "Dense Inception supplied...".
The largest reductions depended on the model family and temporal horizon (Table~\ref{tab:representation_comparison}). In H0, combining all 24 channels reduced validation RMSE from 3.1085 to 2.1842 for the CNN (29.73\%) and from 7.4337 to 5.9138 for the MLP (20.45\%). For Compact ResNet, the graph-only representation reduced RMSE from 3.0997 to 2.6509 (14.48\%), whereas the full stack produced RMSE 2.8129. In H1, gradient-only inputs reduced CNN RMSE from 11.2057 to 7.6779 (31.48\%) and Compact ResNet RMSE from 8.4543 to 7.0480 (16.63\%). The full stack reduced HGB RMSE from 13.5235 to 10.1787 (24.73\%) and XGBoost RMSE from 13.6541 to 11.1100 (18.63\%).

These improvements were not uniform. With all channels, H1 validation RMSE increased from 14.1516 to 15.8146 for U-Net (11.75\%) and from 15.7953 to 17.2071 for ResUNet (8.94\%); Dense Inception H0 increased from 7.5020 to 7.7109 (2.79\%). Gradient-only and graph-only conditions replace the original inputs, whereas the full stack jointly includes original, gradient, and graph channels. Thus, these contrasts must be distinguished from the explicit original-plus-gradient and original-plus-graph addition controls described below.

% ADD after the existing final paragraph of this subsection
% (the paragraph beginning "The external comparison evaluated...").
\subsubsection{Learning Dynamics across Input Representations}
The epoch-wise histories provide additional evidence on fitting dynamics (Figure~\ref{fig:representation_learning}; Supplementary Figure S5). For CNN H0, the full-stack candidate had a selected internal-selection RMSE of 3.8181 at epoch 25, compared with 5.2306 at epoch 39 for the original-input candidate. Both candidates used 32 filters, a learning rate of 0.001, and weight decay of 0.01. For Compact ResNet H0, the graph-only candidate had a selected RMSE of 4.7779 at epoch 17, compared with 5.4721 at epoch 40 for the original-input candidate; these two candidates also shared the same recorded values of those three hyperparameters. These are earlier selected checkpoints with lower internal-selection error in the recorded runs, rather than replicated estimates of convergence speed.

The search-stage curves and final validation results did not always give the same ordering. For Compact ResNet H1, gradient-only inputs produced a selected internal-selection RMSE of 11.8929, compared with 10.7184 for original inputs, while final validation favoured gradient-only inputs (7.0480 versus 8.4543). The two representations also selected different network widths. Accordingly, the histories characterize the internal selection stage and do not replace the final validation or external comparisons. The complete unsmoothed curves include the dense-network cases, whose training cores rotated across epochs, in Supplementary Figure S5.
'''
(OUT/'ADD_TO_METHODS.tex').write_text(method,encoding='utf-8')
(OUT/'ADD_TO_RESULTS.tex').write_text(result,encoding='utf-8')
caption='Recorded internal-selection RMSE across epochs for MLP, CNN, and Compact ResNet. Rows identify model families; columns identify H0 and H1. Curves show the candidate selected within each input representation, with markers at the selected epochs. Counts in the legend denote input channels. The vertical axis is logarithmic; all recorded epochs are shown without smoothing. Candidates may differ in hyperparameters and stopping epochs. Selection supports differ between model families and horizons. These are search-stage selection curves, not final-validation or external-test curves.'
mainfig=r'''% ADD after the Learning Dynamics paragraphs. Figure numbering is automatic.
\begin{figure}[H]
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\includegraphics[width=\fulllength,height=0.75\textheight,keepaspectratio]{figures/Figure_learning_curves_compact.pdf}
\caption{'''+caption+r'''\label{fig:representation_learning}}
\end{adjustwidth}
\end{figure}
'''
suppcap='Recorded internal-selection RMSE across epochs for U-Net, ResUNet, and Dense Inception. Rows identify model families; columns identify H0 and H1. Curves show the originally selected candidate for each unfiltered representation; markers identify its selected epoch. The vertical axis is logarithmic and all recorded epochs are shown without smoothing. Training cores rotated across epochs, whereas internal-selection cores were fixed. The curves are internal-selection diagnostics, not final-validation or external-test results. Lower or smoother search-stage curves do not by themselves establish better subsequent performance. Counts in the legend denote input channels.'
suppfig=r'''% APPEND to Supplementary_Materials.tex before end{document}.
\clearpage
\setcounter{figure}{4}
\renewcommand{\theHfigure}{S5learning}
\begin{figure}[H]
\begin{adjustwidth}{-\extralength}{0cm}
\centering
\includegraphics[width=\fulllength,height=0.75\textheight,keepaspectratio]{figures/Figure_S5_learning_curves_dense.pdf}
\caption{'''+suppcap+r'''\label{fig:supp_dense_learning}}
\end{adjustwidth}
\end{figure}
'''
(OUT/'FIGURE_MAIN.tex').write_text(mainfig,encoding='utf-8')
(OUT/'FIGURE_SUPPLEMENT_S5.tex').write_text(suppfig,encoding='utf-8')

guide='''# Acréscimos: representações de entrada e dinâmica de aprendizagem

## Conclusão da conferência

Há ganhos grandes e históricos reais por época. A seção 4.2 proposta já comparava os conjuntos; estes blocos acrescentam os números mais informativos e as curvas. Nenhum texto anterior foi removido. O manuscrito e o suplemento aberto não foram modificados.

GPA, neste experimento, corresponde à magnitude do gradiente e ao seno/cosseno da direção, não a um descritor G1/G2. A legenda usa “Gradient only / GPA”. “Graph only” usa conectividade e suporte. As comparações Original versus Gradient only ou Graph only são substituições de representação. All channels adiciona conjuntamente ambos os grupos às originais. Só a Dense Inception tem os dois controles de adição isolada.

## Onde acrescentar

1. **Materials and Methods / Models, Fitting, and Internal Selection**: acrescente ADD_TO_METHODS.tex ao final da subseção, após o parágrafo de seleção de vencedores. Não substitua o texto existente.
2. **Results / Input Representations and Selected Channels**: acrescente os dois primeiros parágrafos de ADD_TO_RESULTS.tex após o primeiro parágrafo existente (terminado em “Supplementary Table S8.”), antes de “Dense Inception supplied the two explicit addition controls.”
3. Insira TABLE_COMPARISON.tex depois desses dois novos parágrafos. Reutiliza a declaração `\\newlength{\\rasterTableBodyWidth}` já fornecida: ela deve ocorrer apenas uma vez no preâmbulo.
4. Depois do último parágrafo existente dessa subseção, iniciado por “The external comparison evaluated...”, acrescente a subsubseção Learning Dynamics do mesmo ADD_TO_RESULTS.tex e depois FIGURE_MAIN.tex.
5. **Suplemento**: copie FIGURE_SUPPLEMENT_S5.tex antes de `\\end{document}`. Essa figura é a nova S5; as anteriores S1–S4 permanecem. Atualize “four supplementary figures” para “five supplementary figures” no guia de leitura e “Figures S1--S4” para “Figures S1--S5” no comando supplementary do artigo. Acrescente à relação de conteúdos: `Figure S5, recorded internal-selection learning curves for the dense neural networks`.
6. Envie os PDFs da pasta figures para a pasta figures dos respectivos projetos Overleaf. Apenas colar o `.tex` não envia as imagens. SVGs e PNGs a 600 dpi estão disponíveis para edição e impressão.

As novas tabelas e figuras do artigo recebem numeração automática. Como entram na seção 4.2, os números dos elementos seguintes podem mudar; preserve labels e use ref, em vez de escrever números manualmente. A figura anterior sobre a redução de RMSE (Figure 2 no roteiro inicial) permanece.

## Leitura rigorosa

- A tabela contém 24 comparações modelo/horizonte e preserva as quatro representações sem seleção de canais. Os 148 ajustes completos continuam na tabela S1 do suplemento.
- As 48 curvas foram vinculadas aos candidatos efetivamente selecionados: seis famílias, dois horizontes, quatro representações. O RMSE do checkpoint foi comparado com o resumo da busca e o JSON de métricas.
- O eixo vertical mostra RMSE na seleção interna, em escala logarítmica. A seleção pode ordenar as representações de modo diferente da validação posterior; Compact ResNet H1 fornece um exemplo concreto.
- Marcadores indicam os checkpoints escolhidos originalmente. Não se escolheram novos vencedores. Não há interpolação, suavização, novas rodadas ou intervalos inventados.
- CNN H0 (All) e Compact ResNet H0 (Graph) atingiram checkpoints selecionados mais cedo, com menor erro de seleção. Isso não prova uma vantagem geral de velocidade nem reduz todas as diferenças apenas aos canais.
- HGB e XGBoost são contemplados na tabela de resultados. Etapas de boosting não foram apresentadas como épocas de uma rede neural.

## Arquivos e verificação

data/validation_representation_comparison.csv: valores e percentuais com precisão integral.
data/selected_candidate_audit.csv: candidato, parâmetros, época, métricas e origem de cada curva.
data/recorded_learning_curves.csv: todos os pontos exibidos.

Os PDFs são figuras, não uma compilação do artigo ou do suplemento completo. Os blocos LaTeX são fragmentos para inserção no projeto MDPI existente.
'''
(OUT/'ONDE_ACRESCENTAR.md').write_text(guide,encoding='utf-8')
page='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:35px auto;padding:20px;color:#243841}img{max-width:100%}pre{white-space:pre-wrap;background:#f1f4f5;padding:18px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ccc;text-align:right}td:first-child{text-align:left}</style><h1>Resultados: GPA, grafos e curvas de aprendizagem</h1><p><a href="ONDE_ACRESCENTAR.md">Onde acrescentar e como interpretar</a></p><h2>Comparação na validação</h2><table><tr><th>Model</th><th>Horizon</th><th>Original RMSE</th><th>Gradient-only RMSE</th><th>Graph-only RMSE</th><th>All-channel RMSE</th><th>All reduction (%)</th></tr>'
for r in tabrows:page+='<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in r)+'</tr>'
page+='</table>'
for name,cap in zip(names,[caption,suppcap]):page+=f'<h2>{name.replace("_"," ")}</h2><img src="figures/{name}.png"><p>{html.escape(cap)}</p>'
for name in ['ADD_TO_METHODS.tex','ADD_TO_RESULTS.tex','TABLE_COMPARISON.tex','FIGURE_MAIN.tex','FIGURE_SUPPLEMENT_S5.tex']:
    page+=f'<h2>{name}</h2><pre>'+html.escape((OUT/name).read_text(encoding='utf-8'))+'</pre>'
(OUT/'LEITURA.html').write_text(page+'</html>',encoding='utf-8')
(OUT/'VERIFICACAO.json').write_text(json.dumps({'comparison_rows':24,'curves':48,'recorded_points':len(points),'checkpoints_match_archived_metrics':True,'no_new_fits':True,'unchanged_existing_manuscript':True},indent=2),encoding='utf-8')
print(json.dumps({'directory':str(OUT),'curves':len(audit),'points':len(points),'figures':names},indent=2))
