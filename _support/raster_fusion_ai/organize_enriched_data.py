"""Copia inputs originais e derivados para treino/teste em um conjunto auditado."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import rasterio

ROOT=Path(__file__).resolve().parent
LAYERS=('Garimpos','Ocorrencias_Minerais','Pistas_Irregulares','Pistas_Regulares')
REGIONS={'regiao_treino':'treinamento','regiao_teste_final':'teste_final'}
INPUTS=('original','magnitude','direcao_seno','direcao_cosseno','conectividade')
AUX=('direcao_graus','suporte')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    tmp.replace(path)


def latest_complete(folder):
    runs=sorted(p for p in folder.iterdir() if p.is_dir() and (p/'manifest.json').is_file())
    complete=[]
    for p in runs:
        m=json.loads((p/'manifest.json').read_text(encoding='utf8'))
        if m.get('status')=='complete' and len(m.get('results',{}))==32: complete.append(p)
    if not complete: raise ValueError(f'Nenhuma execução completa com 32 casos: {folder}')
    return complete[-1]


def profile(path):
    with rasterio.open(path) as ds:
        if ds.count!=1: raise ValueError(f'Banda inesperada: {path}')
        return ds.width,ds.height,ds.crs,ds.transform


def inputs(data,gradient,graph):
    records=[]; evaluation_targets={}
    for region,destination in REGIONS.items():
        for year in range(2019,2023):
            y=str(year)
            for layer in LAYERS:
                base=data/region/y/'inputs'/f'{layer}.tif'
                g=gradient/region/y/'inputs'
                n=graph/region/y/layer
                paths={
                    'original':base,
                    'magnitude':g/f'{layer}_magnitude.tif',
                    'direcao_seno':g/f'{layer}_direcao_seno.tif',
                    'direcao_cosseno':g/f'{layer}_direcao_cosseno.tif',
                    'conectividade':n/'conectividade.tif',
                    'direcao_graus':g/f'{layer}_direcao_graus.tif',
                    'suporte':n/'suporte.tif',
                }
                for name,p in paths.items():
                    if not p.is_file(): raise FileNotFoundError(p)
                ref=profile(base)
                for name,p in paths.items():
                    if profile(p)!=ref: raise ValueError(f'Grade incompatível em {region}/{y}/{layer}/{name}: {p}')
                    sub='inputs' if name in INPUTS else 'auxiliares'
                    records.append(dict(role=sub,source=str(p),relative=f'{destination}/{y}/{sub}/{layer}/{name}.tif',
                                        region=region,year=year,layer=layer,kind=name,bytes=p.stat().st_size))
            target=data/region/y/'target.tif'
            if year>=2020:
                if not target.is_file(): raise FileNotFoundError(target)
                evaluation_targets[f'{destination}/{y}']=str(target)
                if destination=='treinamento':
                    records.append(dict(role='target',source=str(target),relative=f'{destination}/{y}/targets/target.tif',
                                        region=region,year=year,layer=None,kind='target',bytes=target.stat().st_size))
    return records,evaluation_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--gradient-run',type=Path)
    p.add_argument('--graph-run',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    data=args.data.resolve()
    gradient=(args.gradient_run or latest_complete(ROOT/'outputs/gradient_inputs')).resolve()
    graph=(args.graph_run or latest_complete(ROOT/'outputs/graph_inputs')).resolve()
    records,targets=inputs(data,gradient,graph)
    counts={role:sum(r['role']==role for r in records) for role in ('inputs','auxiliares','target')}
    print(f"Pré-verificação OK: {counts}; {sum(r['bytes'] for r in records)/1e9:.2f} GB a copiar",flush=True)
    if args.dry_run:return
    dest=(args.output or data/'conjuntos_enriquecidos'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if dest==data or data in dest.parents and dest in (data/'regiao_treino',data/'regiao_teste_final'):
        p.error('Pasta de saída inválida')
    dest.mkdir(parents=True,exist_ok=True)
    manifest_file=dest/'manifest.json'
    config=dict(data=str(data),gradient_run=str(gradient),graph_run=str(graph),
                source_code_hash=digest(Path(__file__)),records=records,evaluation_targets=targets)
    if manifest_file.exists():
        manifest=json.loads(manifest_file.read_text(encoding='utf8'))
        if not args.resume or manifest['config']!=config: p.error('Retomada exige mesmas fontes, código e parâmetros')
    else:
        if any(dest.iterdir()): p.error('Escolha uma pasta vazia')
        manifest=dict(status='running',config=config,files={})
        shutil.copy2(__file__,dest/Path(__file__).name)
        save(manifest_file,manifest)
    for i,record in enumerate(records,1):
        source=Path(record['source']); target=dest/record['relative']
        if record['relative'] in manifest['files']:
            saved=manifest['files'][record['relative']]
            if saved['source_sha256']!=digest(source): raise ValueError(f'Fonte alterada: {source}')
            if target.is_file() and digest(target)==saved['sha256']:
                print(f'[{i}/{len(records)}] Conferido {record["relative"]}',flush=True)
                continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.partial.tif')
        source_hash=digest(source)
        shutil.copy2(source,temp)
        copied_hash=digest(temp)
        if copied_hash!=source_hash:
            raise IOError(f'Cópia não confere: {source} → {temp}')
        temp.replace(target)
        manifest['files'][record['relative']]=dict(source_sha256=source_hash,sha256=copied_hash,bytes=record['bytes'])
        save(manifest_file,manifest)
        print(f'[{i}/{len(records)}] Copiado {record["relative"]}',flush=True)
    manifest['status']='complete';save(manifest_file,manifest)
    readme='''# Conjunto enriquecido para experimentos

Pastas `treinamento` e `teste_final`, com anos 2019–2022 e quatro camadas.
Em `ANO/inputs/CAMADA/`, cinco entradas independentes: original, magnitude,
direcao_seno, direcao_cosseno e conectividade. Em `ANO/auxiliares/CAMADA/`,
direcao_graus para visualização e suporte binário do grafo. Não alimentar o
modelo com graus como variável linear. O suporte é auxiliar, não um input
automaticamente selecionado. Zeros válidos são preservados.

Targets de treino 2020–2022 ficam em `treinamento/ANO/targets/target.tif`,
isolados dos inputs. Não há target de 2019. A pasta `teste_final` tem apenas
entradas e auxiliares. Targets de avaliação externa continuam na origem,
referenciados em manifest.json/evaluation_targets. Não usar esses targets para
ajustar, normalizar, selecionar ou treinar modelos.

Tudo foi copiado byte a byte; hashes SHA-256 conferidos e registrados no
manifesto. Grades de cada input original e derivado foram comparadas antes
da cópia. As grades de targets podem diferir das dos inputs: alinhar durante
a leitura usando a política do experimento, não sobrescrever fontes.

`Conectividade` é o raster ponderado extraído de um grafo independente da
camada/ano/região. `suporte` indica alcance de nó no raio de 50 km; onde
suporte=0, a conectividade=0 significa ausência de nós próximos. Magnitude
e direção são derivados do raster original. São candidatos a atributos,
não ganhos preditivos demonstrados. Ajustar normalizações só no treino.

Este organizador não executa treinamento nem avaliação. O manifesto registra
as execuções exatas de origem dos gradientes e grafos e a correspondência
de cada arquivo; a procedência das feições dos grafos tem as limitações
registradas em docs/GRAPH_INPUTS.md.
'''
    (dest/'LEIA_ME.md').write_text(readme,encoding='utf8')
    print(f'Concluído: {dest}',flush=True)


if __name__=='__main__':main()
