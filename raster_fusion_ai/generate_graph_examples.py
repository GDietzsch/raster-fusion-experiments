"""Quatro exemplos de grafos espaciais independentes, sem target."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import struct
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from generate_gradient_inputs import digest, ROOT, NODATA

LAYERS=['Garimpos','Ocorrencias_Minerais','Pistas_Irregulares','Pistas_Regulares']
RADIUS=50000.
K=8


def read_points(path):
    """Leitor estrito de SHP 2D Point/MultiPoint + flags de exclusão DBF."""
    b=path.read_bytes(); db=path.with_suffix('.dbf').read_bytes()
    count=struct.unpack_from('<I',db,4)[0]
    header,length=struct.unpack_from('<HH',db,8)
    if len(db)<header+count*length: raise ValueError('DBF truncado')
    offset=100; index=0; rows=[]; source_ids=[]
    while offset<len(b):
        _,words=struct.unpack_from('>ii',b,offset)
        record=b[offset+8:offset+8+words*2]
        if len(record)!=words*2: raise ValueError('SHP truncado')
        shape=struct.unpack_from('<i',record,0)[0]
        if index>=count: raise ValueError('SHP/DBF incompatíveis')
        if db[header+index*length:header+index*length+1]!=b'*':
            if shape==1: points=[struct.unpack_from('<2d',record,4)]
            elif shape==8:
                n=struct.unpack_from('<i',record,36)[0]
                points=[struct.unpack_from('<2d',record,40+16*j) for j in range(n)]
            elif shape==0: points=[]
            else: raise ValueError(f'Tipo geométrico não suportado: {shape}')
            for xy in points:
                if not np.isfinite(xy).all(): raise ValueError('Coordenada inválida')
                rows.append(xy); source_ids.append(index+1)
        offset+=8+words*2; index+=1
    if index!=count: raise ValueError('Contagem SHP/DBF diferente')
    return np.array(rows,dtype=float).reshape(-1,2),np.array(source_ids),count


def make_graph(xy,k=K,radius=RADIUS):
    """União simétrica de kNN limitado por raio. Distâncias euclidianas em m."""
    n=len(xy)
    if not n: raise ValueError('Nenhum ponto')
    tree=cKDTree(xy)
    dist,ix=tree.query(xy,k=min(k+1,n))
    if n==1: dist,ix=dist[:,None],ix[:,None]
    pairs=set()
    for i in range(n):
        # Ordenação torna o subconjunto retornado determinístico; empates no
        # limite k ainda dependem do cKDTree, versão registrada no manifesto.
        for d,j in sorted(zip(dist[i],ix[i])):
            if i!=j and d<=radius: pairs.add((min(i,int(j)),max(i,int(j))))
    edges=np.array(sorted(pairs),dtype=int).reshape(-1,2)
    lengths=np.linalg.norm(xy[edges[:,0]]-xy[edges[:,1]],axis=1)
    weights=np.exp(-lengths/radius)
    degree=np.bincount(edges.ravel(),minlength=n)
    strength=np.bincount(edges.ravel(),weights=np.repeat(weights,2),minlength=n)
    matrix=coo_matrix((np.ones(len(edges)*2),(edges.ravel(),edges[:,::-1].ravel())),shape=(n,n)).tocsr()
    nc,labels=connected_components(matrix,directed=False)
    sizes=np.bincount(labels)[labels]
    return tree,edges,lengths,weights,degree,strength,labels,sizes,nc


def surface(tree,xy,strength,transform,shape,radius=RADIUS,k=K):
    """Média ponderada da força de até k nós próximos dentro do raio."""
    result=np.zeros(shape,dtype='float32')
    support=np.zeros(shape,dtype='float32')
    for row in range(0,shape[0],128):
        h=min(128,shape[0]-row)
        rr,cc=np.mgrid[row:row+h,0:shape[1]]
        coords=np.column_stack((transform.c+(cc.ravel()+.5)*transform.a,
                                transform.f+(rr.ravel()+.5)*transform.e))
        d,ids=tree.query(coords,k=min(k,len(xy)))
        if d.ndim==1: d,ids=d[:,None],ids[:,None]
        w=np.where(d<=radius,np.exp(-d/radius),0.)
        s=w.sum(axis=1)
        v=np.divide((w*strength[ids]).sum(axis=1),s,out=np.zeros_like(s),where=s>0)
        result[row:row+h]=v.reshape(h,shape[1])
        support[row:row+h]=(s>0).reshape(h,shape[1])
    return result,support


def main():
    import scipy
    out=ROOT/'outputs/graph_examples'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    manifest=dict(method='independent_point_knn_graph_example_v1',year=2021,region='treino',
                  k=K,radius_m=RADIUS,scipy=scipy.__version__,layers={})
    panels=[]
    for layer in LAYERS:
        print(f'Preparando {layer}',flush=True)
        folder=out/layer; folder.mkdir()
        shp=ROOT.parent/'Projeto/Treinamento/Inputs'/f'{layer}_2021.shp'
        source=ROOT/'data/regiao_treino/2021/inputs'/f'{layer}.tif'
        points,source_ids,records=read_points(shp)
        with rasterio.open(source) as ds:
            src_crs=CRS.from_wkt(shp.with_suffix('.prj').read_text())
            dst_crs=CRS.from_wkt(ds.crs.to_wkt())
            if ds.crs.linear_units_factor[1]!=1 or ds.transform.b or ds.transform.d:
                raise ValueError('Este exemplo exige projeção métrica sem rotação')
            converter=Transformer.from_crs(src_crs,dst_crs,always_xy=True)
            px,py=converter.transform(points[:,0],points[:,1])
            points=np.column_stack((px,py))
            xy,inverse,counts=np.unique(points,axis=0,return_inverse=True,return_counts=True)
            tree,edges,lengths,weights,degree,strength,labels,sizes,nc=make_graph(xy)
            # Recorte ilustrativo deliberadamente em torno do nó mais conectado.
            center=xy[int(np.argmax(strength))]
            r,c=ds.index(*center)
            wh,ww=min(1024,ds.height),min(1024,ds.width)
            row=min(max(0,r-wh//2),ds.height-wh); col=min(max(0,c-ww//2),ds.width-ww)
            win=Window(col,row,ww,wh)
            original=ds.read(1,window=win,masked=True)
            transform=ds.window_transform(win)
            profile=ds.profile.copy()
            profile.update(height=wh,width=ww,transform=transform,dtype='float32',nodata=NODATA,
                           tiled=True,blockxsize=256,blockysize=256,compress='deflate',count=1)
            bounds=rasterio.windows.bounds(win,ds.transform)
        raster,support=surface(tree,xy,strength,transform,original.shape)
        invalid=np.ma.getmaskarray(original)|~np.isfinite(original.data)
        for name,a in [('original',original.filled(NODATA)),('conectividade',raster),('suporte',support)]:
            a=np.asarray(a).copy(); a[invalid]=NODATA
            with rasterio.open(folder/(name+'.tif'),'w',**profile) as dst:
                dst.write(a.astype('float32'),1)
                dst.update_tags(method=manifest['method'],source=str(source),source_points=str(shp))
        with (folder/'nos.csv').open('w',encoding='utf8',newline='') as f:
            writer=csv.writer(f); writer.writerow(['id','x_m','y_m','registros_coincidentes','grau','forca','componente','tamanho_componente'])
            for i,p in enumerate(xy): writer.writerow([i,*p,counts[i],degree[i],strength[i],labels[i],sizes[i]])
        with (folder/'arestas.csv').open('w',encoding='utf8',newline='') as f:
            writer=csv.writer(f); writer.writerow(['origem','destino','distancia_m','peso'])
            for e,d,w in zip(edges,lengths,weights): writer.writerow([*e,d,w])
        with (folder/'origem_nos.csv').open('w',encoding='utf8',newline='') as f:
            writer=csv.writer(f); writer.writerow(['registro_shp','id_no'])
            writer.writerows(zip(source_ids,inverse))
        manifest['layers'][layer]=dict(source=str(source),source_sha256=digest(source),
            shapefile_hashes={p.name:digest(p) for p in [shp,shp.with_suffix('.dbf'),shp.with_suffix('.shx'),shp.with_suffix('.prj')]},
            source_records=records,nodes=len(xy),edges=len(edges),components=int(nc),
            isolated=int(np.count_nonzero(degree==0)),crop=[col,row,ww,wh],
            crop_selection='Maior força do grafo, desempate pela ordem das coordenadas; não representativo',
            crs=dst_crs.to_wkt(),raster_min=float(raster.min()),raster_max=float(raster.max()))
        extent=(bounds[0]/1000,bounds[2]/1000,bounds[1]/1000,bounds[3]/1000)
        visible=(xy[:,0]>=bounds[0])&(xy[:,0]<=bounds[2])&(xy[:,1]>=bounds[1])&(xy[:,1]<=bounds[3])
        # Inclui arestas com pelo menos um extremo no recorte; eixos fazem o corte visual.
        shown=edges[visible[edges].any(axis=1)]
        panels.append((layer,original,raster,xy,shown,strength,visible,extent))
        fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
        draw(axes,panels[-1])
        fig.suptitle(f'{layer} · treino · 2021 | nós: {len(xy):,} · arestas: {len(edges):,}\nRecorte de 256 × 256 km; grafo calculado com todas as feições da camada',fontsize=12)
        fig.savefig(folder/'comparacao.png',dpi=600); plt.close(fig)
        print(f'{layer}: {len(xy)} nós / {len(edges)} arestas / {nc} componentes',flush=True)
    fig,axes=plt.subplots(4,3,figsize=(15,20),layout='constrained')
    for ax,p in zip(axes,panels): draw(ax,p)
    fig.suptitle('Grafos independentes por camada · treino · 2021\nRecortes ilustrativos distintos; conexões espaciais hipotéticas; sem Target',fontsize=16)
    fig.savefig(out/'painel_quatro_camadas.png',dpi=300); plt.close(fig)
    shutil.copy2(__file__,out/Path(__file__).name)
    manifest['outputs']={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    (out/'LEIA_ME.md').write_text('''# Exemplos de grafos independentes

Quatro shapefiles de entrada de 2021 em Projeto/Treinamento/Inputs. Sem Target,
sem ligações entre camadas, sem dados da área de teste e sem treinamento.
Feições coincidentes exatamente são um único nó; multiplicidade registrada,
mas não usada como peso. Cada coordenada de MultiPoint é um nó candidato.
origem_nos.csv mantém a ligação com o número do registro original do shapefile.
Fontes são os shapefiles homônimos do mesmo ano; sua equivalência exata às
feições usadas historicamente no KDE ainda não foi demonstrada.

União não dirigida de até oito vizinhos mais próximos por nó, limitada a 50 km.
A simetrização pode produzir grau maior que oito. Não força conexões entre
componentes distantes. Peso=exp(-distância/50km), força=soma dos pesos incidentes.
São parâmetros demonstrativos, não calibrados nem otimizados para o Target.
Distância é euclidiana na projeção, não distância de deslocamento ou conexão real.
O grafo usa todas as feições do arquivo, inclusive fora do recorte exibido.

O raster de conectividade é a média ponderada da força de até oito nós mais
próximos dentro de 50 km; peso espacial=exp(-distância/50km). Sem suporte,
recebe zero. suporte.tif distingue ausência de suporte de nós com força zero.
Essa rasterização adiciona uma escolha de interpolação e pode produzir limites
visíveis. Não equivale à densidade KDE e não comprova ganho preditivo.

Cada camada tem um recorte nativo de 1024x1024 pixels (256x256 km), centrado no nó
de maior força; não são a mesma área geográfica nem amostras representativas.
O máximo foi escolhido para tornar as conexões legíveis, sem consultar Target.
Escala da força é compartilhada entre os nós visíveis e seu raster em cada linha;
as escalas diferem entre camadas. Grafo completo preservado nos CSVs de nós e
arestas; GeoTIFFs representam só o recorte de demonstração.

Painéis individuais 9000x3600 a 600dpi; painel conjunto 4500x6000 a 300dpi.
Os originais não foram alterados. Não gera intermediação nesta demonstração;
o produto ilustrado é a conectividade ponderada de cada grafo.
''',encoding='utf8')
    print(out,flush=True)


def draw(axes,p):
    layer,original,raster,xy,shown,strength,visible,extent=p
    high=max(float(raster.max()),float(strength[visible].max()) if visible.any() else 0,1e-9)
    images=[]
    images.append(axes[0].imshow(original,extent=extent,cmap='viridis',interpolation='nearest'))
    axes[0].set_title(layer+' · input original')
    axes[1].set_facecolor('#f5f7fa')
    axes[1].add_collection(LineCollection(xy[shown]/1000,colors='#475569',linewidths=.35,alpha=.3))
    images.append(axes[1].scatter(xy[visible,0]/1000,xy[visible,1]/1000,c=strength[visible],
                                s=7,cmap='magma',vmin=0,vmax=high,zorder=3,edgecolors='none'))
    axes[1].set_title('Grafo · nós e conexões')
    images.append(axes[2].imshow(raster,extent=extent,cmap='magma',vmin=0,vmax=high,interpolation='nearest'))
    axes[2].set_title('Raster · conectividade ponderada')
    for ax,im,label in zip(axes,images,['Intensidade original','Força do nó','Média espacial da força']):
        ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='X (km)',ylabel='Y (km)',aspect='equal')
        ax.figure.colorbar(im,ax=ax,orientation='horizontal',pad=.10,label=label)


if __name__=='__main__': main()
