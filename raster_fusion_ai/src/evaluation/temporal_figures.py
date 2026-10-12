"""Reproducible scientific PNGs for saved temporal models and validation samples."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from threadpoolctl import threadpool_limits

STYLE = {'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
         'axes.labelsize':11,'legend.fontsize':10,'axes.spines.top':False,
         'axes.spines.right':False,'savefig.facecolor':'white'}


def _save(fig, path, dpi, caption):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=dpi,facecolor='white',metadata={
        'Description':caption,'Software':'Raster Fusion AI / Matplotlib'})
    width,height=np.round(fig.get_size_inches()*dpi).astype(int)
    plt.close(fig)
    return dict(file=path.name,dpi=dpi,width_pixels=int(width),height_pixels=int(height),caption=caption)


def _number(value):
    return 'undefined' if value is None else f'{value:.4g}'


def _fit_plot(target,prediction,path,dpi,title):
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    low=float(min(target.min(),prediction.min()))
    high=float(max(target.max(),prediction.max()))
    if high==low:
        low,high=low-1,high+1
    count=ax.hexbin(target,prediction,gridsize=90,mincnt=1,norm=LogNorm(),cmap='viridis')
    ax.plot([low,high],[low,high],color='#c13b36',linestyle='--',linewidth=1.2,label='1:1')
    ax.set(xlim=(low,high),ylim=(low,high),xlabel='Observed target (original scale)',
           ylabel='Predicted target (original scale)',title=title)
    ax.set_aspect('equal',adjustable='box')
    ax.legend(loc='upper left')
    fig.colorbar(count,ax=ax,label='Validation samples per hexagon (log scale)')
    fig.supxlabel(f'n = {len(target):,}; all values included; no clipping',fontsize=9)
    return _save(fig,path,dpi,title+'; all validation samples in original target scale; density is logarithmic, axes are linear.')


def _diagnostics(target,prediction,path,dpi):
    error=prediction-target
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    axes[0].hist(error,bins=80,color='#256e91',edgecolor='white',linewidth=.25)
    axes[0].axvline(0,color='#b33432',linestyle='--')
    axes[0].set(xlabel='Residual: predicted − observed',ylabel='Validation samples',title='Residual distribution',yscale='log')
    axes[1].hist(np.abs(error),bins=80,color='#437c58',edgecolor='white',linewidth=.25)
    axes[1].set(xlabel='Absolute error (original target scale)',ylabel='Validation samples',title='Absolute error distribution',yscale='log')
    fig.supxlabel(f'Validation sample; n = {len(target):,}; no clipping',fontsize=9)
    return _save(fig,path,dpi,'Residual and absolute-error histograms of all validation samples. Residual = prediction minus observation. Counts on logarithmic axes; target in original scale.')


def _spatial_map(target,prediction,positions,year,grid,path,dpi):
    from affine import Affine
    affine=Affine(*grid['transform'])
    columns,rows=positions[:,1]+.5,positions[:,0]+.5
    x=affine.a*columns+affine.b*rows+affine.c
    y=affine.d*columns+affine.e*rows+affine.f
    error=prediction-target
    low=float(min(target.min(),prediction.min()))
    high=float(max(target.max(),prediction.max()))
    if low==high:
        high=low+1
    scale=Normalize(vmin=low,vmax=high)
    max_error=max(float(np.max(np.abs(error))),1e-12)
    fig,axes=plt.subplots(1,3,figsize=(12,5),layout='constrained')
    # Render every sampled location; do not interpolate across unseen locations.
    for ax,values,title in zip(axes[:2],[target,prediction],['Observed target','Predicted target']):
        artist=ax.scatter(x,y,c=values,s=1.2,cmap='viridis',norm=scale,linewidths=0)
        ax.set(title=title)
    fig.colorbar(artist,ax=list(axes[:2]),label='Target (original scale)',shrink=.7)
    artist=axes[2].scatter(x,y,c=error,s=1.2,cmap='RdBu_r',
        norm=TwoSlopeNorm(vmin=-max_error,vcenter=0,vmax=max_error),linewidths=0)
    axes[2].set(title='Residual\n(predicted − observed)')
    fig.colorbar(artist,ax=axes[2],label='Residual (original scale)',shrink=.7)
    import rasterio
    unit=rasterio.crs.CRS.from_wkt(grid['crs']).linear_units
    for ax in axes:
        ax.set_aspect('equal',adjustable='box')
        ax.set(xlabel=f'X ({unit})',ylabel=f'Y ({unit})')
        ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0),useOffset=False)
        ax.xaxis.labelpad = 16  # keep scientific-notation multiplier clear of axis label
    fig.suptitle(f'Target year {year} · validation sample locations',fontsize=13)
    fig.supxlabel(f'n = {len(target):,}; blank locations are not evaluated; common scale for observed/predicted panels',fontsize=9)
    return _save(fig,path,dpi,f'Validation sample map for target year {year}, in the common projected CRS. All {len(target)} saved sample locations shown; blank is not zero. No spatial interpolation. Observed and predicted panels share limits; residual limits symmetric. CRS: '+grid['crs'])


def _forest_model(model):
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    return isinstance(model, (RandomForestRegressor, ExtraTreesRegressor))


def _stages(model, values):
    if _forest_model(model):
        count = len(model.estimators_)
        checkpoints = {1, count} | {v for v in (8,16,32,50,64,100,128,150,200,256,300,512) if v<=count}
        total = np.zeros(len(values), dtype=float)
        for number, tree in enumerate(model.estimators_, 1):
            total += tree.predict(values)
            if number in checkpoints:
                yield number, total/number
    else:
        yield from enumerate(model.staged_predict(values), 1)


def _learning_curve(model,train,val,target_parameter,path,dpi):
    forest = _forest_model(model)
    if not hasattr(model,'staged_predict') and not forest:
        return None,[]
    center,scale=target_parameter['center'],target_parameter['scale']
    x,y=train
    vx,vy=val
    rows=[]
    for (iteration,tp),(validation_iteration,vp) in zip(_stages(model,x),_stages(model,vx)):
        if iteration != validation_iteration:
            raise ValueError('Mismatched training/validation curve stages')
        train_error=tp*scale+center-y
        val_error=vp*scale+center-vy
        rows.append(dict(iteration=iteration,train_mse=float(np.mean(train_error**2)),
            validation_mse=float(np.mean(val_error**2)),train_rmse=float(np.sqrt(np.mean(train_error**2))),
            validation_rmse=float(np.sqrt(np.mean(val_error**2)))))
        if forest:
            rows[-1]['n_trees'] = iteration
    if not rows:
        return None,[]
    with path.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    ax.plot([r['iteration'] for r in rows],[r['train_rmse'] for r in rows],label='Training sample',color='#256e91')
    ax.plot([r['iteration'] for r in rows],[r['validation_rmse'] for r in rows],label='Validation sample',color='#b05d23')
    ax.set(xlabel='Number of trees' if forest else 'Boosting iteration',ylabel='RMSE (original target scale)',
           title='Error as trees accumulate' if forest else 'Error over boosting iterations')
    ax.legend()
    ax.grid(alpha=.2)
    caption = ('Training and validation RMSE from cumulative averages of the first N saved forest trees; all saved sample points. '
               'Not epochs, OOB estimates or wall-clock history; no refitting or selection of tree count.' if forest else
               'Training and validation RMSE at each fitted boosting iteration, reconstructed from the saved estimator on the original saved samples. This is not wall-clock history and does not retrain or select a new stopping iteration.')
    return _save(fig,path,dpi,caption),rows


def _epoch_curve(artifact, path, dpi):
    rows = artifact['training_history']
    selected = artifact['training_selection']['selected_epoch']
    if (not rows or [r['epoch'] for r in rows] != list(range(1, len(rows)+1))
            or not 1 <= selected <= len(rows)
            or not np.isfinite([[r['train_rmse'], r['validation_rmse']] for r in rows]).all()
            or selected != 1+int(np.argmin([r['validation_rmse'] for r in rows]))):
        raise ValueError('Invalid recorded epoch history or selection')
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    ax.plot([r['epoch'] for r in rows], [r['train_rmse'] for r in rows], label='Training sample', color='#256e91')
    ax.plot([r['epoch'] for r in rows], [r['validation_rmse'] for r in rows], label='Validation sample', color='#b05d23')
    ax.axvline(selected, color='#6b596b', linestyle='--', label=f'Saved epoch {selected}')
    ax.set(xlabel='Epoch', ylabel='RMSE (original target scale)', title=artifact.get('training_curve_title', 'MLP error recorded during training'))
    ax.legend()
    ax.grid(alpha=.2)
    caption = ('RMSE recorded after each epoch on all frozen training/validation samples. '
        'The saved model is the minimum-validation-RMSE epoch, which can precede the last epoch. '
        'Validation is used for development and stopping; this is not an external test.')
    return _save(fig, path, dpi, caption), rows


def export_model(model_dir, dpi=600):
    """Export from saved artifacts only: no retraining, target raster access or test use."""
    from src.services.temporal_service import scale_values
    model_dir=Path(model_dir)
    if not 72<=dpi<=1200:
        raise ValueError('DPI must be between 72 and 1200')
    neural = (model_dir/'model.json').exists() and (model_dir/'model.pt').exists()
    artifact=json.loads((model_dir/'model.json').read_text(encoding='utf-8')) if neural else joblib.load(model_dir/'model.joblib')
    if artifact.get('format') not in ('raster_fusion_temporal_v1', 'raster_fusion_temporal_spatial_v1', 'raster_fusion_temporal_convolutional_v1', 'raster_fusion_temporal_dense_v1'):
        raise ValueError('Expected a temporal model')
    if neural and not artifact.get('training_history'):
        raise ValueError('Neural checkpoint must include recorded epoch history')
    figures=model_dir/'figures'
    figures.mkdir(exist_ok=True)
    with np.load(model_dir.parent/'samples.npz') as samples:
        train=samples['train']
        validation=samples['validation']
        positions=samples['validation_positions']
        years=samples['validation_years']
    prediction=np.load(model_dir/'validation_predictions.npy').astype(float)
    target=validation[:,-1].astype(float)
    if prediction.shape!=target.shape or not np.isfinite(prediction).all():
        raise ValueError('Saved validation predictions do not match validation samples')
    metrics=json.loads((model_dir/'metrics.json').read_text(encoding='utf-8'))['validation_original_units']
    caption=f"Validation · {model_dir.name} · horizon {artifact['horizon_years']} year(s)\nRMSE = {_number(metrics['rmse'])}; MAE = {_number(metrics['mae'])}; R² = {_number(metrics['r2'])}"
    exports=[]
    with plt.rc_context(STYLE),threadpool_limits(limits=artifact['cpu_threads']):
        exports.append(_fit_plot(target,prediction,figures/'observed_predicted.png',dpi,caption))
        positive=target>0
        if positive.any():
            exports.append(_fit_plot(target[positive],prediction[positive],figures/'observed_predicted_positive.png',dpi,
                f'Validation locations with observed target > 0\n{model_dir.name} · horizon {artifact["horizon_years"]} year(s)'))
        exports.append(_diagnostics(target,prediction,figures/'errors.png',dpi))
        for year in np.unique(years):
            mask=years==year
            exports.append(_spatial_map(target[mask],prediction[mask],positions[mask],int(year),artifact['grid'],
                figures/f'validation_map_{year}.png',dpi))
        if artifact.get('training_history'):
            curve, history = _epoch_curve(artifact, figures/'training_curve.png', dpi)
            curve_index = artifact['training_selection']['selected_epoch']-1
        else:
            curve,history=_learning_curve(artifact['model'],
                (scale_values(train[:,:-1],artifact['inputs_normalization']),train[:,-1].astype(float)),
                (scale_values(validation[:,:-1],artifact['inputs_normalization']),target),
                artifact['target_normalization'],figures/'training_curve.png',dpi)
            curve_index = -1
        if curve:
            exports.append(curve)
            expected=metrics['rmse']
            if not np.isclose(history[curve_index]['validation_rmse'],expected,rtol=1e-5,atol=1e-6):
                raise ValueError('Saved-epoch/stage training-curve value differs from saved validation RMSE')
    # Preserve the actual numerical points underlying maps and error plots.
    with (figures/'validation_points.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['target_year','row','column','observed','predicted','residual_predicted_minus_observed'])
        writer.writerows((int(year),int(pos[0]),int(pos[1]),float(obs),float(pred),float(pred-obs))
                         for year,pos,obs,pred in zip(years,positions,target,prediction))
    manifest=dict(created=datetime.now(timezone.utc).isoformat(),model=str(model_dir/('model.pt' if neural else 'model.joblib')),
        scope='saved validation samples; not external test or full-raster predictions',target_scale='original',
        residual_convention='prediction minus observation',dpi=dpi,figures=exports,
        training_curve=('recorded epoch RMSE; saved model at selected minimum, not necessarily last epoch' if artifact.get('training_history') else ('cumulative means of saved forest trees on saved samples' if _forest_model(artifact['model']) else 'boosting stages reconstructed on saved samples')) if history else 'not applicable: estimator has no staged/epoch training history',
        grid=artifact['grid'])
    (figures/'figures.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (figures/'LEIA_ME.md').write_text(
        '# Figuras de validação\n\nPNGs a '+str(dpi)+' dpi, com dados do piloto e valores na escala original do target.\n\n'
        'Os mapas mostram somente os locais amostrados para validação. Áreas em branco não significam ausência do fenômeno. '
        'O resíduo nestas figuras é previsão menos observação. Os painéis observado/previsto usam a mesma escala.\n\n'
        'As curvas do Gradient Boosting são reconstruídas por iteração a partir do modelo e das amostras salvas; não há novo treinamento. '
        'Para Random Forest e Extra Trees, as curvas usam a média cumulativa das primeiras N árvores salvas; não são épocas nem métricas OOB. '
        'Para MLP, os erros são registrados em cada época; a linha vertical marca a época salva de menor RMSE na validação, que pode anteceder a última. '
        'Ridge não tem curva por épocas.\n\n'
        'Os valores dos gráficos e mapas estão em validation_points.csv; quando aplicável, training_curve.csv guarda os erros por iteração. '
        'figures.json registra dimensões, resolução, CRS e legendas. '
        'As figuras usam rótulos em inglês para facilitar a composição de um artigo. '
        'São resultados de validação do piloto, não evidência de teste externo independente.\n',encoding='utf-8')
    return exports


def export_comparison(run,dpi=600):
    run=Path(run)
    with (run/'comparison.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    exports=[]
    with plt.rc_context(STYLE):
        for horizon in sorted({r['horizon'] for r in rows}):
            subset=[r for r in rows if r['horizon']==horizon]
            if len(subset)>4:
                subset=sorted(subset,key=lambda r:float(r['rmse']))
                fig,axes=plt.subplots(1,2,figsize=(12,max(4.5,.45*len(subset)+1.5)),layout='constrained',sharey=True)
                labels=[r['model'].replace('_',' ') for r in subset]
                colors=['#74818a' if r.get('origin')=='reference' else '#256e91' for r in subset]
                for ax,metric in zip(axes,['rmse','mae']):
                    ax.barh(np.arange(len(subset)),[float(r[metric]) for r in subset],color=colors,height=.65)
                    ax.set(xlabel=metric.upper()+' (original target scale)',title='Validation '+metric.upper(),
                           yticks=np.arange(len(subset)),yticklabels=labels)
                    ax.set_xlim(left=0)
                axes[0].invert_yaxis()
                axes[0].axvline(float(subset[0]['baseline_zero_rmse']),color='#a13536',linestyle='--',label='Always-zero baseline')
                axes[0].set_xlim(0,max(float(subset[0]['baseline_zero_rmse']),max(float(r['rmse']) for r in subset),1e-12)*1.08)
            else:
                fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
                for ax,metric in zip(axes,['rmse','mae']):
                    ax.bar([r['model'].replace('_','\n') for r in subset],[float(r[metric]) for r in subset],color='#256e91',width=.55)
                    ax.set(ylabel=metric.upper()+' (original target scale)',title='Validation '+metric.upper())
                    ax.set_ylim(bottom=0)
                axes[0].axhline(float(subset[0]['baseline_zero_rmse']),color='#a13536',linestyle='--',label='Always-zero baseline')
                axes[0].set_ylim(0,max(float(subset[0]['baseline_zero_rmse']),max(float(r['rmse']) for r in subset),1e-12)*1.08)
            axes[0].legend()
            fig.suptitle(f'Validation comparison · horizon {horizon} year(s)')
            fig.supxlabel('Point estimates on saved validation samples; no uncertainty intervals',fontsize=9)
            exports.append(_save(fig,run/'figures'/f'comparison_horizon_{horizon}.png',dpi,
                'Validation model comparison within one horizon; sampled point estimates with no confidence intervals.'))
    return exports


def export_run(run,dpi=600):
    run=Path(run)
    results={}
    checkpoints = list(run.glob('horizon_*/*/model.joblib'))
    checkpoints += list(run.glob('horizon_*/*/model.pt'))
    checkpoints += list(run.glob('references/horizon_*/*/model.joblib'))
    checkpoints += list(run.glob('horizon_*/variants/*/*/model.joblib'))
    checkpoints += list(run.glob('references/horizon_*/*/*/model.joblib'))
    for checkpoint in sorted(checkpoints):
        results[str(checkpoint.parent)]=export_model(checkpoint.parent,dpi)
    if not results:
        raise ValueError('No saved temporal model found in this run')
    if (run/'comparison.csv').exists():
        results['comparison']=export_comparison(run,dpi)
    return results


def export_raster_png(raster_path,png_path=None,dpi=600,max_dimension=4000):
    """Cartographic rendering of a complete GeoTIFF, not a claim of finer data resolution."""
    import rasterio
    from rasterio.enums import Resampling
    raster_path=Path(raster_path)
    png_path=Path(png_path) if png_path else raster_path.with_suffix('.png')
    if png_path.resolve()==raster_path.resolve() or png_path.suffix.lower()!='.png':
        raise ValueError('PNG output must not overwrite the source raster')
    with rasterio.open(raster_path) as ds:
        if ds.count!=1 or ds.crs is None:
            raise ValueError('Require a single-band georeferenced raster')
        if ds.transform.b or ds.transform.d:
            raise ValueError('Rotated grids must be warped to north-up for this renderer')
        ratio=min(1.,max_dimension/max(ds.width,ds.height))
        shape=(max(1,round(ds.height*ratio)),max(1,round(ds.width*ratio)))
        array=ds.read(1,out_shape=shape,masked=True,resampling=Resampling.nearest)
        array=np.ma.masked_invalid(array)
        bounds=ds.bounds
        units=ds.crs.linear_units if ds.crs.is_projected else 'degrees'
        crs=str(ds.crs)
        original_shape=(ds.height,ds.width)
    if array.count()==0:
        raise ValueError('Raster contains no valid pixels')
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
        image=ax.imshow(array,extent=[bounds.left,bounds.right,bounds.bottom,bounds.top],origin='upper',cmap='viridis',interpolation='nearest')
        ax.set(title=raster_path.stem,xlabel=f'X ({units})',ylabel=f'Y ({units})')
        ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0),useOffset=False)
        fig.colorbar(image,ax=ax,label='Raster value (original scale)',shrink=.8)
        fig.supxlabel(f'{crs}; source grid {original_shape[1]} × {original_shape[0]}; blank = NoData',fontsize=9)
        result=_save(fig,png_path,dpi,'Rendering of the complete GeoTIFF extent in its original value scale; nearest-neighbor display sampling. PNG is not georeferenced. Original data resolution is unchanged.')
    result.update(source=str(raster_path),source_shape=original_shape,display_shape=shape,crs=crs)
    png_path.with_suffix('.figure.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return png_path
