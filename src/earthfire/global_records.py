"""Seleção em blocos de países dos ZIPs locais para o piloto global."""
from __future__ import annotations
import zipfile
from pathlib import Path
import pandas as pd
from .ingest import archive_product, csv_chunks, grid_indices, parse_datetime, region_id, validation_masks
from .io import read_json, sha256_file, write_json
from .runtime import Run

COUNTRIES={'brasil_cerrado':'Brazil','argentina_chaco':'Argentina','california':'United_States',
           'canada_boreal':'Canada','portugal':'Portugal','grecia':'Greece','angola':'Angola','sudao':'Sudan',
           'india':'India','tailandia':'Thailand','australia_norte':'Australia','australia_leste':'Australia'}


def extract(run: Run) -> dict:
    folder=run.path/'satellite'; folder.mkdir(exist_ok=True)
    mp=folder/'manifest.json'; manifest=read_json(mp) if mp.exists() else {'files':{},'code_sha256':sha256_file(Path(__file__))}
    if manifest['code_sha256']!=sha256_file(Path(__file__)): raise ValueError('Extrator alterado; use outro run_id.')
    for source in sorted((run.root/run.config['data_dir']).glob('*.zip')):
        product,year=archive_product(source)
        if ' (1)' in source.name or not 2019<=year<=2025: continue
        run.guard(); digest=sha256_file(source); target=folder/(product+'_'+str(year)+'.parquet')
        if source.name in manifest['files']:
            saved=manifest['files'][source.name]
            if saved['source_sha256']!=digest or not target.exists() or saved['output_sha256']!=sha256_file(target):
                raise ValueError('Partição de satélite alterada.')
            continue
        selected=[]; read=0; invalid=0
        with zipfile.ZipFile(source) as archive:
            for country in sorted(set(COUNTRIES.get(r['name'],r.get('country')) for r in run.config['regions'])):
                members=[n for n in archive.namelist() if Path(n).name==f'{product}_{year}_{country}.csv']
                if len(members)!=1: raise ValueError(f'Partição de país ausente/ambígua: {country}/{source.name}')
                regions=[r for r in run.config['regions'] if COUNTRIES.get(r['name'],r.get('country'))==country]
                for chunk in csv_chunks(source,members[0],10000):
                    run.guard(); read+=len(chunk)
                    checks=validation_masks(chunk,product)
                    bad=checks.get('latitude_invalid',False)|checks.get('longitude_invalid',False)
                    invalid+=int(bad.sum())
                    good=chunk.loc[~bad].copy()
                    lat=pd.to_numeric(good.latitude); lon=pd.to_numeric(good.longitude)
                    ii,jj=grid_indices(lat,lon)
                    for region in regions:
                        a,b=grid_indices(pd.Series([region['latitude']]),pd.Series([region['longitude']]))
                        part=good.loc[(ii==int(a.iloc[0]))&(jj==int(b.iloc[0]))].copy()
                        if part.empty: continue
                        _,stamp=parse_datetime(part)
                        valid=stamp.notna()&~validation_masks(part,product)['confidence_invalid']
                        invalid+=int((~valid).sum()); part=part.loc[valid].copy(); part['acquired_utc']=stamp.loc[valid]
                        part['region_id']=region_id(region['latitude'],region['longitude']); part['product']=product
                        part['source_file']=source.name; part['source_member']=members[0]
                        selected.append(part)
        frame=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame({'region_id':pd.Series(dtype=str),'product':pd.Series(dtype=str),'acquired_utc':pd.Series(dtype='datetime64[ns, UTC]')})
        identity=[n for n in frame if n not in ('source_file','source_member')]
        before=len(frame); frame=frame.drop_duplicates(identity); frame.to_parquet(target,index=False)
        manifest['files'][source.name]={'source_sha256':digest,'source_bytes':source.stat().st_size,'rows_read':read,
                                      'invalid':invalid,'selected':len(frame),'exact_duplicates':before-len(frame),'output_sha256':sha256_file(target)}
        write_json(mp,manifest); print(f'Satélite {source.name}: {len(frame)} registros selecionados',flush=True)
    run.stage('global_records','complete',partitions=len(manifest['files']))
    return manifest
