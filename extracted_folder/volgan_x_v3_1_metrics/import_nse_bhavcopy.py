from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NORM_COLS = ['TRADE_DATE','INSTRUMENT','SYMBOL','EXPIRY_DT','STRIKE_PR',
             'OPTION_TYP','CLOSE','SETTLE_PR','CONTRACTS','OPEN_INT',
             'VAL_INLAKH','UNDERLYING_CLOSE','LOT_SIZE']


def read_local_source(path:Path) -> pd.DataFrame:
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as archive:
            files=[x for x in archive.namelist() if x.lower().endswith('.csv') and not x.startswith('__MACOSX')]
            if len(files)!=1:
                raise ValueError(f'{path.name}: expected exactly one CSV in ZIP (found {len(files)})')
            with archive.open(files[0]) as fd:
                return pd.read_csv(fd,low_memory=False)
    if path.suffix.lower()=='.csv':
        return pd.read_csv(path,low_memory=False)
    raise ValueError(f'Unsupported input: {path}')


def read_underlying(source:Path|None):
    if source is None:
        return None
    d=pd.read_csv(source)
    if not {'TRADE_DATE','UNDERLYING_CLOSE'}.issubset(d):
        raise ValueError('Underlying index CSV requires TRADE_DATE,UNDERLYING_CLOSE')
    d=d[['TRADE_DATE','UNDERLYING_CLOSE']].copy()
    d.TRADE_DATE=pd.to_datetime(d.TRADE_DATE)
    d.UNDERLYING_CLOSE=pd.to_numeric(d.UNDERLYING_CLOSE,errors='coerce')
    if d.TRADE_DATE.duplicated().any() or d.UNDERLYING_CLOSE.isna().any() or (d.UNDERLYING_CLOSE<=0).any():
        raise ValueError('Underlying prices contain duplicate days, missing or nonpositive values.')
    return d.set_index('TRADE_DATE').UNDERLYING_CLOSE


def normalize(source: pd.DataFrame, underlying=None)->pd.DataFrame:
    source=source.rename(columns=lambda k:str(k).strip())
    udiff='FinInstrmTp' in source
    if udiff:
        fields={'TradDt','TckrSymb','FinInstrmTp','XpryDt','StrkPric','OptnTp',
                'ClsPric','SttlmPric','TtlTradgVol','OpnIntrst','UndrlygPric','NewBrdLotQty'}
        missing=fields-set(source)
        if missing:raise ValueError('UDiFF missing fields: '+str(sorted(missing)))
        df=pd.DataFrame({
            'TRADE_DATE':source.TradDt,
            'INSTRUMENT':source.FinInstrmTp.map({'IDO':'OPTIDX'}),
            'SYMBOL':source.TckrSymb,
            'EXPIRY_DT':source.XpryDt,
            'STRIKE_PR':source.StrkPric,'OPTION_TYP':source.OptnTp,
            'CLOSE':source.ClsPric,'SETTLE_PR':source.SttlmPric,
            'CONTRACTS':source.TtlTradgVol,'OPEN_INT':source.OpnIntrst,
            'VAL_INLAKH':np.nan,
            'UNDERLYING_CLOSE':source.UndrlygPric,'LOT_SIZE':source.NewBrdLotQty,
        })
    else:
        source=source.rename(columns=lambda k:str(k).strip().upper())
        fields={'INSTRUMENT','SYMBOL','EXPIRY_DT','STRIKE_PR','OPTION_TYP',
                'CLOSE','SETTLE_PR','CONTRACTS','OPEN_INT'}
        if fields-set(source):raise ValueError('Legacy source missing: '+str(sorted(fields-set(source))))
        if 'TIMESTAMP' in source:
            trading_date=source.TIMESTAMP
        elif 'TRADE_DATE' in source:
            trading_date=source.TRADE_DATE
        else:
            raise ValueError('Legacy source missing TIMESTAMP or TRADE_DATE')
        df=source.copy()
        df['TRADE_DATE']=trading_date
        if 'UNDERLYING_CLOSE' not in df:df['UNDERLYING_CLOSE']=np.nan
        if 'LOT_SIZE' not in df:df['LOT_SIZE']=np.nan
        if 'VAL_INLAKH' not in df:df['VAL_INLAKH']=0.
    df=df.loc[(df['INSTRUMENT']=='OPTIDX')&(df['SYMBOL'].astype(str).str.strip()=='NIFTY')&
              df.OPTION_TYP.isin(['CE','PE']),NORM_COLS].copy()
    if df.empty:raise ValueError('No NIFTY OPTIDX CE/PE contracts found')
    df.TRADE_DATE=pd.to_datetime(df.TRADE_DATE,errors='raise')
    df.EXPIRY_DT=pd.to_datetime(df.EXPIRY_DT,errors='raise')
    if df.TRADE_DATE.nunique()!=1:raise ValueError('One snapshot must contain only one trade date')
    for col in ['STRIKE_PR','CLOSE','SETTLE_PR','CONTRACTS','OPEN_INT','UNDERLYING_CLOSE','LOT_SIZE','VAL_INLAKH']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    if underlying is not None:
        day=df.TRADE_DATE.iloc[0]
        if day not in underlying.index:raise ValueError(f'{day.date()} missing from underlying index CSV')
        df['UNDERLYING_CLOSE']=float(underlying.loc[day])
    if df.UNDERLYING_CLOSE.isna().any() or (df.UNDERLYING_CLOSE<=0).any():
        raise ValueError('No verified underlying close. Supply --underlying-csv with a daily NIFTY index close.')
    if df[['STRIKE_PR','CLOSE','SETTLE_PR','CONTRACTS','OPEN_INT']].isna().any().any():
        raise ValueError('Missing critical numeric values')
    if (df.STRIKE_PR<=0).any() or (df.CLOSE<0).any() or (df.CONTRACTS<0).any() or (df.OPEN_INT<0).any():
        raise ValueError('Invalid strike, premium, volume or open interest')
    if df.TRADE_DATE.iloc[0]>=pd.Timestamp('2024-04-26'):
        if df.LOT_SIZE.isna().any() or (df.LOT_SIZE<=0).any():
            raise ValueError('Post-transition contract-specific LOT_SIZE absent; cannot infer lot size.')
    if df.duplicated(['TRADE_DATE','EXPIRY_DT','STRIKE_PR','OPTION_TYP']).any():
        raise ValueError('Duplicate NIFTY contract row')
    df['TRADE_DATE']=df.TRADE_DATE.dt.strftime('%Y-%m-%d')
    df['EXPIRY_DT']=df.EXPIRY_DT.dt.strftime('%d-%b-%Y')
    return df.reset_index(drop=True)


def convert_folder(input_dir:Path,output_dir:Path,underlying_file:Path|None=None):
    under=read_underlying(underlying_file)
    files=sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.csv','.zip') and (underlying_file is None or p.resolve()!=underlying_file.resolve()))
    if not files:raise FileNotFoundError(f'No CSV or ZIP in {input_dir}')
    output_dir.mkdir(parents=True,exist_ok=True)
    result=[]
    for f in files:
        frame=normalize(read_local_source(f),under)
        dest=output_dir / ('nifty_options_'+frame.TRADE_DATE.iloc[0]+'.csv')
        if dest.exists():raise FileExistsError(f'{dest} already exists; refusing overwrite')
        frame.to_csv(dest,index=False)
        result.append((f.name,dest.name,len(frame)))
    return result


def main():
    p=argparse.ArgumentParser(description='Normalize already-downloaded NSE F&O ZIP/CSV files (offline, strict)')
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--underlying-csv',type=Path,default=None)
    a=p.parse_args()
    for name,dest,n in convert_folder(a.input_dir,a.output_dir,a.underlying_csv):
        print(f'{name} -> {dest} ({n} rows)')
    print('Import complete. Review the output before research; no trades are authorized.')

if __name__=='__main__':main()
