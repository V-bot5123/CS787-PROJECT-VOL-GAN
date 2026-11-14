import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

import volgan_model_research as m
import import_nse_bhavcopy as adapter
import historical_threshold_research as hist
from volgan_combined import run_trade_decision_agent

DIR=Path(__file__).parent / 'data'


def test_dataset_constructs_actual_masked_iv():
    raw, _ = hist.read_option_csvs(DIR)
    dates, x, mask, spot, stats = m.make_dataset(raw)
    assert len(dates)==13 and x.shape==(13,8,40) and x.dtype==np.float32
    assert 0 < mask.sum() < mask.size
    assert np.all(np.isfinite(x[mask>0]))
    assert np.isnan(x[mask==0]).all()
    assert all(s['observed_cells']==int(mask[i].sum()) for i,s in enumerate(stats))


def test_scaler_train_only_ignore_future():
    X=np.full((5,8,40),.2,dtype='float32'); M=np.ones_like(X)
    a,b=m.fit_scaler(X[:3],M[:3]); X[4]=5.
    x,y=m.fit_scaler(X[:3],M[:3]);
    np.testing.assert_allclose(a,x);np.testing.assert_allclose(b,y)
    zz=m.standardize(X,M,a,b)
    assert zz.shape==X.shape


def test_label_and_gap_are_causal():
    dates=[pd.Timestamp('2024-01-08')+pd.Timedelta(days=i) for i in range(5)]
    dates += [pd.Timestamp('2024-02-01'),pd.Timestamp('2024-06-04')]
    assert m.get_windows(dates,3)==[2,3]
    r=np.zeros((2,8,40),dtype='float32');M=np.ones_like(r)
    r[0,0,0]=.02;r[1,0,0]=.02
    labels,observed=m.persistence_labels(r,M,0)
    assert labels[0,0]==1 and observed[0,0]==1
    r[1,0,0]=-.02
    labels,_=m.persistence_labels(r,M,0)
    assert labels[0,0]==0


def test_legacy_import_requires_underlying():
    raw=pd.DataFrame({'TIMESTAMP':['01-Jan-2024'],'INSTRUMENT':['OPTIDX'],
        'SYMBOL':['NIFTY'],'EXPIRY_DT':['04-Jan-2024'],'STRIKE_PR':[22000],
        'OPTION_TYP':['CE'],'CLOSE':[100.], 'SETTLE_PR':[100.],
        'CONTRACTS':[100],'OPEN_INT':[1000]})
    with pytest.raises(ValueError,match='underlying close'):
        adapter.normalize(raw)
    underlying=pd.Series([21800.],index=[pd.Timestamp('2024-01-01')])
    normalized=adapter.normalize(raw,underlying)
    assert normalized.UNDERLYING_CLOSE.iloc[0]==21800.
    assert normalized.TRADE_DATE.iloc[0]=='2024-01-01'


def test_udiff_import_preserves_board_lot():
    raw=pd.DataFrame({'TradDt':['2025-05-06'], 'TckrSymb':['NIFTY'],
        'FinInstrmTp':['IDO'],'XpryDt':['2025-05-29'],'StrkPric':[24000],
        'OptnTp':['CE'],'ClsPric':[150.], 'SttlmPric':[152.],
        'TtlTradgVol':[140],'OpnIntrst':[2000], 'UndrlygPric':[24100.],
        'NewBrdLotQty':[75]})
    normalized=adapter.normalize(raw)
    assert normalized.LOT_SIZE.iloc[0]==75
    assert normalized.INSTRUMENT.iloc[0]=='OPTIDX'
    raw.NewBrdLotQty=np.nan
    with pytest.raises(ValueError,match='LOT_SIZE'):
        adapter.normalize(raw)


def test_missing_lot_never_guessed_in_transition():
    import historical_threshold_research as h
    with pytest.raises(ValueError):
        h.historical_lot_size(pd.Timestamp('2025-04-01'),pd.Timestamp('2025-04-24'))


def test_cond_missing_history_is_explicit():
    dates=pd.to_datetime(['2024-01-01','2024-01-02','2024-02-01']).tolist()
    spot=np.array([22000.,22100.,22500.]);mu=np.zeros((3,5))
    c=m.rv_cond(dates,spot,2,mu)
    assert c.shape==(9,) and c[5]==-1. and c[6]==-1. and c[-1]==0.


def test_gate_never_executes_despite_validation_flag():
    c={'lots':1,'lot_size':50,'premium':200.,'side':'sell','models_validated':True}
    state=run_trade_decision_agent(c,(0,20),resid_iv=.10,vega_unit=1000.,
       y_hat=.99,anomaly_z=0.,cvar_95=-100.,row_qc_ok=True)
    assert state['decision']=='ABORT'
    assert 'LIVE_DISABLED' in state['abort_reason']


def test_model_pipeline_real_data_reproducible(tmp_path):
    r=m.run_model_research(DIR,tmp_path,vae_epochs=2,pers_epochs=2,gan_steps=2)
    assert r['data']['files']==13 and r['data']['dates']==13
    assert r['chronology']['train_windows']==4
    assert r['chronology']['validation_windows']==2
    assert r['chronology']['holdout_windows']==2
    assert r['vae']['test_metrics']['nmse'] is not None
    assert len(r['vae']['epoch_history'])==2
    assert len(r['persistence']['train']['epoch_history'])==2
    assert len(r['wgan_gp']['step_history'])==2
    assert (tmp_path/'vae_epoch_metrics.csv').exists()
    assert (tmp_path/'persistence_epoch_metrics.csv').exists()
    assert (tmp_path/'gan_step_metrics.csv').exists()
    assert r['wgan_gp']['transition_count']>=3
    assert r['wgan_gp']['rv21_condition_available'] is False
    assert r['thresholds']['THETA_Y']['value'] is None
    assert r['trade_authorization']=='DISABLED' and r['decision']=='ABORT'
    assert (tmp_path/'model_training_report.json').exists()
    assert (tmp_path/'walkforward_threshold_proxy.csv').exists()
    ckpt=torch.load(tmp_path/'experimental_checkpoints'/'vae_research.pt',weights_only=True)
    assert ckpt['status']=='EXPERIMENTAL_UNVALIDATED'
    assert len(r['wgan_gp']['holdout_spot_tail_diagnostics'])==2


def test_invalid_training_steps_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='must all be positive'):
        m.run_model_research(DIR,tmp_path,vae_epochs=0)


def test_nmse_mask_and_zero_baseline():
    y=np.array([[[1.,2.,50.]]]); yhat=np.array([[[2.,0.,-10.]]]); mask=np.array([[[1,1,0]]])
    d=m.masked_reconstruction_metrics(y,yhat,mask)
    assert d['observed_cells']==2
    assert d['mse']==2.5
    assert d['nmse']==1.0

def test_masked_bce_brier_only_observed():
    p=np.array([[[.9,.1,.1]]]);y=np.array([[[1.,0.,1.]]]);masks=np.array([[[1,1,0]]])
    d=m.masked_classifier_metrics(p,y,masks)
    assert d['observed_cells']==2
    assert abs(d['brier']-.01)<1e-8
    assert abs(d['bce']+np.log(.9))<1e-8
