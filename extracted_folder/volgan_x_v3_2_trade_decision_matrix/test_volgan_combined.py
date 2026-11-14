import math
import numpy as np
import pytest
import torch
import volgan_combined as v


def test_black76_iv_roundtrip():
    for option in ('CE', 'PE'):
        f, k, tau, vol = 24500.0, 24400.0, 30 / 365, .22
        price = v.black76_price(f, k, tau, vol, .065, option)
        recovered = v.invert_iv_black76(price, f, k, tau, .065, option)
        assert abs(recovered - vol) < 1e-4


def test_black76_put_call_parity():
    f, k, t, r, sigma = 24500, 24700, 21 / 365, .065, .16
    diff = v.black76_price(f,k,t,sigma,r,'CE') - v.black76_price(f,k,t,sigma,r,'PE')
    assert abs(diff - math.exp(-r*t)*(f-k)) < 1e-6


def test_vg_martingale_and_fft():
    f, t, r = 24500.0, 30 / 365, .065
    assert v.verify_vg_martingale(f, t, .15, .2, -.10)
    strikes = np.array([23500, 24500, 25500.])
    prices = v.carrmadan_fft_call_prices(f,t,r,.15,.2,-.10,strikes)
    assert prices.shape == (3,)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0)
    assert np.all(np.diff(prices) <= 0)


def test_mock_market_surface_shape_and_provenance():
    x, mask = v.get_market_surface()
    assert x.shape == mask.shape == (8, 40)
    assert np.all(np.isfinite(x))
    assert 0 < np.count_nonzero(mask) < 8*40
    with pytest.raises(NotImplementedError):
        v.get_market_surface(mock=False)


def test_cnn_lstm_and_generator_shapes():
    torch.manual_seed(1)
    logits, attention = v.CellPersistenceNet()(torch.randn(1, 3, 5, 8, 40))
    assert logits.shape == (1, 8, 40)
    assert attention.shape == (1, 8, 40, 3)
    assert torch.allclose(attention.sum(-1), torch.ones(1, 8, 40), atol=1e-5)
    ds, div = v.VolGANGenerator()(torch.randn(4, 32), torch.zeros(4, 8))
    assert ds.shape == (4,)
    assert div.shape == (4, 8, 40)


def test_arbitrage_penalty_constant_vol():
    x = torch.full((2, 8, 40), .15)
    cal, fly = v.compute_arbitrage_penalties(x)
    assert torch.allclose(cal, torch.tensor(0.))
    assert torch.allclose(fly, torch.tensor(0.), atol=1e-6)


def test_scenario_repricing_cell_and_friction():
    f, k, tau, r = 24500., 24500., 14 / 365, .065
    x = np.full((8,40), .15)
    p0 = v.black76_price(f,k,tau,.15,r)
    legs = [dict(type='CE',K=k,F0=f,tau=tau,qty=-1,lot=25,v0=p0,tenor_idx=1,strike_idx=19)]
    pnl = v.scenario_pnl(legs,x,np.zeros(2),np.zeros((2,8,40)),h_days=0)
    assert np.allclose(pnl, -40., atol=1e-6)
    dlogiv = np.zeros((2,8,40))
    dlogiv[1,1,19] = .25
    pnl_changed = v.scenario_pnl(legs,x,np.zeros(2),dlogiv,h_days=0)
    assert pnl_changed[1] < pnl_changed[0]


def test_demo_fails_closed_on_untrained_models():
    contract = dict(lots=1,lot_size=25,premium=150.,side='sell',models_validated=False)
    result = v.run_trade_decision_agent(contract,(0,20),.05,100.,.99,0.,-10.,True)
    assert result['decision'] == 'ABORT'
    assert result['abort_reason'].startswith('G0_MODEL')


def test_conservative_cvar():
    values = np.arange(-100.,100.)
    assert v.calculate_conservative_cvar(values) == np.mean(values[:10])


def test_cannot_authorize_when_models_validated_flag_is_forged():
    contract = dict(lots=1, lot_size=50, premium=150., side='sell', models_validated=True)
    result = v.run_trade_decision_agent(contract, (0, 20), .05, 100., .99, 0., -10., True)
    assert result['decision'] == 'ABORT'
    assert result['abort_reason'].startswith('G0_LIVE_DISABLED')
