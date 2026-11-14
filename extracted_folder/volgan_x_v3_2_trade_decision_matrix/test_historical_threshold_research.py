import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import historical_threshold_research as h

CSV_DIR = Path(__file__).parent / 'data'


def test_sources_loaded_and_columns_valid():
    frame, overview = h.read_option_csvs(CSV_DIR)
    assert overview['source_files'] == 13
    assert overview['source_rows'] == 19678
    assert len(overview['dates']) == 13
    assert len(frame) == 19678
    assert not frame.duplicated(['TRADE_DATE', 'EXPIRY_DT', 'STRIKE_PR', 'OPTION_TYP']).any()


def test_historical_contract_lots_and_transition_rejection():
    assert h.historical_lot_size(pd.Timestamp('2024-01-15'), pd.Timestamp('2024-01-18')) == 50
    assert h.historical_lot_size(pd.Timestamp('2024-06-04'), pd.Timestamp('2024-06-06')) == 25
    with pytest.raises(ValueError, match='contract master'):
        h.historical_lot_size(pd.Timestamp('2024-04-29'), pd.Timestamp('2024-06-27'))


def test_roundtrip_cost_includes_two_legs():
    assert h.pair_roundtrip_cost(100., 100., 50) > h.pair_roundtrip_cost(100., 100., 25)
    assert h.pair_roundtrip_cost(100., 100., 50) > 80.0


def test_daily_candidate_never_looks_at_next_day(monkeypatch):
    frame, _ = h.read_option_csvs(CSV_DIR)
    d1 = frame[frame.TRADE_DATE == pd.Timestamp('2024-01-08')]
    all_signal = h.create_daily_candidates(frame)
    day_signal = h.create_daily_candidates(d1)
    keys = ['entry_date', 'expiry', 'strike', 'estimated_net_edge_inr', 'side']
    selected = all_signal[all_signal.entry_date == pd.Timestamp('2024-01-08')][keys].reset_index(drop=True)
    pd.testing.assert_frame_equal(selected, day_signal[keys].reset_index(drop=True))


def test_no_next_session_bridge_across_large_gaps():
    frame, _ = h.read_option_csvs(CSV_DIR)
    candidates = h.create_daily_candidates(frame)
    evaluated = h.evaluate_next_session(candidates, frame)
    assert evaluated['entry_date'].max() == pd.Timestamp('2024-01-12')
    assert (evaluated['gap_days'] <= h.MAX_SESSION_GAP_DAYS).all()
    assert (evaluated.next_date < evaluated.expiry).all()
    assert not evaluated.duplicated(['entry_date']).any()


def test_missing_exit_is_not_counted_as_zero():
    frame, _ = h.read_option_csvs(CSV_DIR)
    evaluated = h.evaluate_next_session(h.create_daily_candidates(frame), frame)
    missing = evaluated[~evaluated.exit_observed]
    assert len(missing) > 0
    assert missing.pnl_net_inr.isna().all()


def test_empirical_thresholds_not_trade_approved(tmp_path):
    report = h.run_historical_research(CSV_DIR, tmp_path)
    assert report['decision'] == 'ABORT'
    assert report['trade_authorization'] == 'DISABLED'
    assert report['training_period']['dates'] == 6
    assert report['holdout_period']['dates'] == 4
    assert report['provisional_research_min_edge_inr'] is None
    assert report['volgan_trading_thresholds']['THETA_Y']['value'] is None
    assert report['volgan_trading_thresholds']['TAU_HEALTHY']['value'] is None
    assert report['volgan_trading_thresholds']['MIN_EDGE_INR']['value'] is None
    assert report['volgan_trading_thresholds']['KAPPA']['value'] is None
    assert report['volgan_trading_thresholds']['RISK_BUDGET_INR']['value'] is None
    assert (tmp_path / 'historical_threshold_report.json').exists()
    assert (tmp_path / 'historical_daily_candidates.csv').exists()
    assert (tmp_path / 'historical_threshold_comparison.csv').exists()
    written = json.loads((tmp_path / 'historical_threshold_report.json').read_text())
    assert written['status'] == 'RESEARCH_ONLY'
    assert len(written['gap_exclusions']) == 2


def test_chronological_split_does_not_use_holdout_to_pick_threshold():
    a = [{'min_edge_inr': 0, 'train': {'n_trades': 4, 'sum_net_pnl_inr': 100}},
         {'min_edge_inr': 250, 'train': {'n_trades': 4, 'sum_net_pnl_inr': 80}}]
    selected = max(a, key=lambda row: row['train']['sum_net_pnl_inr'])
    assert selected['min_edge_inr'] == 0


def test_bundled_data_is_auto_detected(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    directory = h.resolve_option_data_dir()
    assert directory == CSV_DIR.resolve()
    raw, overview = h.read_option_csvs()
    assert overview['source_files'] == 13
    assert len(raw) == 19678


def test_missing_data_directory_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        h.read_option_csvs(tmp_path)
    msg = str(exc.value)
    assert 'Searched:' in msg
    assert '--data-dir' in msg
    assert str(tmp_path) in msg
