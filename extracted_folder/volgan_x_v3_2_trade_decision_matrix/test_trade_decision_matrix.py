from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from historical_threshold_research import read_option_csvs
from trade_decision_matrix import build_matrix, run_decision_matrix, LIVE_AUTHORIZATION

DATA = Path(__file__).parent / 'data'


@pytest.fixture(scope='module')
def inputs():
    raw, _ = read_option_csvs(DATA)
    return raw


def test_per_contract_ranking_and_reasons(inputs):
    all_pairs, gates, daily = build_matrix(inputs, top_n=5)
    assert len(all_pairs) > 800
    assert len(daily) == 13
    assert all_pairs.candidate_id.is_unique
    assert all_pairs.groupby('entry_date').daily_rank.min().eq(1).all()
    assert all_pairs.live_decision.eq(LIVE_AUTHORIZATION).all()
    assert all_pairs.research_selection.isin({'SELECTED_FOR_REVIEW','REJECTED_BY_SCREEN','NOT_SELECTED_RANK'}).all()
    assert all_pairs.groupby('entry_date').research_selection.apply(
        lambda x: (x == 'SELECTED_FOR_REVIEW').sum()).le(1).all()
    assert gates.groupby('candidate_id').gate.nunique().eq(8).all()
    assert gates[gates.gate == 'G7'].status.eq(LIVE_AUTHORIZATION).all()


def test_cost_gate_explains_no_trade_for_negative_scores(inputs):
    frame, gates, _ = build_matrix(inputs)
    neg = frame[frame.estimated_net_edge_inr <= 0]
    assert len(neg) > 0
    assert neg.research_selection.eq('REJECTED_BY_SCREEN').all()
    assert neg.G1_proxy_edge.eq('NEGATIVE_PROXY').all()
    positive = frame[frame.research_selection == 'SELECTED_FOR_REVIEW']
    assert positive.estimated_net_edge_inr.gt(0).all()
    assert (positive.estimated_net_edge_inr < 250).any()
    assert gates[gates.gate == 'G4'].status.eq('NOT_VALIDATED').all()


def test_model_gates_remain_unvalidated_and_missing(inputs):
    frame, gates, _ = build_matrix(inputs)
    assert frame['persistence_probability'].isna().all()
    assert frame['vae_anomaly_z'].isna().all()
    assert frame['cvar_95_inr'].isna().all()
    assert gates[gates.gate.isin(['G2','G3','G4','G6'])].status.eq('NOT_VALIDATED').all()
    assert gates[gates.gate == 'G5'].status.eq('NOT_AVAILABLE').all()


def test_future_prices_never_used_for_research_rank(inputs):
    baseline, _, _ = build_matrix(inputs)
    modified = inputs.copy()
    modified.loc[modified.TRADE_DATE == pd.Timestamp('2024-01-12'), 'CLOSE'] *= 3
    second, _, _ = build_matrix(modified)
    original = baseline[baseline.entry_date == '2024-01-11'][['candidate_id','research_selection','daily_rank','estimated_net_edge_inr']]
    retest = second[second.entry_date == '2024-01-11'][['candidate_id','research_selection','daily_rank','estimated_net_edge_inr']]
    pd.testing.assert_frame_equal(original.reset_index(drop=True),retest.reset_index(drop=True))


def test_outputs_are_consistent(inputs, tmp_path):
    report = run_decision_matrix(output_dir=tmp_path, raw=inputs, top_n=4)
    assert report['status'] == 'RESEARCH_ONLY'
    assert report['live_authorization'] == LIVE_AUTHORIZATION
    top = pd.read_csv(tmp_path/'trade_decision_matrix.csv')
    all_pairs = pd.read_csv(tmp_path/'trade_decision_matrix_all.csv')
    gates = pd.read_csv(tmp_path/'trade_gate_breakdown.csv')
    assert len(top) == report['matrix_rows']
    assert top.daily_rank.le(4).all()
    assert len(all_pairs) == report['candidate_pairs']
    assert len(gates) == 8*len(top)
    assert (tmp_path/'trade_decision_matrix.html').exists()
    assert (tmp_path/'trade_daily_summary.csv').exists()


def test_invalid_topn_fails_closed(inputs):
    with pytest.raises(ValueError, match='top_n'):
        build_matrix(inputs, top_n=0)
