from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from historical_threshold_research import (read_option_csvs, all_parity_candidates,
                                           evaluate_next_session)

DISPLAY_DEFAULTS = {'THETA_Y': 0.70, 'TAU_HEALTHY': 2.50,
                    'MIN_EDGE_INR': 250.0, 'KAPPA': 3.0,
                    'RISK_BUDGET_INR': 25000.0}
LIVE_AUTHORIZATION = 'BLOCKED_LIVE'


def build_matrix(raw: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if top_n < 1:
        raise ValueError('top_n must be >=1')
    all_candidates = all_parity_candidates(raw)
    if all_candidates.empty:
        raise ValueError('No liquid, near-money call-put pairs available for screening.')

    x = all_candidates.sort_values(['entry_date', 'estimated_net_edge_inr', 'strike'],
                                   ascending=[True, False, True]).copy().reset_index(drop=True)
    x['daily_rank'] = x.groupby('entry_date').cumcount() + 1
    x['candidate_id'] = (x['entry_date'].dt.strftime('%Y%m%d') + '_' +
                         x['expiry'].dt.strftime('%Y%m%d') + '_' +
                         x['strike'].map(lambda v: f'{v:g}'))
    x['instrument'] = 'NIFTY INDEX CE/PE PAIR'
    x['dte_days'] = (x['expiry'] - x['entry_date']).dt.days
    x['proxy_score_points'] = np.abs(x['residual_points'])
    x['cost_coverage_x'] = x['gross_discrepancy_inr'] / x['estimated_cost_inr']
    x['side_description'] = np.where(x['side'] == -1, 'SELL CE / BUY PE (proxy)',
                                     'BUY CE / SELL PE (proxy)')

    x['G0_data'] = 'PASS_FILTERED'
    x['G1_proxy_edge'] = np.where(x['estimated_net_edge_inr'] > 0,
                                   'POSITIVE_PROXY', 'NEGATIVE_PROXY')
    x['G2_persistence'] = 'NOT_VALIDATED'
    x['G3_anomaly'] = 'NOT_VALIDATED'
    x['G4_min_edge'] = 'NOT_VALIDATED'
    x['G5_tail_CVaR'] = 'NOT_AVAILABLE'
    x['G6_risk_policy'] = 'NOT_VALIDATED'
    x['G7_live_execution'] = LIVE_AUTHORIZATION
    x['persistence_probability'] = np.nan
    x['vae_anomaly_z'] = np.nan
    x['cvar_95_inr'] = np.nan

    x['research_selection'] = 'NOT_SELECTED_RANK'
    x.loc[x['estimated_net_edge_inr'] <= 0, 'research_selection'] = 'REJECTED_BY_SCREEN'
    best = (x['daily_rank'] == 1) & (x['estimated_net_edge_inr'] > 0)
    x.loc[best, 'research_selection'] = 'SELECTED_FOR_REVIEW'
    x['decision_reason'] = np.select(
        [x['research_selection'].eq('SELECTED_FOR_REVIEW'),
         x['research_selection'].eq('REJECTED_BY_SCREEN')],
        ['Top-ranked pair for its date and POSITIVE estimated after-cost parity discrepancy; research review only.',
         'Estimated gross parity discrepancy does not exceed modeled round-trip cost; research screen rejects.'],
        default='Positive modeled proxy edge, but another same-day pair ranks higher; not selected for review.')
    x['live_reason'] = ('No verified executable quotes, contract-level validated persistence/anomaly/CVaR, '
                        'or independently validated trading thresholds; execution explicitly disabled.')

    observed = evaluate_next_session(all_candidates.drop_duplicates('entry_date', keep='first'), raw)
    if not observed.empty:
        observations = observed[['entry_date', 'expiry', 'strike', 'result',
                                 'exit_observed', 'pnl_net_inr']].copy()
        x = x.merge(observations, on=['entry_date','expiry','strike'], how='left', validate='one_to_one')
    else:
        x['result'] = np.nan
        x['exit_observed'] = np.nan
        x['pnl_net_inr'] = np.nan
    x['posthoc_exit_status'] = np.where(x['daily_rank'] != 1, 'NOT_EVALUATED_OTHER_RANK',
                                       x['result'].fillna('NOT_EVALUATED_GAP_OR_EXPIRY'))
    x['live_decision'] = LIVE_AUTHORIZATION

    gate_rows = []
    for r in x.itertuples(index=False):
        gates = [
            ('G0', 'CLOSE/Liquidity input filter', 'BOTH LEGS',
             'CLOSE>=₹1; volume>=10; OI>=100; DTE 3–45; |moneyness|<=2.5%',
             'PASS_FILTERED', 'Paired CE and PE closes survived input filters; they are NOT executable bid/ask quotes.'),
            ('G1', 'Research parity proxy after costs', round(r.estimated_net_edge_inr,2),
             '>₹0 is a descriptive screen, NOT a validated profitability threshold',
             'POSITIVE_PROXY' if r.estimated_net_edge_inr > 0 else 'NEGATIVE_PROXY',
             f'Gross ₹{r.gross_discrepancy_inr:,.2f} - modeled costs ₹{r.estimated_cost_inr:,.2f}. Comparison is illustrative.'),
            ('G2', 'CNN-LSTM persistence', None, f'THETA_Y={DISPLAY_DEFAULTS["THETA_Y"]:.2f} (unvalidated default)',
             'NOT_VALIDATED', 'No independently calibrated contract-specific persistence probability; cannot pass this gate.'),
            ('G3', 'VAE anomaly / regime', None, f'TAU_HEALTHY={DISPLAY_DEFAULTS["TAU_HEALTHY"]:.2f} (unvalidated default)',
             'NOT_VALIDATED', 'No independently calibrated contract-specific anomaly z-score; cannot pass this gate.'),
            ('G4', 'Original VolGAN minimum edge', round(r.estimated_net_edge_inr,2),
             f'MIN_EDGE_INR=₹{DISPLAY_DEFAULTS["MIN_EDGE_INR"]:.0f} (UNVALIDATED; different proxy metric)',
             'NOT_VALIDATED', 'Parity-proxy edge is NOT VolGAN vega-weighted residual edge and must not be compared as a live entry rule.'),
            ('G5', 'GAN-generated tail risk', None, 'Calibrated contract-level 95% CVaR required',
             'NOT_AVAILABLE', 'No validated scenario loss distribution or contract-specific 95% CVaR.'),
            ('G6', 'Risk mandate / capital limits', None,
             f'KAPPA={DISPLAY_DEFAULTS["KAPPA"]:.1f}; risk budget ₹{DISPLAY_DEFAULTS["RISK_BUDGET_INR"]:,.0f} (defaults only)',
             'NOT_VALIDATED', 'No approved capital/risk mandate or validated CVaR to compare with a risk budget.'),
            ('G7', 'Live authorization', 'DISABLED', 'Must be enabled in separately validated production system',
             LIVE_AUTHORIZATION, 'Research build explicitly forbids live execution regardless of research-screen outcome.'),
        ]
        for code, gate, value, comparison, status, reason in gates:
            gate_rows.append({'candidate_id':r.candidate_id, 'entry_date':r.entry_date.strftime('%Y-%m-%d'),
                              'expiry':r.expiry.strftime('%Y-%m-%d'), 'strike':r.strike,
                              'daily_rank':int(r.daily_rank), 'gate':code, 'check':gate,
                              'observed_value':value,'criterion':comparison,'status':status,
                              'reason':reason,'research_selection':r.research_selection,
                              'live_decision':LIVE_AUTHORIZATION})
    gates = pd.DataFrame(gate_rows)
    x['entry_date'] = x['entry_date'].dt.strftime('%Y-%m-%d')
    x['expiry'] = x['expiry'].dt.strftime('%Y-%m-%d')
    x = x.drop(columns=['result','exit_observed'], errors='ignore')
    summary = x[x['daily_rank'] == 1].copy()
    return x, gates, summary


def _table_html(frame: pd.DataFrame, title: str) -> str:
    return f'<h2>{html.escape(title)}</h2>\n' + frame.to_html(index=False, escape=True,
                                                          classes='matrix', na_rep='—', border=0)


def run_decision_matrix(data_dir=None, output_dir='research_outputs', top_n=5,
                        raw: pd.DataFrame | None = None, overview: dict | None = None) -> dict:
    if raw is None:
        raw, overview = read_option_csvs(data_dir)
    x, gates, summary = build_matrix(raw, top_n=top_n)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compact_cols = ['candidate_id','entry_date','expiry','strike','daily_rank','instrument',
                    'side_description','dte_days','spot_entry','call_entry','put_entry','lot',
                    'residual_points','proxy_score_points','gross_discrepancy_inr',
                    'estimated_cost_inr','estimated_net_edge_inr','cost_coverage_x',
                    'G0_data','G1_proxy_edge','G2_persistence','G3_anomaly','G4_min_edge',
                    'G5_tail_CVaR','G6_risk_policy','G7_live_execution',
                    'persistence_probability','vae_anomaly_z','cvar_95_inr',
                    'research_selection','decision_reason','live_decision','live_reason',
                    'posthoc_exit_status','pnl_net_inr']
    x = x[compact_cols]
    x.to_csv(out/'trade_decision_matrix_all.csv',index=False,na_rep='')
    top = x[x.daily_rank <= top_n].copy()
    top.to_csv(out/'trade_decision_matrix.csv',index=False,na_rep='')
    gates[gates.daily_rank <= top_n].to_csv(out/'trade_gate_breakdown.csv',index=False,na_rep='')
    summary = x[x.daily_rank == 1].copy()
    summary.to_csv(out/'trade_daily_summary.csv',index=False,na_rep='')
    n_selected = int((x.research_selection == 'SELECTED_FOR_REVIEW').sum())
    report = {
        'status':'RESEARCH_ONLY','live_authorization':LIVE_AUTHORIZATION,
        'screen_name':'Daily same-expiry NIFTY CE/PE parity discrepancy (NOT VolGAN model trade signal)',
        'dates':int(x.entry_date.nunique()), 'candidate_pairs':int(len(x)),
        'matrix_top_n_per_date':top_n, 'matrix_rows':int(len(top)),
        'selected_for_research_review':n_selected,
        'rejected_by_proxy_cost_screen':int((x.research_selection=='REJECTED_BY_SCREEN').sum()),
        'passed_proxy_but_out_ranked':int((x.research_selection=='NOT_SELECTED_RANK').sum()),
        'cost_screen':'Estimated absolute parity residual in points × lot minus illustrative paired option costs > 0; NOT a live threshold',
        'example_defaults_NOT_VALIDATED':DISPLAY_DEFAULTS,
        'gates':{
            'G0':'Entry data filters (daily CLOSE, volume, OI, DTE, moneyness)',
            'G1':'Illustrative parity proxy net of modeled friction',
            'G2':'CNN-LSTM contract persistence: NOT VALIDATED',
            'G3':'VAE contract anomaly: NOT VALIDATED',
            'G4':'VolGAN MIN_EDGE_INR: NOT VALIDATED',
            'G5':'GAN CVaR: NOT AVAILABLE/VALIDATED',
            'G6':'Approved risk mandate: NOT VALIDATED',
            'G7':'LIVE EXECUTION: BLOCKED',
        },
        'why_not_exec':'No properly validated per-contract persistence, anomaly, calibrated CVaR or trading thresholds; close prices are not executable quotes.',
        'lookahead_control':'Research ranking uses same-date closes only. Next-session exit P&L is optional posthoc annotation and NOT a gate.',
        'candidate_type':'One index call/put parity pair at a single strike; not an individual stock.',
        'missing_value_policy':'Model scores are NA; never set to zero or replaced by guessed values.',
        'files':['trade_decision_matrix_all.csv','trade_decision_matrix.csv','trade_daily_summary.csv','trade_gate_breakdown.csv','trade_decision_matrix.html'],
    }
    (out/'trade_decision_summary.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    cols = ['entry_date','expiry','strike','daily_rank','proxy_score_points','gross_discrepancy_inr',
            'estimated_cost_inr','estimated_net_edge_inr','research_selection','live_decision','decision_reason']
    display_top = top[cols].copy().round(2)
    audit_example = gates[(gates.daily_rank==1)&(gates.entry_date==summary.entry_date.iloc[0])].copy()
    doc = '''<!doctype html><html lang="en"><meta charset="UTF-8"><title>VolGAN-X Research Trade Matrix</title>
<style>body{font:14px system-ui,Arial,sans-serif;margin:28px;color:#1c2431;background:#fafbfd}
 h1{font-size:24px} h2{font-size:17px;margin-top:28px}p{max-width:1100px;line-height:1.55}
 .matrix{font-size:12px;border-collapse:collapse;width:100%;background:white}
 .matrix td,.matrix th{border:1px solid #e0e5eb;padding:7px 8px;vertical-align:top;text-align:left}
 .matrix th{background:#1e3552;color:white;position:sticky;top:0}.matrix tr:nth-child(even){background:#f4f7fa}
 .note{border-left:4px solid #df9442;padding:10px 14px;background:#fff9ed}
 </style><h1>NIFTY Options: Trade-by-trade Research Decision Matrix</h1>
 <p class="note"><strong>Research selection ≠ live trade approval.</strong> All candidates are BLOCKED_LIVE.
 Experimental CNN-LSTM/VAE/GAN scores are not validated at the contract level; missing values are intentionally left blank.
 A positive modeled parity discrepancy is not guaranteed profit or an executable arbitrage.</p>
 '''
    doc += f'<p>{report["candidate_pairs"]} eligible CE/PE pairs across {report["dates"]} dates; '+\
           f'{report["selected_for_research_review"]} daily candidates selected for research review. '+\
           f'The table displays up to {top_n} best-ranked pairs per date.</p>'
    doc += _table_html(display_top, 'Daily top-ranked candidates and reasons')
    doc += _table_html(summary[cols].round(2), 'One top candidate per date')
    doc += _table_html(audit_example[['gate','check','observed_value','criterion','status','reason']],
                        'Example: full gate-by-gate explanation of first daily candidate')
    doc += '<p>See trade_gate_breakdown.csv for the gate matrix of every displayed candidate, '+\
           'and trade_decision_matrix_all.csv for all qualifying pairs.</p></html>'
    (out/'trade_decision_matrix.html').write_text(doc,encoding='utf-8')
    return report


def print_decision_matrix(outdir='research_outputs', max_rows=15, explain_date=None):
    p = Path(outdir)/'trade_daily_summary.csv'
    df = pd.read_csv(p)
    names = {'entry_date':'DATE','strike':'STRIKE','residual_points':'PARITY_PTS',
             'estimated_cost_inr':'COST_INR','estimated_net_edge_inr':'PROXY_EDGE_INR',
             'research_selection':'RESEARCH_DECISION','live_decision':'LIVE'}
    cols = list(names)
    table = df[cols].head(max_rows).rename(columns=names)
    print('\n=== PER-DATE TRADE RESEARCH DECISION MATRIX ===')
    print(table.round(2).to_string(index=False))
    print('\nGate breakdown: trade_gate_breakdown.csv; complete pairs: trade_decision_matrix_all.csv')
    print('IMPORTANT: SELECTED_FOR_REVIEW is NEVER EXECUTE. All live decisions remain BLOCKED_LIVE.')
    all_gates = pd.read_csv(Path(outdir)/'trade_gate_breakdown.csv')
    date = explain_date if explain_date is not None else df.entry_date.iloc[0]
    selected_gate = all_gates[(all_gates.entry_date == date) & (all_gates.daily_rank == 1)]
    if explain_date and selected_gate.empty:
        raise ValueError(f'No ranked candidate found for explain-date {date}; choose a date from trade_daily_summary.csv.')
    print(f'\n=== DETAILED GATE REASONS: {date}, TOP-RANKED PAIR ===')
    print(selected_gate[['gate','check','observed_value','status','reason']].fillna('N/A').to_string(index=False,
          col_space=8, max_colwidth=85))


def main():
    parser=argparse.ArgumentParser(description='Explain each NIFTY options parity-proxy candidate and its blocked live gate')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default='research_outputs')
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--explain-date', default=None, help='Show detailed 8-gate result for top pair on this YYYY-MM-DD date')
    args=parser.parse_args()
    report=run_decision_matrix(args.data_dir,args.output_dir,args.top_n)
    print('Candidate pairs:', report['candidate_pairs'], '| Review candidates:',report['selected_for_research_review'])
    print_decision_matrix(args.output_dir, explain_date=args.explain_date)

if __name__=='__main__':
    main()
