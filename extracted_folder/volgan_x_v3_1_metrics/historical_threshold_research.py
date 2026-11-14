from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RATE = 0.065
V_MIN = 10
OI_MIN = 100
MIN_TRAIN_DAYS = 20
MIN_HOLDOUT_DAYS = 10
MAX_SESSION_GAP_DAYS = 4
MIN_PREMIUM = 1.0
MAX_MONEYNESS = 0.025
MAX_DAYS_TO_EXPIRY = 45
MIN_EXPIRY_DAYS = 3
CANDIDATE_EDGES = (0.0, 250.0, 500.0, 1000.0, 2000.0)
REQUIRED_COLUMNS = {
    'TRADE_DATE', 'INSTRUMENT', 'SYMBOL', 'EXPIRY_DT', 'STRIKE_PR',
    'OPTION_TYP', 'CLOSE', 'SETTLE_PR', 'CONTRACTS', 'OPEN_INT', 'UNDERLYING_CLOSE'
}


def historical_lot_size(trade_date: pd.Timestamp, expiry_date: pd.Timestamp) -> int:
    d = pd.Timestamp(trade_date)
    exp = pd.Timestamp(expiry_date)
    if d < pd.Timestamp('2024-04-26'):
        return 50
    if d < pd.Timestamp('2024-05-30') and exp >= pd.Timestamp('2024-04-26'):
        raise ValueError('Overlapping 2024-04/05 contract lot versions require contract master data.')
    if d >= pd.Timestamp('2024-05-30') and d < pd.Timestamp('2024-11-20'):
        return 25
    raise ValueError('Lot size outside verified 2024 historical range needs contract master data.')


def resolve_option_data_dir(data_dir: str | Path | None = None) -> Path:
    script_dir = Path(__file__).resolve().parent
    if data_dir is None or str(data_dir).lower() == 'auto':
        candidates = [script_dir / 'data', script_dir,
                      Path.cwd() / 'data', Path.cwd()]
    else:
        candidates = [Path(data_dir).expanduser()]
    seen: set[Path] = set()
    searched = []
    for item in candidates:
        directory = item.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        searched.append(directory)
        if directory.is_dir() and any(directory.glob('nifty_options_*.csv')):
            return directory
    checked = '\n  '.join(str(x) for x in searched)
    raise FileNotFoundError(
        'No nifty_options_*.csv files found.\n'
        f'Searched:\n  {checked}\n'
        'The original daily CSV files must be extracted into the package data/ folder, '
        'or specify their actual directory with --data-dir "C:\\path\\to\\csvs". '
        'Note: "." refers to your terminal working directory, not the script folder.'
    )


def read_option_csvs(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    directory = resolve_option_data_dir(data_dir)
    paths = sorted(directory.glob('nifty_options_*.csv'))
    pieces = []
    file_stats = []
    for path in paths:
        raw = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f'{path.name}: missing columns: {sorted(missing)}')
        raw['TRADE_DATE'] = pd.to_datetime(raw['TRADE_DATE'], errors='raise')
        raw['EXPIRY_DT'] = pd.to_datetime(raw['EXPIRY_DT'], format='%d-%b-%Y', errors='raise')
        dates_in_file = raw['TRADE_DATE'].unique()
        if len(dates_in_file) != 1 or pd.Timestamp(dates_in_file[0]).strftime('%Y-%m-%d') not in path.name:
            raise ValueError(f'{path.name}: expected exactly one date matching filename')
        if raw[['TRADE_DATE', 'EXPIRY_DT', 'STRIKE_PR', 'OPTION_TYP']].duplicated().any():
            raise ValueError(f'{path.name}: duplicate contract keys')
        raw = raw.loc[(raw['INSTRUMENT'] == 'OPTIDX') & (raw['SYMBOL'] == 'NIFTY')
                      & raw['OPTION_TYP'].isin(['CE', 'PE'])].copy()
        pieces.append(raw)
        file_stats.append({'file': path.name, 'rows': int(len(raw))})
    df = pd.concat(pieces, ignore_index=True).sort_values(['TRADE_DATE', 'EXPIRY_DT', 'STRIKE_PR'])
    if df.duplicated(['TRADE_DATE', 'EXPIRY_DT', 'STRIKE_PR', 'OPTION_TYP']).any():
        raise ValueError('Duplicate dates/contracts across CSV files')
    for col in ['STRIKE_PR', 'CLOSE', 'SETTLE_PR', 'CONTRACTS', 'OPEN_INT', 'UNDERLYING_CLOSE'] + (['LOT_SIZE'] if 'LOT_SIZE' in df else []):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[list(['STRIKE_PR', 'CLOSE', 'CONTRACTS', 'OPEN_INT', 'UNDERLYING_CLOSE'])].isna().any().any():
        raise ValueError('Missing critical numeric fields in supplied CSVs')
    report = {
        'source_files': len(paths), 'source_rows': len(df),
        'dates': [d.strftime('%Y-%m-%d') for d in sorted(df.TRADE_DATE.unique())],
        'stale_or_illiquid_rows': int(((df['CONTRACTS'] < V_MIN) | (df['OPEN_INT'] < OI_MIN)).sum()),
        'close_settlement_disagreements': int((np.abs(df['CLOSE'] - df['SETTLE_PR']) > 0.05).sum()),
        'file_stats': file_stats,
    }
    return df, report


def half_spread_proxy(premium: float) -> float:
    return max(0.50, 0.015 * premium)


def pair_roundtrip_cost(call: float, put: float, lot: int) -> float:
    total_premium = (call + put) * lot
    spread = 2.0 * (half_spread_proxy(call) + half_spread_proxy(put)) * lot
    brokerage = 4 * 20.0
    exchange = 0.00050 * total_premium * 2
    sebi = 0.000001 * total_premium * 2
    stt = 0.001 * total_premium
    stamp = 0.00003 * total_premium
    gst = 0.18 * (brokerage + exchange + sebi)
    hedging_proxy = 0.05 * lot
    return float(spread + brokerage + exchange + sebi + stt + stamp + gst + hedging_proxy)


def create_daily_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    liquid = raw[(raw['CONTRACTS'] >= V_MIN) & (raw['OPEN_INT'] >= OI_MIN)
                 & (raw['CLOSE'] >= MIN_PREMIUM) & (raw['STRIKE_PR'] > 0)
                 & (raw['UNDERLYING_CLOSE'] > 0)].copy()
    if liquid.empty:
        return pd.DataFrame()
    if 'LOT_SIZE' in liquid.columns:
        missing_lot = liquid['LOT_SIZE'].isna()
        if missing_lot.any():
            liquid.loc[missing_lot, 'LOT_SIZE'] = liquid.loc[missing_lot].apply(
                lambda r: historical_lot_size(r['TRADE_DATE'],r['EXPIRY_DT']), axis=1)
        if (liquid['LOT_SIZE'] <= 0).any():
            raise ValueError('Invalid contract lot size')
    pair_index = ['TRADE_DATE', 'EXPIRY_DT', 'STRIKE_PR', 'UNDERLYING_CLOSE']
    if 'LOT_SIZE' in liquid.columns:
        pair_index.append('LOT_SIZE')
    pairs = liquid.pivot_table(
        index=pair_index,
        columns='OPTION_TYP', values='CLOSE', aggfunc='first', observed=True
    ).dropna(subset=['CE', 'PE']).reset_index()
    if pairs.empty:
        return pd.DataFrame()
    pairs.columns.name = None
    pairs['dte'] = (pairs['EXPIRY_DT'] - pairs['TRADE_DATE']).dt.days
    pairs = pairs[(pairs['dte'] >= MIN_EXPIRY_DAYS) & (pairs['dte'] <= MAX_DAYS_TO_EXPIRY)]
    pairs = pairs[(pairs['STRIKE_PR'] / pairs['UNDERLYING_CLOSE'] - 1).abs() <= MAX_MONEYNESS].copy()
    if pairs.empty:
        return pd.DataFrame()
    pairs['df'] = np.exp(-DEFAULT_RATE * pairs['dte'] / 365)
    pairs['parity_intercept'] = pairs['CE'] - pairs['PE'] + pairs['df'] * pairs['STRIKE_PR']
    out = []
    for (date, expiry), group in pairs.groupby(['TRADE_DATE', 'EXPIRY_DT']):
        if len(group) < 7:
            continue
        intercepts = group['parity_intercept'].to_numpy()
        for idx, (_, row) in enumerate(group.iterrows()):
            other = np.delete(intercepts, idx)
            if len(other) < 6:
                continue
            baseline = float(np.median(other))
            residual = float(row['parity_intercept'] - baseline)
            lot = int(row.LOT_SIZE) if 'LOT_SIZE' in pairs.columns else historical_lot_size(date, expiry)
            cost = pair_roundtrip_cost(float(row.CE), float(row.PE), lot)
            out.append({
                'entry_date': pd.Timestamp(date), 'expiry': pd.Timestamp(expiry),
                'strike': float(row.STRIKE_PR), 'call_entry': float(row.CE),
                'put_entry': float(row.PE), 'spot_entry': float(row.UNDERLYING_CLOSE),
                'lot': int(lot), 'residual_points': residual,
                'gross_discrepancy_inr': abs(residual) * lot,
                'estimated_cost_inr': cost,
                'estimated_net_edge_inr': abs(residual) * lot - cost,
                'side': -1 if residual > 0 else 1,
            })
    if not out:
        return pd.DataFrame()
    signals = pd.DataFrame(out).sort_values(
        ['entry_date', 'estimated_net_edge_inr', 'strike'], ascending=[True, False, True]
    )
    return signals.drop_duplicates('entry_date', keep='first').reset_index(drop=True)


def evaluate_next_session(candidates: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    dates = sorted(raw['TRADE_DATE'].unique())
    next_map = {pd.Timestamp(a): pd.Timestamp(b) for a, b in zip(dates[:-1], dates[1:])}
    result = candidates.copy()
    result['next_date'] = result['entry_date'].map(next_map)
    result['gap_days'] = (result['next_date'] - result['entry_date']).dt.days
    result = result[(result['gap_days'] > 0) & (result['gap_days'] <= MAX_SESSION_GAP_DAYS)
                    & (result['next_date'] < result['expiry'])].copy()
    if result.empty:
        return result
    closing = raw[(raw['CLOSE'] >= MIN_PREMIUM) & (raw['CONTRACTS'] >= V_MIN)
                  & (raw['OPEN_INT'] >= OI_MIN)].pivot_table(
        index=['TRADE_DATE', 'EXPIRY_DT', 'STRIKE_PR', 'UNDERLYING_CLOSE'],
        columns='OPTION_TYP', values='CLOSE', aggfunc='first'
    ).dropna(subset=['CE', 'PE']).reset_index()
    closing.columns.name = None
    closing = closing.rename(columns={
        'TRADE_DATE': 'next_date', 'EXPIRY_DT': 'expiry', 'STRIKE_PR': 'strike',
        'UNDERLYING_CLOSE': 'spot_next', 'CE': 'call_next', 'PE': 'put_next'
    })
    merged = result.merge(closing, on=['next_date', 'expiry', 'strike'], how='left', validate='one_to_one')
    merged['exit_observed'] = merged['call_next'].notna() & merged['put_next'].notna()
    spread_move = (merged['call_next'] - merged['put_next']) - (merged['call_entry'] - merged['put_entry'])
    spot_move = merged['spot_next'] - merged['spot_entry']
    merged['pnl_net_inr'] = (merged['side'] * merged['lot'] * (spread_move - spot_move)
                             - merged['estimated_cost_inr']).where(merged['exit_observed'])
    merged['result'] = np.where(merged['exit_observed'], 'OBSERVED_PROXY', 'NO_VALID_EXIT')
    return merged


def pnl_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {'n_dates': 0, 'n_trades': 0, 'sum_net_pnl_inr': 0.0,
                'avg_net_pnl_inr': None, 'win_rate': None, 'worst_net_pnl_inr': None,
                'cvar_95_inr': None, 'max_cumulative_drawdown_inr': None}
    pnl = frame['pnl_net_inr'].dropna().to_numpy(dtype=float)
    if not len(pnl):
        return {'n_dates': int(frame['entry_date'].nunique()), 'n_trades': 0,
                'sum_net_pnl_inr': 0.0, 'avg_net_pnl_inr': None, 'win_rate': None,
                'worst_net_pnl_inr': None, 'cvar_95_inr': None, 'max_cumulative_drawdown_inr': None}
    equity = np.concatenate([[0], np.cumsum(pnl)])
    drawdown = np.maximum.accumulate(equity) - equity
    k = max(1, math.ceil(0.05 * len(pnl)))
    return {'n_dates': int(frame['entry_date'].nunique()), 'n_trades': int(len(pnl)),
            'sum_net_pnl_inr': round(float(pnl.sum()), 2),
            'avg_net_pnl_inr': round(float(pnl.mean()), 2),
            'win_rate': round(float(np.mean(pnl > 0)), 4),
            'worst_net_pnl_inr': round(float(np.min(pnl)), 2),
            'cvar_95_inr': round(float(np.mean(np.sort(pnl)[:k])), 2),
            'max_cumulative_drawdown_inr': round(float(drawdown.max()), 2)}


def run_historical_research(data_dir: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    raw, overview = read_option_csvs(data_dir)
    all_candidates = create_daily_candidates(raw)
    evaluated = evaluate_next_session(all_candidates, raw)
    distinct_dates = sorted(evaluated.entry_date.unique()) if not evaluated.empty else []
    n_train = max(1, int(len(distinct_dates) * 0.6)) if len(distinct_dates) >= 2 else len(distinct_dates)
    split_dates = set(distinct_dates[:n_train])
    train = evaluated[evaluated.entry_date.isin(split_dates)].copy()
    test = evaluated[~evaluated.entry_date.isin(split_dates)].copy()
    rows = []
    for min_edge in CANDIDATE_EDGES:
        t = train[train.estimated_net_edge_inr >= min_edge]
        h = test[test.estimated_net_edge_inr >= min_edge]
        rows.append({'min_edge_inr': min_edge, 'train': pnl_metrics(t), 'holdout': pnl_metrics(h)})
    eligible = [row for row in rows if row['train']['n_trades'] >= 3]
    winner = max(eligible, key=lambda x: (x['train']['sum_net_pnl_inr'],
                                           x['train']['avg_net_pnl_inr'], x['min_edge_inr'])) if eligible else None
    selected = winner['min_edge_inr'] if winner else None
    holdout = winner['holdout'] if winner else None
    source_gaps = []
    for a, b in zip(overview['dates'][:-1], overview['dates'][1:]):
        gap = (pd.Timestamp(b) - pd.Timestamp(a)).days
        if gap > MAX_SESSION_GAP_DAYS:
            source_gaps.append({'from': a, 'to': b, 'calendar_days': gap})
    has_enough_days = len(train) >= MIN_TRAIN_DAYS and len(test) >= MIN_HOLDOUT_DAYS
    no_missing_exit = bool(evaluated.empty or evaluated['exit_observed'].all())
    blockers = [
        'Original persistence classifier, VAE and WGAN-GP have no authenticated trained and validated checkpoints.',
        'CSV files contain daily closes, not synchronized bid/ask or executable hedging quotes.',
        'Original VolGAN THETA_Y, TAU_HEALTHY, KAPPA and RISK_BUDGET_INR cannot be fitted from these fields.',
        'Any experimental MIN_EDGE threshold belongs to a parity-proxy baseline, not VolGAN residual trading.',
        'Realized funding, dividends, STT treatment, margin, futures hedging and market impact are not verified.',
    ]
    if not has_enough_days:
        blockers.append(f'Insufficient chronological observations: {len(train)} training and {len(test)} holdout dates (require at least {MIN_TRAIN_DAYS}/{MIN_HOLDOUT_DAYS} research dates).')
    if not no_missing_exit:
        blockers.append('Some entry-selected contracts lack an eligible subsequent closing mark; results may have selection bias.')
    if winner is None:
        blockers.append('Fewer than three observed qualifying training cases for every threshold.')
    if holdout is not None and holdout['n_trades'] < 5:
        blockers.append('Selected provisional threshold has fewer than five holdout trades.')
    report = {
        'status': 'RESEARCH_ONLY', 'trade_authorization': 'DISABLED', 'decision': 'ABORT',
        'data': overview, 'gap_exclusions': source_gaps,
        'methodology': 'Leave-one-strike-out same-expiry call-put parity reference; one daily candidate; next-source-session spot-hedged close-to-close synthetic PnL, estimated costs.',
        'entries_selected': int(len(all_candidates)),
        'eligible_transition_dates': int(len(evaluated)),
        'observed_exits': int(evaluated['exit_observed'].sum()) if not evaluated.empty else 0,
        'missing_exits': int((~evaluated['exit_observed']).sum()) if not evaluated.empty else 0,
        'training_period': {'start': train.entry_date.min().strftime('%Y-%m-%d') if len(train) else None,
                            'end': train.entry_date.max().strftime('%Y-%m-%d') if len(train) else None,
                            'dates': int(train.entry_date.nunique())},
        'holdout_period': {'start': test.entry_date.min().strftime('%Y-%m-%d') if len(test) else None,
                           'end': test.entry_date.max().strftime('%Y-%m-%d') if len(test) else None,
                           'dates': int(test.entry_date.nunique())},
        'threshold_comparison': rows,
        'training_selected_candidate_costs_inr': {
            'median': round(float(train['estimated_cost_inr'].median()), 2) if len(train) else None,
            'p90': round(float(train['estimated_cost_inr'].quantile(0.9)), 2) if len(train) else None,
            'n_candidate_dates': int(len(train)),
            'interpretation': 'Illustrative modeled round-trip friction, not optimized MIN_EDGE or actual quotes.'
        },
        'provisional_research_min_edge_inr': selected,
        'provisional_holdout': holdout,
        'volgan_trading_thresholds': {
            'THETA_Y': {'value': None, 'status': 'NOT_ESTIMABLE_UNTRAINED'},
            'TAU_HEALTHY': {'value': None, 'status': 'NOT_ESTIMABLE_UNTRAINED'},
            'MIN_EDGE_INR': {'value': None, 'status': 'NOT_VALIDATED_FOR_VOLGAN'},
            'KAPPA': {'value': None, 'status': 'REQUIRES_CAPITAL_AND_RISK_POLICY'},
            'RISK_BUDGET_INR': {'value': None, 'status': 'REQUIRES_CAPITAL_AND_RISK_POLICY'},
        },
        'limitations_and_blockers': blockers,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / 'historical_threshold_report.json'
        audit_path = output / 'historical_daily_candidates.csv'
        comparison_path = output / 'historical_threshold_comparison.csv'
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        evaluated.to_csv(audit_path, index=False, float_format='%.5f')
        pd.DataFrame([{
            'min_edge_inr': item['min_edge_inr'],
            **{'train_' + k: v for k, v in item['train'].items()},
            **{'holdout_' + k: v for k, v in item['holdout'].items()},
        } for item in rows]).to_csv(comparison_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Research-only NIFTY historical threshold comparison')
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Folder with nifty_options_*.csv; omit to auto-find bundled data/')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    report = run_historical_research(args.data_dir, args.output_dir)
    print(f"Source dates: {len(report['data']['dates'])}, records: {report['data']['source_rows']}")
    print(f"Training/holdout: {report['training_period']['dates']}/{report['holdout_period']['dates']} dates")
    print(f"Provisional MIN_EDGE parity-proxy research candidate: {report['provisional_research_min_edge_inr']}")
    for row in report['threshold_comparison']:
        print(f"Threshold {row['min_edge_inr']:>7.0f} | train {row['train']['n_trades']} trades / INR {row['train']['sum_net_pnl_inr']:>10.2f}"
              f" | holdout {row['holdout']['n_trades']} trades / INR {row['holdout']['sum_net_pnl_inr']:>10.2f}")
    print('DECISION: ABORT -- RESEARCH ONLY. LIVE TRADE AUTHORIZATION DISABLED.')
    print(f'Reports saved under {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
