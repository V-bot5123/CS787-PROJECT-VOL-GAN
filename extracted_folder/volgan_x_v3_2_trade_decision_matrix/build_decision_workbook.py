from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from artifact_tool import Workbook, SpreadsheetFile

COL = {'navy':'#173354', 'muted':'#E9EFF5', 'light':'#F5F8FC',
       'blue':'#0000FF','text':'#172433','green':'#147D56',
       'red':'#B42318','amber':'#875F00','white':'#FFFFFF'}


def clean(v):
    if pd.isna(v):
        return None
    if hasattr(v, 'item'):
        return v.item()
    return v


def write_matrix(sheet, data, columns, headings, source_note, formulas=False):
    end = len(data)+1
    sheet.get_range_by_indexes(0,0,1,len(headings)).values=[headings]
    matrix=[[clean(v) for v in row] for row in data[columns].itertuples(index=False, name=None)]
    if matrix:
        sheet.get_range_by_indexes(1,0,len(matrix),len(columns)).values=matrix
    sheet.get_range_by_indexes(0,0,1,len(headings)).format={
        'fill':COL['navy'],'font':{'bold':True,'color':COL['white']},
        'row_height':38, 'wrap_text':True,'vertical_alignment':'center'}
    sheet.freeze_panes.freeze_rows(1)
    if matrix:
        sheet.get_range_by_indexes(1,0,len(matrix),len(headings)).format.row_height=20
    for j, col in enumerate(columns):
        width = 31 if ('reason' in col or 'decision' in col or 'selection' in col) else (25 if col in ('candidate_id','side_description') else 17)
        sheet.get_range_by_indexes(0,j,end,1).format.column_width=width
    for j, col in enumerate(columns):
        if col in ('estimated_net_edge_inr','gross_discrepancy_inr','estimated_cost_inr','pnl_net_inr'):
            sheet.get_range_by_indexes(1,j,max(1,len(matrix)),1).set_number_format('#,##0.00;[Red](#,##0.00);–')
    if matrix:
        for j,c in enumerate(columns):
            if c in ('strike','call_entry','put_entry','spot_entry','lot','residual_points','estimated_cost_inr'):
                sheet.get_range_by_indexes(1,j,len(matrix),1).format.font.color=COL['blue']
    return end


def build(outdir: Path):
    daily=pd.read_csv(outdir/'trade_daily_summary.csv')
    top=pd.read_csv(outdir/'trade_decision_matrix.csv')
    allpairs=pd.read_csv(outdir/'trade_decision_matrix_all.csv')
    gates=pd.read_csv(outdir/'trade_gate_breakdown.csv')

    wb=Workbook.create()
    overview=wb.worksheets.add('Read Me')
    overview.get_range('A1:D1').merge()
    overview.get_range('A1').values=[['VOLGAN-X 3.2  |  RESEARCH-ONLY OPTIONS DECISION MATRIX']]
    overview.get_range('A1:D1').format={'fill':COL['navy'],'font':{'color':COL['white'],'bold':True},'row_height':34}
    overview.get_range('A3:B10').values=[
        ['Metric','Result'],['Source trading dates',int(daily.entry_date.nunique())],
        ['Eligible paired-option candidates',len(allpairs)],
        ['Selected for research review',int((allpairs.research_selection=='SELECTED_FOR_REVIEW').sum())],
        ['Negative proxy after-cost screen',int((allpairs.research_selection=='REJECTED_BY_SCREEN').sum())],
        ['Positive but lower-ranked candidates',int((allpairs.research_selection=='NOT_SELECTED_RANK').sum())],
        ['Live execution decisions','ALL BLOCKED_LIVE'],
        ['Source','Bundled data/nifty_options_2024-*.csv; closing marks only']]
    overview.get_range('A3:B3').format={'fill':COL['muted'],'font':{'bold':True},'row_height':24}
    overview.get_range('A3:A17').format.column_width=43
    overview.get_range('B3:B17').format.column_width=79
    overview.get_range('A12:B17').values=[
        ['Code','Meaning'],
        ['SELECTED_FOR_REVIEW','Highest positive estimated net parity-proxy discrepancy on a date; NOT a live trade'],
        ['REJECTED_BY_SCREEN','Estimated gross parity proxy <= conservative modeled cost'],
        ['NOT_SELECTED_RANK','Positive proxy; better-ranked pair on same day'],
        ['NOT_VALIDATED','No valid contract-specific model result, risk threshold, or production quote'],
        ['BLOCKED_LIVE','Always applies; no trade can execute from this research build']]
    overview.get_range('A12:B12').format={'fill':COL['muted'],'font':{'bold':True}}
    overview.get_range('A19:B20').values=[['IMPORTANT','Historical close-price proxy scores are NOT expected profits or executable arbitrage'],
                                           ['Research scope','Index CE+PE pairs, not standalone stock picks; no validated model gates']]
    overview.get_range('A19:A20').format={'font':{'bold':True,'color':COL['red']}}

    maincols=['entry_date','expiry','strike','lot','residual_points','gross_discrepancy_inr',
             'estimated_cost_inr','estimated_net_edge_inr','daily_rank','research_selection',
             'live_decision','decision_reason','posthoc_exit_status','pnl_net_inr']
    headers=['ENTRY DATE','EXPIRY','STRIKE (pts)','LOT','PARITY RESIDUAL (pts)','GROSS PROXY (₹)',
             'MODELED COST (₹)','NET PROXY EDGE (₹)','DAILY RANK','RESEARCH SCREEN',
             'LIVE ORDER','WHY SELECTED / REJECTED','POST-HOC EXIT (not a gate)','POST-HOC PNL (₹)']
    write_matrix(wb.worksheets.add('Daily Decisions'),daily,maincols,headers,'')
    write_matrix(wb.worksheets.add('Top Candidates'),top,maincols,headers,'')
    gatecols=['entry_date','candidate_id','strike','daily_rank','gate','check','observed_value',
              'criterion','status','reason']
    write_matrix(wb.worksheets.add('Gate Breakdown'),gates,gatecols,
                 ['DATE','CANDIDATE ID','STRIKE','RANK','GATE','CHECK','OBSERVED / REPORTED',
                  'COMPARISON / DEFAULT','RESULT','WHY'], '')
    allcols=['entry_date','expiry','strike','daily_rank','lot','residual_points','gross_discrepancy_inr',
             'estimated_cost_inr','estimated_net_edge_inr','research_selection','decision_reason','live_decision']
    write_matrix(wb.worksheets.add('All 867 Pairs'),allpairs,allcols,
                 ['ENTRY DATE','EXPIRY','STRIKE','RANK','LOT','PARITY POINTS','GROSS ₹',
                  'COST ₹','NET PROXY ₹','RESEARCH STATUS','EXPLANATION','LIVE STATUS'], '')

    assumptions=wb.worksheets.add('Threshold Notes')
    assumptions.get_range('A1:D1').values=[['THRESHOLD','ORIGINAL DEFAULT','VALIDATION','HOW USED IN V3.2']]
    assumptions.get_range('A1:D1').format={'fill':COL['navy'],'font':{'bold':True,'color':COL['white']},'row_height':25}
    assumptions.get_range('A2:D9').values=[
        ['THETA_Y',0.70,'NOT VALIDATED','Not evaluated; no valid per-contract CNN-LSTM probability'],
        ['TAU_HEALTHY',2.50,'NOT VALIDATED','Not evaluated; no valid per-contract anomaly z'],
        ['MIN_EDGE_INR',250,'NOT VALIDATED','NOT compared with parity proxy because VolGAN residual edge is different'],
        ['KAPPA',3,'NOT VALIDATED','No validated per-contract CVaR or approved risk mandate'],
        ['RISK_BUDGET_INR',25000,'NOT VALIDATED','Not applied; capital policy not approved'],
        ['PARITY_PROXY_EDGE',0,'DESCRIPTIVE ONLY','Positive if |leave-one-out parity residual|×lot - modeled costs > ₹0'],
        ['DATA SOURCES','13 historical daily NSE-style CSV files','HISTORICAL CLOSE ONLY','No bid/ask or true executable hedge quotations'],
        ['LIVE AUTHORIZATION','DISABLED','FIXED BLOCK','No execution path in this research build']]
    for j,width in enumerate([26,25,24,92]):
        assumptions.get_range_by_indexes(0,j,11,1).format.column_width=width
    assumptions.freeze_panes.freeze_rows(1)
    assumptions.get_range('B2:B7').format.font.color=COL['blue']

    target=outdir/'trade_decision_matrix.xlsx'
    SpreadsheetFile.export_xlsx(wb).save(str(target))
    print('Wrote:',target)
    result=wb.inspect({'kind':'table','range':'Daily Decisions!A1:J5','include':'values,formulas',
                       'table_max_rows':5,'table_max_cols':10})
    print('Inspection:',result.ndjson[:700])
    return target


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',default='research_outputs')
    args=ap.parse_args()
    build(Path(args.output_dir))
