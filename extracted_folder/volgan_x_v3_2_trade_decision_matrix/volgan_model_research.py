from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnF

from historical_threshold_research import read_option_csvs, evaluate_next_session, create_daily_candidates, CANDIDATE_EDGES, pnl_metrics
from volgan_combined import (VolatilityVAE, CellPersistenceNet, VolGANGenerator,
                             VolGANCritic, invert_iv_black76, STD_MONEYNESS_X,
                             TENORS_DAYS, ATM_REF_VOL, RISK_FREE_RATE,
                             NUM_TENORS, NUM_STRIKES)

MAX_GAP = 4
LOOKBACK = 3
LABEL_RESID_VOL = 0.005
MIN_VALID_TRAIN_DATES = 120
MIN_VALID_HOLDOUT_DATES = 40
MIN_VALID_GAN_TRANSITIONS = 150
SEED = 1703


def seed_all(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))


def build_surface(day: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    out = np.full((NUM_TENORS, NUM_STRIKES), np.nan, dtype=np.float32)
    mask = np.zeros_like(out, dtype=np.float32)
    good = day[(day.CONTRACTS >= 10) & (day.OPEN_INT >= 100) & (day.CLOSE > 1) &
               (day.STRIKE_PR > 0)].copy()
    if good.empty:
        return out, mask, {'observed_cells': 0, 'tenors_observed': 0}
    date = pd.Timestamp(day.TRADE_DATE.iloc[0])
    spot = float(day.UNDERLYING_CLOSE.median())
    good['dte'] = (good.EXPIRY_DT - date).dt.days
    expiries = sorted((int(v), t) for t, v in good.groupby('EXPIRY_DT').dte.first().items() if v > 0)
    used: set[pd.Timestamp] = set()
    forward_sources = []
    for i, target in enumerate(TENORS_DAYS):
        available = [(abs(d - target), d, exp) for d, exp in expiries if exp not in used]
        if not available:
            continue
        distance, dte, expiry = min(available)
        if distance > max(3, min(10, round(target * 0.22))):
            continue
        subset = good[good.EXPIRY_DT == expiry]
        pairs = subset.pivot_table(index='STRIKE_PR', columns='OPTION_TYP',
                                   values='CLOSE', aggfunc='first').dropna(subset=['CE','PE'])
        pairs = pairs[(pairs.index >= spot * .97) & (pairs.index <= spot * 1.03)]
        if len(pairs) < 4:
            continue
        tau = float(dte / 365)
        fvals = pairs.index.to_numpy() + math.exp(RISK_FREE_RATE * tau) * (pairs.CE - pairs.PE).to_numpy()
        forward = float(np.median(fvals))
        if not np.isfinite(forward) or not .93*spot <= forward <= 1.07*spot:
            continue
        used.add(expiry)
        forward_sources.append({'target_days': int(target), 'actual_days': int(dte), 'expiry': str(expiry.date()), 'paired_strikes': int(len(pairs))})
        candidates: dict[int,list[float]] = {}
        for r in subset.itertuples(index=False):
            strike = float(r.STRIKE_PR)
            otm = ('CE' if strike >= forward else 'PE')
            if r.OPTION_TYP != otm:
                continue
            x = math.log(strike / forward) / (ATM_REF_VOL * math.sqrt(tau))
            if x < STD_MONEYNESS_X[0] or x > STD_MONEYNESS_X[-1]:
                continue
            j = int(np.argmin(np.abs(STD_MONEYNESS_X - x)))
            if abs(STD_MONEYNESS_X[j] - x) > .5*(STD_MONEYNESS_X[1]-STD_MONEYNESS_X[0]):
                continue
            iv = invert_iv_black76(float(r.CLOSE), forward, strike, tau, RISK_FREE_RATE, otm)
            if np.isfinite(iv) and .02 < iv < 2.0:
                candidates.setdefault(j, []).append(float(iv))
        for j, vals in candidates.items():
            out[i,j] = np.median(vals)
            mask[i,j] = 1.
    return out, mask, {'observed_cells': int(mask.sum()),
                       'tenors_observed': int((mask.sum(axis=1)>0).sum()),
                       'forward_estimates': forward_sources}


def make_dataset(raw: pd.DataFrame):
    dates = sorted(pd.Timestamp(v) for v in raw.TRADE_DATE.unique())
    surfaces, masks, details, spots = [], [], [], []
    for date in dates:
        day = raw[raw.TRADE_DATE == date]
        x,m,info = build_surface(day)
        surfaces.append(x); masks.append(m)
        spots.append(float(day.UNDERLYING_CLOSE.median()))
        details.append({'date': date.strftime('%Y-%m-%d'), **info})
    return dates, np.stack(surfaces), np.stack(masks), np.array(spots), details


def fit_scaler(x: np.ndarray, mask: np.ndarray):
    observed = np.where(mask > 0, x, np.nan)
    global_med = float(np.nanmedian(observed)) if np.isfinite(observed).any() else .2
    count = (mask > 0).sum(axis=0)
    summed = np.nansum(observed, axis=0)
    mean = np.divide(summed, count, out=np.full_like(summed,global_med), where=count>0)
    variance = np.divide(np.nansum((observed-mean)**2, axis=0),count,
                         out=np.zeros_like(mean),where=count>0)
    std = np.sqrt(variance)
    std = np.where((count >= 3) & (std >= .01), std, .05)
    return mean.astype('float32'), std.astype('float32')


def standardize(x: np.ndarray, mask: np.ndarray, mean, std):
    return np.clip(np.where(mask > 0, (x-mean)/std, 0.), -6., 6.).astype('float32')


def local_residual(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    res = np.zeros_like(x, dtype='float32')
    for t in range(len(x)):
        for i in range(NUM_TENORS):
            valid = np.flatnonzero(mask[t,i] > 0)
            if len(valid) < 4:
                continue
            for j in valid:
                adjacent = valid[np.abs(valid-j) <= 4]
                if len(adjacent) >= 3:
                    res[t,i,j] = x[t,i,j] - float(np.median(x[t,i,adjacent]))
    return res


def adjacent_indices(dates: list[pd.Timestamp]) -> list[int]:
    return [t for t in range(len(dates)-1) if 0 < (dates[t+1]-dates[t]).days <= MAX_GAP]


def get_windows(dates: list[pd.Timestamp], length: int = LOOKBACK):
    ends=[]
    for t in range(length-1, len(dates)-1):
        if all(0 < (dates[k+1]-dates[k]).days <= MAX_GAP for k in range(t-length+1,t+1)):
            ends.append(t)
    return ends


def make_features(xstd, r, mask, spot):
    n = len(spot)
    spotret = np.zeros(n, dtype='float32')
    spotret[1:] = np.clip(np.diff(np.log(spot)) * 50., -5., 5.)
    moneyness = np.broadcast_to(STD_MONEYNESS_X[None,None,:], (n,NUM_TENORS,NUM_STRIKES)).astype('float32')
    return np.stack([xstd, np.clip(r / .02, -5, 5), mask,
                     np.broadcast_to(spotret[:,None,None],mask.shape), moneyness], axis=1).astype('float32')


def persistence_labels(residual, mask, t):
    obs = (mask[t] > 0) & (mask[t+1] > 0)
    good = ((np.abs(residual[t]) >= LABEL_RESID_VOL) &
            (np.abs(residual[t+1]) >= LABEL_RESID_VOL) &
            (residual[t]*residual[t+1] > 0))
    return good.astype('float32'), obs.astype('float32')


def masked_reconstruction_metrics(target, prediction, mask, reference=0.0):
    obs=np.asarray(mask)>0
    n=int(obs.sum())
    if n==0:
        return {'observed_cells':0,'mse':None,'rmse':None,'mae':None,'nmse':None}
    y=np.asarray(target)[obs]; yhat=np.asarray(prediction)[obs]
    error=yhat-y
    sse=float(np.square(error).sum())
    denom=float(np.square(y-reference).sum())
    return {'observed_cells':n,'mse':sse/n,'rmse':math.sqrt(sse/n),
            'mae':float(np.abs(error).mean()),
            'nmse':sse/denom if denom>1e-12 else None}


def masked_classifier_metrics(probs, labels, mask):
    obs=np.asarray(mask)>0
    n=int(obs.sum())
    if n==0:
        return {'observed_cells':0,'positive_cells':0,'bce':None,'brier':None,'accuracy':None}
    p=np.clip(np.asarray(probs)[obs].astype(float),1e-7,1-1e-7)
    y=np.asarray(labels)[obs].astype(float)
    return {'observed_cells':n,'positive_cells':int(y.sum()),
            'bce':float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
            'brier':float(np.square(p-y).mean()),
            'accuracy':float(((p>=.5)==y).mean())}


def vae_measure(model, xstd, mask, indices):
    index=list(indices)
    if not index:
        return masked_reconstruction_metrics(np.zeros((0,8,40)),np.zeros((0,8,40)),np.zeros((0,8,40)))
    with torch.no_grad():
        xs=torch.from_numpy(xstd[index].reshape(len(index),-1))
        mu,logv=model.encode(xs)
        rec=model.decode(mu).numpy().reshape(xstd[index].shape)
    return masked_reconstruction_metrics(xstd[index],rec,mask[index])


def train_vae(xstd, mask, n_train, val_indices=(), epochs=12):
    vae=VolatilityVAE()
    opt=torch.optim.Adam(vae.parameters(),lr=1e-3)
    xs=torch.from_numpy(xstd[:n_train].reshape(n_train,-1))
    ms=torch.from_numpy(mask[:n_train].reshape(n_train,-1))
    history=[]
    for epoch in range(1,epochs+1):
        vae.train()
        rec, mu, logv=vae(xs)
        rec_loss=(((rec-xs)**2)*ms).sum()/ms.sum().clamp(min=1)
        kl=-.5*(1+logv-mu.square()-logv.exp()).sum(dim=1).mean()
        loss=rec_loss+.005*kl
        opt.zero_grad();loss.backward();opt.step()
        vae.eval()
        val=vae_measure(vae,xstd,mask,val_indices)
        train=vae_measure(vae,xstd,mask,range(n_train))
        history.append({'epoch':epoch,'train_objective':float(loss.item()),
                        'train_stochastic_recon_mse':float(rec_loss.item()),
                        'train_kl':float(kl.item()),
                        'train_reconstruction_mse':train['mse'],
                        'train_nmse':train['nmse'],
                        'val_reconstruction_mse':val['mse'],
                        'val_nmse':val['nmse']})
    vae.eval()
    with torch.no_grad():
        full=torch.from_numpy(xstd.reshape(len(xstd),-1))
        mu, _ = vae.encode(full)
        recon=vae.decode(mu).numpy().reshape(xstd.shape)
    err=np.array([float(np.mean((recon[i][mask[i]>0]-xstd[i][mask[i]>0])**2))
                  if mask[i].sum() else np.nan for i in range(len(xstd))])
    return vae,mu.numpy(),err,float(loss.item()),history


def evaluate_persistence(model, features, r, mask, indices):
    model.eval()
    probs=[];ys=[];observed=[];daily=[]
    with torch.no_grad():
        for t in indices:
            logits,_=model(torch.from_numpy(features[t-LOOKBACK+1:t+1][None]))
            p=torch.sigmoid(logits).numpy()[0]
            y,o=persistence_labels(r,mask,t)
            metrics=masked_classifier_metrics(p,y,o)
            if metrics['observed_cells']:
                daily.append({'date':str(t),'index':int(t),**metrics,
                              'positives':metrics['positive_cells'],
                              'probability_mean':float(p[o>0].mean())})
            probs.append(p);ys.append(y);observed.append(o)
    if not probs:
        return masked_classifier_metrics(np.zeros((0,8,40)),np.zeros((0,8,40)),np.zeros((0,8,40))),[]
    return masked_classifier_metrics(np.stack(probs),np.stack(ys),np.stack(observed)),daily


def train_persistence(features,r,mask,ends,train_ends,val_ends,test_ends,epochs=12):
    model=CellPersistenceNet(C=5,F_dim=8,d=16,a=8)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    ytrain=[];otrain=[]
    for t in train_ends:
        y,o=persistence_labels(r,mask,t);ytrain.append(y);otrain.append(o)
    y_np=np.stack(ytrain);o_np=np.stack(otrain)
    positive=float((y_np*o_np).sum());negatives=float(((1-y_np)*o_np).sum())
    pos_weight=min(20.,negatives/max(1.,positive))
    history=[]
    for epoch in range(1,epochs+1):
        model.train();weighted_loss=0.;denom=0.
        for t in train_ends:
            xx=torch.from_numpy(features[t-LOOKBACK+1:t+1][None])
            target,obs=persistence_labels(r,mask,t)
            logits,_=model(xx)
            yy=torch.from_numpy(target[None]);oo=torch.from_numpy(obs[None])
            cell_loss=nnF.binary_cross_entropy_with_logits(logits,yy,reduction='none',
                                                          pos_weight=torch.tensor(pos_weight))
            loss=(cell_loss*oo).sum()/oo.sum().clamp(min=1)
            opt.zero_grad();loss.backward();opt.step()
            weighted_loss+=float(loss.item())*float(oo.sum());denom+=float(oo.sum())
        train,_=evaluate_persistence(model,features,r,mask,train_ends)
        val,_=evaluate_persistence(model,features,r,mask,val_ends)
        history.append({'epoch':epoch,'train_weighted_bce':weighted_loss/max(1.,denom),
                        'train_bce':train['bce'],'train_brier':train['brier'],
                        'val_bce':val['bce'],'val_brier':val['brier']})
    test_metrics,rows=evaluate_persistence(model,features,r,mask,test_ends)
    val_metrics,_=evaluate_persistence(model,features,r,mask,val_ends)
    return model,rows,{'train_positive_cells':int(positive),
                       'train_negative_cells':int(negatives),
                       'last_train_loss':history[-1]['train_weighted_bce'],
                       'train_metrics':train,'val_metrics':val_metrics,
                       'test_metrics':test_metrics,'epoch_history':history}

def datestr(t, ends):
    return str(t)


def rv_cond(dates, spots, idx, mu):
    contiguous=0
    for k in range(idx-1,-1,-1):
        if not (0 < (dates[k+1]-dates[k]).days <= MAX_GAP): break
        contiguous+=1
    past=np.diff(np.log(spots[max(0,idx-contiguous):idx+1]))
    rv5=float(np.std(past[-5:],ddof=1)*math.sqrt(252)) if len(past)>=5 else -1.
    rv21=float(np.std(past[-21:],ddof=1)*math.sqrt(252)) if len(past)>=21 else -1.
    last=float(past[-1]) if len(past) else 0.
    return np.r_[mu[idx],rv5,rv21,last,float(len(past)>=21)].astype('float32')


def train_gan(dates, x, mask, spots, mus, ends, train_end_limit, steps=16):
    transitions=[];coverage=[];conds=[]
    for t in adjacent_indices(dates):
        if t >= train_end_limit:
            continue
        both=(mask[t]>0)&(mask[t+1]>0)
        if both.sum()<12:
            continue
        dlogIV=np.zeros_like(x[t],dtype='float32')
        dlogIV[both]=np.clip(np.log(x[t+1][both]/x[t][both]),-.5,.5)
        spotmove=float(np.clip(np.log(spots[t+1]/spots[t]),-.25,.25))
        transitions.append(np.r_[spotmove,dlogIV.ravel()]);coverage.append(float(both.mean()))
        conds.append(rv_cond(dates,spots,t,mus))
    if len(transitions)<3:
        return None, {'status':'SKIPPED','reason':'Fewer than 3 observed adjacent transitions'}
    real=torch.tensor(np.stack(transitions), dtype=torch.float32)
    cond=torch.tensor(np.stack(conds), dtype=torch.float32)
    generator=VolGANGenerator(c_dim=9)
    critic=VolGANCritic(c_dim=9)
    opt_d=torch.optim.Adam(critic.parameters(),lr=1e-4,betas=(.5,.9))
    opt_g=torch.optim.Adam(generator.parameters(),lr=1e-4,betas=(.5,.9))
    n=len(real)
    history=[]
    for step in range(1,steps+1):
        indices=torch.randint(n,(min(8,n),)); y=real[indices]; c=cond[indices]
        for _ in range(2):
            z=torch.randn(len(y),32)
            ds,div=generator(z,c)
            fake=torch.cat([ds[:,None],div.flatten(1)],dim=1).detach()
            alpha=torch.rand(len(y),1)
            interp=(alpha*y+(1-alpha)*fake).requires_grad_(True)
            pred=critic(interp,c)
            grad=torch.autograd.grad(pred.sum(),interp,create_graph=True)[0]
            gp=((grad.norm(2,dim=1)-1)**2).mean()*10.
            dloss=critic(fake,c).mean()-critic(y,c).mean()+gp
            opt_d.zero_grad();dloss.backward();opt_d.step()
        ds,div=generator(torch.randn(len(y),32),c)
        fake=torch.cat([ds[:,None],div.flatten(1)],dim=1)
        gloss=-critic(fake,c).mean()
        opt_g.zero_grad();gloss.backward();opt_g.step()
        history.append({'step':step, 'critic_loss':float(dloss.detach()),
                        'generator_loss':float(gloss.detach()),
                        'gradient_penalty':float(gp.detach())})
    return generator, {'status':'EXPERIMENTAL_UNVALIDATED','transition_count':n,
                       'mean_joint_observed_fraction':round(float(np.mean(coverage)),4),
                       'rv21_condition_available':bool((cond[:,-1]>0).any().item()),
                       'last_critic_loss':float(dloss.detach()),'last_generator_loss':float(gloss.detach()),
                       'step_history':history,
                       'warning':'Missing IV shocks were zero-filled but flagged; critic is not mask-aware; not a calibrated tail distribution.'}


def walkforward_parity_proxy(raw, min_train=4):
    candidates=create_daily_candidates(raw)
    evals=evaluate_next_session(candidates,raw)
    if evals.empty:
        return pd.DataFrame(), {'folds':0,'out_of_sample_trades':0}
    dates=sorted(evals.entry_date.unique())
    rows=[]
    for idx in range(min_train,len(dates)):
        current=dates[idx]
        past=evals[(evals.entry_date < current) & (evals.next_date <= current) & evals.exit_observed]
        row=evals[evals.entry_date==current].iloc[0]
        scores=[]
        for threshold in CANDIDATE_EDGES:
            selected=past[past.estimated_net_edge_inr>=threshold]
            if len(selected)<5:
                continue
            scores.append((float(selected.pnl_net_inr.mean()),float(threshold)))
        chosen=max(scores)[1] if scores else None
        qualifies=bool(chosen is not None and row.estimated_net_edge_inr>=chosen)
        observed=bool(row.exit_observed)
        rows.append({'entry_date':pd.Timestamp(current).strftime('%Y-%m-%d'),
                     'train_last_entry':pd.Timestamp(past.entry_date.max()).strftime('%Y-%m-%d') if len(past) else None,
                     'previous_observed_trades':len(past),
                     'train_selected_threshold_inr':chosen,
                     'threshold_applied':qualifies, 'exit_observed':observed,
                     'proxy_pnl_inr':float(row.pnl_net_inr) if qualifies and observed else None})
    result=pd.DataFrame(rows)
    n=int(result.proxy_pnl_inr.notna().sum()) if len(result) else 0
    return result, {'folds':len(rows), 'out_of_sample_trades':n,
                    'out_of_sample_net_pnl_inr':round(float(result.proxy_pnl_inr.sum()),2) if n else None,
                    'selected_thresholds_validated':False,
                    'method':'Expanding training window; each next date used only after previous exit; parity proxy only'}


def run_model_research(data_dir=None, output_dir='.', vae_epochs=12,pers_epochs=12,gan_steps=16):
    if min(vae_epochs,pers_epochs,gan_steps) < 1:
        raise ValueError('VAE epochs, persistence epochs and GAN steps must all be positive.')
    seed_all()
    raw, overview = read_option_csvs(data_dir)
    dates, x, mask, spot, surface_stats=make_dataset(raw)
    ends=get_windows(dates)
    if len(ends) < 4:
        raise ValueError('At least four contiguous, labelable lookback windows are required for this exploratory training run.')
    if len(ends) < 6:
        raise ValueError('Need at least six labelable windows for train/validation/test diagnostics.')
    n_val=max(1,int(.25*len(ends)))
    n_test=max(1,int(.25*len(ends)))
    n_train_windows=len(ends)-n_val-n_test
    train_ends=ends[:n_train_windows]
    val_ends=ends[n_train_windows:n_train_windows+n_val]
    holdout_ends=ends[n_train_windows+n_val:]
    last_train_exit=train_ends[-1]+1
    last_val_exit=val_ends[-1]+1
    train_end_limit=last_train_exit
    train_snapshots=last_train_exit+1
    val_snapshots=range(last_train_exit+1,last_val_exit+1)
    test_snapshots=range(last_val_exit+1,holdout_ends[-1]+2)
    scaler_mean,scaler_std=fit_scaler(x[:train_snapshots],mask[:train_snapshots])
    xstd=standardize(x,mask,scaler_mean,scaler_std)
    residual=local_residual(x,mask)
    features=make_features(xstd,residual,mask,spot)
    vae, mu, errs, vae_loss, vae_history=train_vae(xstd,mask,train_snapshots,val_snapshots,vae_epochs)
    vae_train_metrics=vae_measure(vae,xstd,mask,range(train_snapshots))
    vae_val_metrics=vae_measure(vae,xstd,mask,val_snapshots)
    vae_test_metrics=vae_measure(vae,xstd,mask,test_snapshots)
    tr_err=errs[:train_snapshots];tr_err=tr_err[np.isfinite(tr_err)]
    median=float(np.median(tr_err)) if len(tr_err) else None
    mad=float(np.median(np.abs(tr_err-median))) if len(tr_err) else None
    exp_anomaly_cutoff=float(np.quantile(tr_err,.95)) if len(tr_err)>1 else None
    model, pers_rows, persistence_training=train_persistence(features,residual,mask,ends,train_ends,val_ends,holdout_ends,pers_epochs)
    for row in pers_rows:
        row['date']=dates[row['index']].strftime('%Y-%m-%d')
    gen, gan_training=train_gan(dates,x,mask,spot,mu,ends,train_end_limit,steps=gan_steps)
    if gen is not None:
        with torch.no_grad():
            tail_rows=[]
            for t in holdout_ends:
                c=torch.from_numpy(rv_cond(dates,spot,t,mu))[None].repeat(256,1)
                gen_spot,_=gen(torch.randn(256,32),c)
                var05=float(np.quantile(gen_spot.numpy(),.05))
                realised=float(np.log(spot[t+1]/spot[t]))
                tail_rows.append({'entry_date':dates[t].strftime('%Y-%m-%d'),
                                  'simulated_spot_5pct_log_return':var05,
                                  'observed_next_session_log_return':realised,
                                  'below_simulated_5pct':bool(realised < var05)})
        gan_training['holdout_spot_tail_diagnostics']=tail_rows
        gan_training['holdout_exception_count']=sum(int(v['below_simulated_5pct']) for v in tail_rows)
        gan_training['tail_risk_validation']='NOT_ESTABLISHED_TOO_FEW_HOLDOUTS'
    walk, walk_summary=walkforward_parity_proxy(raw)
    brier=float(np.average([v['brier'] for v in pers_rows], weights=[v['observed_cells'] for v in pers_rows])) if pers_rows else None
    train_prob = persistence_training['train_positive_cells'] / max(1, persistence_training['train_positive_cells']+persistence_training['train_negative_cells'])
    holdout_obs=sum(r['observed_cells'] for r in pers_rows)
    holdout_pos=sum(r['positives'] for r in pers_rows)
    baseline_brier=(holdout_pos*(1-train_prob)**2+(holdout_obs-holdout_pos)*train_prob**2)/holdout_obs if holdout_obs else None
    brier_skill=(1-brier/baseline_brier) if brier is not None and baseline_brier else None
    readiness = {
        'min_train_dates':MIN_VALID_TRAIN_DATES,
        'min_holdout_dates':MIN_VALID_HOLDOUT_DATES,
        'min_gan_transitions':MIN_VALID_GAN_TRANSITIONS,
        'train_dates_sufficient':len(train_ends)>=MIN_VALID_TRAIN_DATES,
        'holdout_dates_sufficient':len(holdout_ends)>=MIN_VALID_HOLDOUT_DATES,
        'gan_transitions_sufficient':gan_training.get('transition_count',0)>=MIN_VALID_GAN_TRANSITIONS,
        'rv21_available':gan_training.get('rv21_condition_available',False),
        'quote_source_executable':False,
        'can_validate_trading_thresholds':False,
    }
    report={
        'status':'RESEARCH_ONLY', 'decision':'ABORT', 'trade_authorization':'DISABLED',
        'data':{'files':overview['source_files'],'rows':overview['source_rows'],
                'dates':len(dates),'earliest':str(dates[0].date()), 'latest':str(dates[-1].date()),
                'observed_iv_cells':int(mask.sum()),'available_iv_fraction':round(float(mask.mean()),4),
                'surface_coverage_by_date':surface_stats},
        'chronology':{'lookback':LOOKBACK,'train_windows':len(train_ends), 'validation_windows':len(val_ends), 'holdout_windows':len(holdout_ends), 'test_windows':len(holdout_ends),
                      'train_last_target_date':str(dates[last_train_exit].date()),
                      'validation_entries':[str(dates[t].date()) for t in val_ends],
                      'holdout_entries':[str(dates[t].date()) for t in holdout_ends],
                      'no_labels_across_gaps':True},
        'vae':{'status':'EXPERIMENTAL_FIT','train_snapshots':train_snapshots,
               'training_last_loss':round(vae_loss,6), 'train_metrics':vae_train_metrics,
               'validation_metrics':vae_val_metrics, 'test_metrics':vae_test_metrics,
               'epoch_history':vae_history, 'nmse_definition':'Observed-cell standardized-IV SSE / SSE of training-standardized zero prediction',
               'holdout_median_reconstruction_mse':float(np.nanmedian(errs[last_val_exit+1:holdout_ends[-1]+2])) if np.isfinite(errs[last_val_exit+1:holdout_ends[-1]+2]).any() else None, 'train_median_reconstruction_mse':median,
               'train_mad_reconstruction_mse':mad, 'provisional_train_p95_error':exp_anomaly_cutoff,
               'TAU_HEALTHY_validated':None,
               'note':'Training-only error percentiles are not an out-of-sample healthy-regime threshold.'},
        'persistence':{'status':'EXPERIMENTAL_FIT','definition':'Next-session same-sign local-IV-median residual >=0.005 at both closes; observed overlapping cells only, not VG-residual','train':persistence_training,
                       'holdout_scores':pers_rows,'holdout_masked_brier':brier,
                       'validation_metrics':persistence_training['val_metrics'],
                       'test_metrics':persistence_training['test_metrics'],
                       'holdout_constant_train_prevalence_brier':baseline_brier,
                       'holdout_brier_skill_vs_constant':brier_skill,
                       'THETA_Y_validated':None},
        'wgan_gp':gan_training, 'readiness_checks':readiness,
        'walk_forward_parity_proxy':walk_summary,
        'thresholds':{k:{'value':None,'status':'NOT_VALIDATED'} for k in ['THETA_Y','TAU_HEALTHY','MIN_EDGE_INR','KAPPA','RISK_BUDGET_INR']},
        'blocking_reasons':[
            'Only 13 dates, 11 in one continuous block; train/validation/test windows are highly overlapping and statistically inadequate.',
            'No verified 21-session realised-vol history for GAN conditioning, even if missingness is explicitly marked.',
            'Daily close pairs are not synchronous or executable option/futures bid-ask quotes.',
            'Persistence target is local IV-smile proxy, not the original calibrated VG residual.',
            'Sparse 8x40 observed quote coverage and synthetic zero-fill for unobserved GAN shifts.',
            'No genuinely calibrated crash-tail model or independently validated CVaR; no volatility-regime generalisation.',
            'KAPPA and RISK_BUDGET require separately documented capital/risk mandate.',
        ],
    }
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'model_training_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(surface_stats).drop(columns='forward_estimates').to_csv(out/'surface_coverage.csv',index=False)
    pd.DataFrame(pers_rows).to_csv(out/'persistence_holdout_metrics.csv',index=False)
    pd.DataFrame(vae_history).to_csv(out/'vae_epoch_metrics.csv',index=False)
    pd.DataFrame(persistence_training['epoch_history']).to_csv(out/'persistence_epoch_metrics.csv',index=False)
    pd.DataFrame(gan_training.get('step_history',[])).to_csv(out/'gan_step_metrics.csv',index=False)
    summary=[]
    for part,key in [('train','train'),('validation','validation'),('test','test')]:
        metrics=report['vae'][key+'_metrics']
        summary.append({'model':'VAE','split':part,'windows':None,'observed_cells':metrics['observed_cells'],
                        'mse':metrics['mse'],'rmse':metrics['rmse'],'nmse':metrics['nmse'],
                        'bce':None,'brier':None})
        clskey={'train':'train','validation':'val','test':'test'}[part]
        cm=persistence_training[clskey+'_metrics']
        summary.append({'model':'CNN_LSTM','split':part,
                        'windows':len({'train':train_ends,'validation':val_ends,'test':holdout_ends}[part]),
                        'observed_cells':cm['observed_cells'],'mse':None,'rmse':None,'nmse':None,
                        'bce':cm['bce'],'brier':cm['brier']})
    pd.DataFrame(summary).to_csv(out/'metrics_summary.csv',index=False)

    walk.to_csv(out/'walkforward_threshold_proxy.csv',index=False)
    ckpt=out/'experimental_checkpoints';ckpt.mkdir(exist_ok=True)
    torch.save({'state_dict':vae.state_dict(), 'input_dim':320, 'latent_dim':5,
                'train_mean':torch.from_numpy(scaler_mean), 'train_std':torch.from_numpy(scaler_std),
                'status':'EXPERIMENTAL_UNVALIDATED'},ckpt/'vae_research.pt')
    torch.save({'state_dict':model.state_dict(),'C':5,'F_dim':8,'d':16,'a':8,
                'lookback':LOOKBACK,'status':'EXPERIMENTAL_UNVALIDATED'},ckpt/'persistence_research.pt')
    if gen is not None:
        torch.save({'state_dict':gen.state_dict(),'c_dim':9,
                    'status':'EXPERIMENTAL_UNVALIDATED'},ckpt/'wgan_generator_research.pt')
    return report


def main():
    parser=argparse.ArgumentParser(description='Research-only experimental model fitting on REAL NIFTY option closes')
    parser.add_argument('--data-dir',default=None)
    parser.add_argument('--output-dir',default='research_outputs')
    parser.add_argument('--vae-epochs',type=int,default=12)
    parser.add_argument('--persistence-epochs',type=int,default=12)
    parser.add_argument('--gan-steps',type=int,default=16)
    args=parser.parse_args()
    report=run_model_research(args.data_dir,args.output_dir,args.vae_epochs,args.persistence_epochs,args.gan_steps)
    print('Historical source:',report['data']['files'],'files,',report['data']['rows'],'rows')
    print('Observed IV cells:',report['data']['observed_iv_cells'])
    print('Supervised windows train / val / test:',report['chronology']['train_windows'],'/',report['chronology']['validation_windows'],'/',report['chronology']['test_windows'])
    print('VAE train / val / test NMSE:',*[report['vae'][part+'_metrics']['nmse'] for part in ('train','validation','test')])
    print('Persistence train / val / test BCE:',*[report['persistence']['train'][part+'_metrics']['bce'] for part in ('train','val','test')])
    print('Persistence holdout Brier (unvalidated):', report['persistence']['holdout_masked_brier'])
    print('WGAN-GP:', report['wgan_gp']['status'])
    print('Walk-forward proxy OOS trades:',report['walk_forward_parity_proxy']['out_of_sample_trades'])
    print('Decision: ABORT / trading disabled. No trading thresholds validated.')

if __name__=='__main__':main()
