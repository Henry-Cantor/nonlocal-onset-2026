#!/usr/bin/env python3
"""Three kill-shots, the overstatement fixes, and the required
statistics, in one run on the real fleet (reduced_v2). Uses the exact Paper-1 label_and_clean.

  K1  coherence vs reconstruction: reconstruction radial correlation length (recon_resid_te and
      quiet residual, in grid cells and in physical units) vs the onset band-width distribution.
  K2  true no-gradient control: drop every R/L_Te-derived feature (not only the Q_GB set).
  K3  coherent vs incoherent among cells already above the amplitude threshold.
  + physics baselines, coherence-width scan, cleaning scan, per-fold LOMO spread, cluster-bootstrap
    CIs, pooled all-regime fit, and the Theta predicted-vs-fitted sign table.
Writes killshots_results.json and paper1_k3_plotdata.json (the K3 figures render in figures/make_figures.ipynb).
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
from onset_common import (fit_logistic, predict_logistic, roc_auc, coherent_extent,
                          machine_weights, pooled_loso, regime_filter)

AGM = "/Users/henrycantor/mentorship2/ACTUALLY_GOOD_MAYBE"
BASE = ['accel','prate_frac','inv_etae','etae_surge','betae','gExB_hat','ge','TeTi','gExB','load8',
        'rstar','corrug','overdrive','gmc','prox','nustar','crit']
GEM  = ['net_ratio','eff_drive','uni_drive','u_peak']
POWER = BASE + GEM
QGB   = ['ge','betae','rstar','TeTi','gExB_hat']                    # Q_GB-building (current control)
RLTE  = ['accel','overdrive','gmc','prox','crit','corrug','load8']  # further R/L_Te-derived (K2)
DROP_NOGRAD = QGB + RLTE
NOGRAD = [f for f in POWER if f not in DROP_NOGRAD]                 # genuinely non-(R/L_Te) predictors
THETA = 2.0
COLS = list(dict.fromkeys(['machine','shot','regime','q','time','rho','Q_cond_e','Q_GB',
                           'recon_resid_te','ring_te'] + POWER))
PHYS5 = ['rstar','eff_drive','betae','gExB_hat','nustar']
# theory (Theta) sign of the onset response, per physical group (higher value -> more onset)
THETA_SIGN = {'rstar':+1,'eff_drive':+1,'ge':+1,'overdrive':+1,'crit':-1,'betae':-1,
              'gExB_hat':-1,'gExB':-1,'nustar':-1}


def log1(v): return np.log(np.maximum(np.abs(np.asarray(v,float)),1e-12))


def label(sub):
    r = np.clip(sub['Q_cond_e'].values / np.maximum(sub['Q_GB'].values,1e-9), 0, 1e4)
    shot = pd.factorize(sub['shot'].values)[0]; mach = pd.factorize(sub['machine'].values)[0]
    c0 = pd.Series(r).groupby(mach).transform('median').values
    resid = r / np.maximum(THETA*c0,1e-6) - 1.0
    ext = coherent_extent(resid > 0.4, shot, sub['time'].values, sub['rho'].values)
    y = (resid > 1.0) & (ext >= 4)
    rr, rg = sub['recon_resid_te'].values, sub['ring_te'].values
    clean = (rr <= np.nanquantile(rr,0.75)) & (rg <= np.nanquantile(rg,0.75))
    keep = clean & ((resid < 0.4) | y)
    return resid, ext, y, keep, clean, shot, mach


def cell_runlength(above, shot, time, rho):
    """Per-cell length of the contiguous above-threshold radial run it belongs to (vectorized)."""
    a = np.asarray(above, bool); tr = np.round(np.asarray(time,float),5)
    order = np.lexsort((rho, tr, shot)); a_s = a[order]; s_s = shot[order]; t_s = tr[order]
    n = len(a_s)
    gchg = np.empty(n,bool); gchg[0]=True; gchg[1:] = (s_s[1:]!=s_s[:-1])|(t_s[1:]!=t_s[:-1])
    brk = np.empty(n,bool); brk[0]=True; brk[1:] = gchg[1:] | (a_s[1:]!=a_s[:-1])
    rid = np.cumsum(brk); counts = np.bincount(rid)
    length_s = counts[rid].astype(float); length_s[~a_s] = 0.0
    out = np.zeros(n); out[order] = length_s
    return out


def radial_corr_len(field, shot, time, rho, mask, maxlag=8, cap=6000, seed=0):
    """Mean radial autocorrelation over (shot,time) profiles entirely inside `mask`; returns the
    curve and the lag where it first drops below 1/e (reconstruction correlation length, cells)."""
    df = pd.DataFrame({'s':shot,'t':np.round(time,5),'rho':rho,'v':field,'m':np.asarray(mask,bool)})
    groups = [g for _,g in df.groupby(['s','t']) if len(g)>=8 and g['m'].all()]
    if len(groups) > cap:
        rng = np.random.default_rng(seed); groups=[groups[i] for i in rng.choice(len(groups),cap,replace=False)]
    ac = np.zeros(maxlag+1); nn = np.zeros(maxlag+1)
    for g in groups:
        v = g.sort_values('rho')['v'].values.astype(float); v = v-v.mean()
        if v.std()<1e-9: continue
        v/=v.std()
        for L in range(0,min(maxlag,len(v)-1)+1):
            ac[L]+=np.mean(v[:len(v)-L]*v[L:]); nn[L]+=1
    ac = ac/np.maximum(nn,1)
    cl = next((L for L in range(1,maxlag+1) if ac[L] < 1/np.e), maxlag+1)
    return ac, cl, int(nn[0])


def propagation(y, resid, shot, time, rho):
    """Radial group velocity of onset bands: for each shot, the onset-weighted band centroid rho at
    each time, differenced between consecutive onset-containing slices. Returns drho/dt (a-units/s;
    negative = inward). Weighted by residual so the hottest surface dominates the centroid."""
    df = pd.DataFrame({'s':shot,'t':np.asarray(time,float),'rho':rho,'y':np.asarray(y,bool),
                       'wgt':np.maximum(resid,0.0)})
    df = df[df['y']]
    vel=[]
    for _,g in df.groupby('s'):
        cen = g.groupby('t').apply(lambda h: np.average(h['rho'], weights=h['wgt']+1e-9)).sort_index()
        if len(cen) < 2: continue
        dt = np.diff(cen.index.values); dr = np.diff(cen.values)
        ok = (np.abs(dt) > 0) & (np.abs(dt) < 0.02)          # consecutive slices only (<20 ms gap)
        vel.extend((dr[ok]/dt[ok]).tolist())
    return np.array(vel)


def design(sub, idx, cols):
    return np.column_stack([log1(sub[c].values[idx].astype(float)) for c in cols])


def loso_oof(X, y, shot, mach, cap=140000, seed=0):
    """Out-of-fold leave-one-shot-out predictions + labels + shot ids (for cluster bootstrap)."""
    rng = np.random.default_rng(seed); n=len(y)
    pool = rng.choice(n,cap,replace=False) if n>cap else np.arange(n)
    Xp,yp,sp,mp = X[pool],y[pool],shot[pool],mach[pool]
    order = np.argsort(shot,kind='stable'); shs=shot[order]
    uniq,first = np.unique(shs,return_index=True); bounds=np.append(first,len(shs))
    P,Y,S=[],[],[]
    for k in range(len(uniq)):
        te = order[bounds[k]:bounds[k+1]]
        if len(te)<40 or not (0<y[te].mean()<1): continue
        keep = sp!=uniq[k]
        if keep.sum()<200 or not (0<yp[keep].mean()<1): continue
        w,mu,sd = fit_logistic(Xp[keep],yp[keep].astype(float),l2=2.0,wts=machine_weights(mp[keep]))
        P.append(predict_logistic(X[te],w,mu,sd)); Y.append(y[te].astype(bool)); S.append(np.full(len(te),uniq[k]))
    if not P: return None,None,None
    return np.concatenate(P),np.concatenate(Y),np.concatenate(S)


def boot_ci(pred, y, cluster, B=500, seed=0):
    """95% CI for AUC by resampling CLUSTERS (shots or machines) with replacement."""
    if pred is None: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); uc=np.unique(cluster)
    idxby={c:np.where(cluster==c)[0] for c in uc}
    aucs=[]
    for _ in range(B):
        pick=rng.choice(uc,len(uc),replace=True)
        ii=np.concatenate([idxby[c] for c in pick])
        a=roc_auc(pred[ii],y[ii])
        if np.isfinite(a): aucs.append(a)
    lo,hi=np.percentile(aucs,[2.5,97.5]) if aucs else (np.nan,np.nan)
    return (float(roc_auc(pred,y)),float(lo),float(hi))


def lomo_folds(X, y, mach, names, cap=250000, seed=0):
    """Per-machine leave-one-machine-out AUCs."""
    rng=np.random.default_rng(seed); out={}
    for m in np.unique(mach):
        te=mach==m; tr=np.where(mach!=m)[0]
        if te.sum()<500 or not (0<y[te].mean()<1) or not (0<y[tr].mean()<1): continue
        samp=rng.choice(tr,min(cap,len(tr)),replace=False)
        w,mu,sd=fit_logistic(X[samp],y[samp].astype(float),l2=2.0,wts=machine_weights(mach[samp]))
        a=roc_auc(predict_logistic(X[te],w,mu,sd),y[te])
        if np.isfinite(a): out[str(names[m])]=float(a)
    return out


def main():
    report={}; pool={'X2':[],'y':[],'shot':[],'mach':[],'so':0,'mo':0}
    k3_plot={}
    for regime in ['conv_NBI','spherical','conv_0torque']:
        ush = pq.read_table(f"{AGM}/reduced_v2.parquet",columns=['regime','shot'],
                            filters=regime_filter(regime)).to_pandas()['shot'].unique()
        rng=np.random.default_rng(0)
        # 80 shots/regime is the memory-safe ceiling here (this script holds several design matrices +
        # bootstrap resamples at once; 110 OOMs). The kill-shots are RELATIVE comparisons (full vs
        # no-gradient vs no-Q_GB, coherence/cleaning scans) that are stable at 80 shots and carry
        # shot-cluster bootstrap CIs. For an all-shots rerun use a bigger-RAM host (Stellar).
        NKS=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--shots=')),80))
        ks = ush if len(ush)<=NKS else rng.choice(ush,NKS,replace=False)
        sub = pq.read_table(f"{AGM}/reduced_v2.parquet",columns=COLS,
                            filters=regime_filter(regime)+[('shot','in',list(ks))]).to_pandas()
        if regime!='conv_NBI': sub=sub[sub['q'].values>1.5].reset_index(drop=True)
        resid,ext,y,keep,clean,shot,mach = label(sub)
        names = pd.factorize(sub['machine'].values)[1]
        ki=np.where(keep)[0]; yk=y[ki].astype(bool); shk=shot[ki]; mck=mach[ki]
        R={'n_cells':int(len(ki)),'n_machines':int(len(np.unique(mck))),'onset_pct':float(yk.mean()*100)}
        print(f"\n===== {regime}  ({R['n_machines']} machines, {len(ki):,} cells, onset {R['onset_pct']:.1f}%) =====")

        # ---- K1: reconstruction correlation length vs onset band width ----
        rl = cell_runlength(resid>0.4, shot, sub['time'].values, sub['rho'].values)
        onband = rl[y]                                                     # band width (cells) at onset cells
        _,cl_recon,nrp = radial_corr_len(sub['recon_resid_te'].values,shot,sub['time'].values,sub['rho'].values,
                                         np.ones(len(sub),bool))
        _,cl_quiet,_ = radial_corr_len(resid,shot,sub['time'].values,sub['rho'].values,resid<0.4)
        # physical units per machine: drho (=Delta_r/a, since rho=r/a), rho_i/a ~ rstar
        drho=[]; rhoi=[]
        for m in np.unique(mach):
            mm=mach==m; rr=np.sort(np.unique(np.round(sub['rho'].values[mm],4)))
            if len(rr)>3: drho.append(np.median(np.diff(rr)))
            rhoi.append(np.nanmedian(np.abs(sub['rstar'].values[mm])))
        drho=float(np.nanmedian(drho)); rhoi=float(np.nanmedian(rhoi))
        band_a = float(np.nanmedian(onband))*drho                          # band width in units of a
        band_rhoi = band_a/max(rhoi,1e-9)                                  # band width in units of rho_i
        R['K1']={'recon_corr_len_cells':cl_recon,'quiet_resid_corr_len_cells':cl_quiet,
                 'onset_band_median_cells':float(np.nanmedian(onband)),
                 'onset_band_p90_cells':float(np.nanpercentile(onband,90)),
                 'grid_drho_over_a':drho,'rho_i_over_a':rhoi,
                 'onset_band_a':band_a,'onset_band_rho_i':band_rhoi,
                 'PASS':bool(max(cl_recon,cl_quiet) < 4)}
        print(f"  K1: recon corr len {cl_recon} cells, quiet-resid {cl_quiet} cells; onset band median "
              f"{np.nanmedian(onband):.1f} cells = {band_a:.3f} a = {band_rhoi:.0f} rho_i.  "
              f"{'PASS (coherence > reconstruction)' if R['K1']['PASS'] else 'FLAG'}")

        # ---- Q_GB definition robustness: drop the R/L_Te (=ge) factor and re-derive the label ----
        qgb_alt = np.maximum(sub['Q_GB'].values,1e-9)/np.maximum(np.abs(sub['ge'].values),1e-6)
        ra = np.clip(sub['Q_cond_e'].values/qgb_alt,0,1e4)
        c0a = pd.Series(ra).groupby(mach).transform('median').values
        resid_a = ra/np.maximum(THETA*c0a,1e-6)-1.0
        ext_a = coherent_extent(resid_a>0.4, shot, sub['time'].values, sub['rho'].values)
        y_alt = (resid_a>1.0)&(ext_a>=4)
        jacc = float((y&y_alt).sum()/max((y|y_alt).sum(),1))
        auc_alt = float(pooled_loso(design(sub,ki,POWER), y_alt[ki].astype(bool), shk, mck, cap=100000)) \
                  if 0<y_alt[ki].mean()<1 else np.nan
        R['QGB_robust']={'jaccard':jacc,'onset_alt_pct':float(y_alt.mean()*100),'auc_alt':auc_alt}
        print(f"  Q_GB robustness (drop R/L_Te): onset Jaccard {jacc:.2f}, alt-onset {y_alt.mean()*100:.1f}%, "
              f"law AUC on alt {auc_alt:.3f}")

        # ---- avalanche propagation direction + speed ----
        vel = propagation(y, resid, shot, sub['time'].values, sub['rho'].values)
        if len(vel) > 50:
            R['propagation']={'n':int(len(vel)),'inward_frac':float(np.mean(vel<0)),
                              'median_speed_a_per_s':float(np.nanmedian(np.abs(vel)))}
            print(f"  propagation: {np.mean(vel<0)*100:.0f}% inward, median speed "
                  f"{np.nanmedian(np.abs(vel)):.1f} a/s (n={len(vel)})")

        # ---- K2: full / current-noQGB / TRUE no-gradient, with shot-cluster-bootstrap CIs ----
        R['K2']={}
        for name,cols in [('full21',POWER),('current_dropQGB',[f for f in POWER if f not in QGB]),
                          ('true_no_gradient',NOGRAD)]:
            X=design(sub,ki,cols); p,yy,ss=loso_oof(X,yk,shk,mck)
            R['K2'][name]=dict(zip(('auc','lo','hi'),boot_ci(p,yy,ss)))
        k2=R['K2']
        print(f"  K2 LOSO[95% shot-CI]: full {k2['full21']['auc']:.3f}[{k2['full21']['lo']:.3f}-{k2['full21']['hi']:.3f}] "
              f"| dropQGB {k2['current_dropQGB']['auc']:.3f}[{k2['current_dropQGB']['lo']:.3f}-{k2['current_dropQGB']['hi']:.3f}] "
              f"| no-grad {k2['true_no_gradient']['auc']:.3f}[{k2['true_no_gradient']['lo']:.3f}-{k2['true_no_gradient']['hi']:.3f}]")
        print(f"     no-gradient features: {NOGRAD}")

        # ---- K3: among amplitude-exceeding cells, coherent (ext>=4) vs incoherent ----
        hi = (resid>1.0) & clean
        yc = (ext>=4)[hi]
        R['K3']={'n_high':int(hi.sum()),'coherent_frac':float(yc.mean()) if hi.sum() else np.nan}
        if hi.sum()>4000 and 0.02<yc.mean()<0.98:
            hidx=np.where(hi)[0]
            Xc=design(sub,hidx,[f for f in POWER if f not in QGB])         # drivers only -> answers K2 too
            p,yy,ss=loso_oof(Xc,yc,shot[hidx],mach[hidx])
            R['K3'].update(dict(zip(('auc','lo','hi'),boot_ci(p,yy,ss))))
            print(f"  K3 coherent-vs-incoherent (high-amplitude, drivers only): AUC {R['K3']['auc']:.3f}"
                  f"[{R['K3']['lo']:.3f}-{R['K3']['hi']:.3f}], coherent {yc.mean()*100:.0f}%.  "
                  f"{'PASS (predicts non-locality)' if R['K3']['auc']>0.6 else 'WEAK'}")
            # save for the amplitude-vs-coherence plot (subsample)
            amp = resid[hidx]; wc,muc,sdc=fit_logistic(Xc,yc.astype(float),l2=2.0)
            pc = 1/(1+np.exp(-np.clip(predict_logistic(Xc,wc,muc,sdc),-40,40)))
            take=np.random.default_rng(0).choice(len(hidx),min(20000,len(hidx)),replace=False)
            k3_plot[regime]={'amp':amp[take].tolist(),'coh':yc[take].astype(int).tolist(),
                             'rstar':np.abs(sub['rstar'].values[hidx][take]).tolist(),'pred':pc[take].tolist()}
        else:
            print(f"  K3 skipped (high cells {hi.sum()}, coherent {yc.mean()*100:.0f}%)")

        # ---- physics baselines (single-feature + 2-term critical-gradient excess) ----
        R['baselines']={}
        for name,cols in [('overdrive',['overdrive']),('ge',['ge']),('accel',['accel']),
                          ('crit_excess',['overdrive','crit'])]:
            R['baselines'][name]=float(pooled_loso(design(sub,ki,cols),yk,shk,mck,cap=120000))
        b=R['baselines']
        print(f"  baselines LOSO: overdrive {b['overdrive']:.3f}, ge {b['ge']:.3f}, accel {b['accel']:.3f}, "
              f"crit-excess(2-term) {b['crit_excess']:.3f}  vs full {k2['full21']['auc']:.3f}")

        # ---- coherence-width scan (run-length >= m) and cleaning scan ----
        R['coherence_scan']={}; Xfull=design(sub,ki,POWER)
        for m in [3,4,5,6]:
            ym=((resid>1.0)&(rl>=m))[ki]
            R['coherence_scan'][m]=float(pooled_loso(Xfull,ym.astype(bool),shk,mck,cap=120000)) if 0<ym.mean()<1 else np.nan
        print(f"  coherence-width scan LOSO (m=3..6): "
              + ", ".join(f"m>={m}:{R['coherence_scan'][m]:.3f}" for m in [3,4,5,6]))
        R['cleaning_scan']={}
        rr,rg=sub['recon_resid_te'].values,sub['ring_te'].values
        for q in [0.5,0.75,0.9,1.01]:
            cl_=(rr<=np.nanquantile(rr,min(q,1.0)))&(rg<=np.nanquantile(rg,min(q,1.0)))
            kq=np.where(cl_&((resid<0.4)|y))[0]
            if len(kq)>5000 and 0<y[kq].mean()<1:
                R['cleaning_scan'][f'p{int(q*100)}']=float(pooled_loso(design(sub,kq,POWER),
                                    y[kq].astype(bool),shot[kq],mach[kq],cap=120000))
        print(f"  cleaning scan LOSO: "+", ".join(f"{k}:{v:.3f}" for k,v in R['cleaning_scan'].items()))

        # ---- Theta predicted-vs-fitted sign (phys5 multivariate) ----
        wp,mup,sdp=fit_logistic(design(sub,ki,PHYS5),yk.astype(float),l2=2.0,wts=machine_weights(mck))
        fit_sign={PHYS5[j]:int(np.sign(wp[j+1])) for j in range(len(PHYS5))}
        agree=[f for f in PHYS5 if f in THETA_SIGN and fit_sign[f]==THETA_SIGN[f]]
        R['theta_signs']={'fitted':fit_sign,'theory':{f:THETA_SIGN[f] for f in PHYS5 if f in THETA_SIGN},
                          'agreement':f"{len(agree)}/{sum(f in THETA_SIGN for f in PHYS5)}"}
        print(f"  Theta signs (phys5) agree {R['theta_signs']['agreement']}: fitted {fit_sign}")

        # accumulate pooled dom2
        X2=design(sub,ki,['rstar','eff_drive'])
        pool['X2'].append(X2); pool['y'].append(yk); pool['shot'].append(shk+pool['so'])
        pool['mach'].append(mck+pool['mo']); pool['so']+=int(shk.max())+1; pool['mo']+=int(mck.max())+1
        report[regime]=R

    # ---- POOLED all-regime dom2 (O1) ----
    X2=np.vstack(pool['X2']); y2=np.concatenate(pool['y']); sh2=np.concatenate(pool['shot']); mc2=np.concatenate(pool['mach'])
    w,mu,sd=fit_logistic(X2,y2.astype(float),l2=2.0,wts=machine_weights(mc2))
    a,b=w[1]/sd[0],w[2]/sd[1]
    p,yy,ss=loso_oof(X2,y2,sh2,mc2,cap=250000); auc_p,lo_p,hi_p=boot_ci(p,yy,ss)
    folds=lomo_folds(X2,y2,mc2,{i:i for i in range(mc2.max()+1)})
    fv=np.array(list(folds.values()))
    print(f"\n===== POOLED all-regime dom2 =====")
    print(f"  onset ~ rho*^({a:+.2f}) * eff_drive^({b:+.2f});  pooled LOSO {auc_p:.3f}[{lo_p:.3f}-{hi_p:.3f}];")
    print(f"  LOMO per-fold mean {fv.mean():.3f} spread [{fv.min():.3f}-{fv.max():.3f}] over {len(fv)} machines.")
    report['pooled_dom2']={'rho_exp':float(a),'eff_exp':float(b),'loso_auc':auc_p,'loso_ci':[lo_p,hi_p],
                           'lomo_mean':float(fv.mean()),'lomo_min':float(fv.min()),'lomo_max':float(fv.max()),
                           'lomo_n':int(len(fv))}
    json.dump(report, open(f"{AGM}/killshots_results.json","w"), indent=2)
    json.dump(k3_plot, open(f"{AGM}/paper1_k3_plotdata.json","w"))
    print(f"\nwrote killshots_results.json + paper1_k3_plotdata.json")


if __name__ == '__main__':
    main()
