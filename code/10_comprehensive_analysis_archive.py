import os, warnings, math, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial, Gaussian, Poisson
from statsmodels.stats.multitest import multipletests
from statsmodels.duration.hazard_regression import PHReg
from patsy import dmatrix
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

BASE='/mnt/data/NBA_RTP_AI_ANALYSIS_BUNDLE/extracted_csv'
OUT='/mnt/data/NBA_RTP_COMPREHENSIVE_RESULTS'
os.makedirs(OUT, exist_ok=True)

def excel_date(s):
    x=pd.to_numeric(s,errors='coerce')
    return pd.Timestamp('1899-12-30') + pd.to_timedelta(x,unit='D')

ep=pd.read_csv(f'{BASE}/injury_episodes.csv',low_memory=False)
w=pd.read_csv(f'{BASE}/rtp_selected.csv',low_memory=False)
for c in [c for c in ep.columns if 'date' in c]:
    if ep[c].dtype!='O' or ep[c].astype(str).str.match(r'^\d+(\.\d+)?$').mean()>0.8:
        ep[c+'_dt']=excel_date(ep[c])
for c in ['game_date']:
    w[c+'_dt']=excel_date(w[c])

# numeric coercion
for d in [ep,w]:
    for c in d.columns:
        if c in ['season','player_name','team','episode_id','injury_region','injury_body_part','injury_laterality','injury_diagnosis_class','next_injury_body_part','next_injury_region','next_injury_laterality','next_injury_diagnosis_class','window_phase','analysis_anchor','absence_reason_family','opponent']:
            continue
        if d[c].dtype=='O':
            # only coerce if largely numeric
            z=pd.to_numeric(d[c],errors='coerce')
            if z.notna().mean()>0.75: d[c]=z

REG_END={'2021-22':pd.Timestamp('2022-04-10'),'2022-23':pd.Timestamp('2023-04-09'),'2023-24':pd.Timestamp('2024-04-14'),'2024-25':pd.Timestamp('2025-04-13')}
FINALS_END={'2021-22':pd.Timestamp('2022-06-17'),'2022-23':pd.Timestamp('2023-06-13'),'2023-24':pd.Timestamp('2024-06-18'),'2024-25':pd.Timestamp('2025-06-23')}

prim=ep[ep.primary_analysis_eligible==1].copy()
prim['return_dt']=prim['meaningful_return_game_date_dt']
prim['injury_start_dt']=prim['injury_start_game_date_dt']
prim['reg_end']=prim.season.map(REG_END)
prim['finals_end']=prim.season.map(FINALS_END)
prim['followup_reg_days']=(prim.reg_end-prim.return_dt).dt.days
prim['followup_finals_days']=(prim.finals_end-prim.return_dt).dt.days
prim['age5']=(prim.season_age-27)/5
prim['baseline5']=(prim.pre_median_minutes_meaningful-27)/5
prim['log_games_missed']=np.log1p(prim.games_missed)
prim['return_ratio25']=prim.return_minutes_ratio/0.25
prim['aggressive100']=(prim.return_minutes_ratio>=1.0).astype(int)
prim['aggressive90']=(prim.return_minutes_ratio>=0.9).astype(int)
prim['b2b2plus']=(prim.b2b_count_first5_team_games_post>=2).astype(int)
prim['travel1000']=prim.return_travel_km/1000.0
# prior injury episodes within dataset by player
prim=prim.sort_values(['player_id','injury_start_dt','episode_id'])
prim['prior_injury_episodes']=prim.groupby('player_id').cumcount()
# collapse diagnoses for adjustment
common_dx={'sprain','soreness','strain','contusion','fracture','tightness','management','surgery_recovery','tendinopathy','inflammation','tear'}
prim['dx_adj']=prim.injury_diagnosis_class.where(prim.injury_diagnosis_class.isin(common_dx),'other')

# analysis helper
results=[]

def glm_cluster(data, outcome, exposure, covars, label, family='binomial', transform=None):
    import re
    def rawvars(term):
        # formula terms here are simple variables or C(variable)
        m=re.fullmatch(r'C\(([^)]+)\)', term)
        return [m.group(1)] if m else [term]
    raw=[outcome,'player_id']
    for term in [exposure]+covars:
        raw.extend(rawvars(term))
    cols=list(dict.fromkeys(raw))
    d=data[cols].dropna().copy()
    if len(d)==0 or d[outcome].nunique()<2:
        return None
    form=outcome+' ~ '+exposure
    if covars: form += ' + ' + ' + '.join(covars)
    fam=Binomial() if family=='binomial' else Gaussian()
    try:
        fit=smf.glm(formula=form,data=d,family=fam).fit(cov_type='cluster',cov_kwds={'groups':d.player_id})
        if exposure not in fit.params: return None
        b,se,p=fit.params[exposure],fit.bse[exposure],fit.pvalues[exposure]
        lo,hi=b-1.96*se,b+1.96*se
        if family=='binomial': est,cil,ciu=np.exp(b),np.exp(lo),np.exp(hi); scale='OR'
        else: est,cil,ciu=b,lo,hi; scale='beta'
        row=dict(label=label,outcome=outcome,exposure=exposure,n=len(d),events=int(d[outcome].sum()) if family=='binomial' else np.nan,
                 estimate=float(est),ci_low=float(cil),ci_high=float(ciu),p=float(p),scale=scale,clusters=int(d.player_id.nunique()),se_beta=float(se))
        results.append(row); return fit,row,d
    except Exception as e:
        results.append(dict(label=label,outcome=outcome,exposure=exposure,n=len(d),events=np.nan,estimate=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,scale='ERROR',clusters=d.player_id.nunique(),error=str(e)))
        return None

base_cov=['age5','baseline5','log_games_missed','prior_injury_episodes','C(injury_region)','C(dx_adj)','C(season)']
sched_cov=base_cov+['return_team_is_b2b','travel1000']

# Descriptives/correlations
ramp_cols=['return_minutes_ratio','post3_mean_minutes_ratio','post5_mean_minutes_ratio','ramp_slope_minutes_per_game','max_positive_minute_step_first5','ramp_deficit_auc_first5','games_to_90pct_baseline']
corr=prim[ramp_cols].corr(method='spearman')
corr.to_csv(f'{OUT}/paper1_ramp_spearman_correlations.csv')

print('PROGRESS 0: PAPER 1: workload restoration -> subsequent injury', flush=True)
# PAPER 1: workload restoration -> subsequent injury
p1_rows=[]
for horizon in [14,30,60]:
    d=prim[prim.followup_reg_days>=horizon].copy()
    out=f'subsequent_any_injury_absence_within_{horizon}d'
    r=glm_cluster(d,out,'return_ratio25',sched_cov,f'P1 main any injury {horizon}d')
    if r: p1_rows.append(r[1])
# recurrence proxies
for horizon,out in [(30,'separated_same_body_event_proxy_30d_ge3_played'),(60,'separated_same_body_laterality_dx_proxy_60d_ge3_played')]:
    d=prim[prim.followup_reg_days>=horizon].copy()
    r=glm_cluster(d,out,'return_ratio25',sched_cov,f'P1 recurrence proxy {horizon}d')
    if r:p1_rows.append(r[1])
# binary aggressive thresholds
for a in ['aggressive100','aggressive90']:
    d=prim[prim.followup_reg_days>=30].copy()
    r=glm_cluster(d,'subsequent_any_injury_absence_within_30d',a,sched_cov,f'P1 binary {a} 30d')
    if r:p1_rows.append(r[1])
# sensitivity strict rotation and plausible ratios
sens=prim[(prim.followup_reg_days>=30)&(prim.strict_rotation_analysis_eligible==1)&prim.return_minutes_ratio.between(.4,1.5)].copy()
r=glm_cluster(sens,'subsequent_any_injury_absence_within_30d','return_ratio25',sched_cov,'P1 strict rotation ratio .4-1.5')
if r:p1_rows.append(r[1])
# team fixed effects sensitivity
r=glm_cluster(prim[prim.followup_reg_days>=30],'subsequent_any_injury_absence_within_30d','return_ratio25',sched_cov+['C(team)'],'P1 team FE sensitivity')
if r:p1_rows.append(r[1])
# negative control outcome: next hand/finger injury within 30d
p1nc=prim[prim.followup_reg_days>=30].copy()
p1nc['next_handfinger30']=((pd.to_numeric(p1nc.days_to_next_subsequent_injury_absence,errors='coerce')<=30)&(p1nc.next_injury_body_part=='hand_finger')).astype(int)
r=glm_cluster(p1nc,'next_handfinger30','return_ratio25',sched_cov,'P1 negative-control next hand/finger 30d')
if r:p1_rows.append(r[1])

# spline vs linear for P1 30d
d=prim[prim.followup_reg_days>=30].dropna(subset=['subsequent_any_injury_absence_within_30d','return_minutes_ratio','age5','baseline5','log_games_missed','prior_injury_episodes','injury_region','dx_adj','season','return_team_is_b2b','travel1000','player_id']).copy()
try:
    lin=smf.glm('subsequent_any_injury_absence_within_30d ~ return_minutes_ratio + age5 + baseline5 + log_games_missed + prior_injury_episodes + C(injury_region)+C(dx_adj)+C(season)+return_team_is_b2b+travel1000',data=d,family=Binomial()).fit()
    spl=smf.glm('subsequent_any_injury_absence_within_30d ~ bs(return_minutes_ratio, df=4, degree=3, include_intercept=False) + age5 + baseline5 + log_games_missed + prior_injury_episodes + C(injury_region)+C(dx_adj)+C(season)+return_team_is_b2b+travel1000',data=d,family=Binomial()).fit()
    lr=2*(spl.llf-lin.llf); ddf=spl.df_model-lin.df_model; p_lr=stats.chi2.sf(lr,ddf)
    pd.DataFrame([{'n':len(d),'LR':lr,'df':ddf,'p_nonlinearity_vs_linear':p_lr,'AIC_linear':lin.aic,'AIC_spline':spl.aic}]).to_csv(f'{OUT}/paper1_spline_test.csv',index=False)
except Exception as e:
    pd.DataFrame([{'error':str(e)}]).to_csv(f'{OUT}/paper1_spline_test.csv',index=False)

print('PROGRESS 1: Cox marginal recurrent-episode model, censored at regular season end', flush=True)
# Cox marginal recurrent-episode model, censored at regular season end
cox=prim[(prim.return_dt.notna())&(prim.return_dt<=prim.reg_end)].copy()
cox['event']=((cox.next_subsequent_injury_absence_date_dt.notna())&(cox.next_subsequent_injury_absence_date_dt<=cox.reg_end)).astype(int)
cox['t_event']=(cox.next_subsequent_injury_absence_date_dt-cox.return_dt).dt.days
cox['t_cens']=(cox.reg_end-cox.return_dt).dt.days
cox['duration']=np.where(cox.event==1,cox.t_event,cox.t_cens)
cox=cox[(cox.duration>0)&cox.return_ratio25.notna()].copy()
# design
cox_cols=['return_ratio25','age5','baseline5','log_games_missed','prior_injury_episodes','return_team_is_b2b','travel1000']
X=cox[cox_cols].copy()
cat=pd.get_dummies(cox[['injury_region','dx_adj','season']],drop_first=True,dtype=float)
X=pd.concat([X,cat],axis=1).replace([np.inf,-np.inf],np.nan)
mask=X.notna().all(axis=1)&cox.duration.notna()&cox.event.notna()
try:
    ph=PHReg(cox.loc[mask,'duration'].astype(float),X.loc[mask].astype(float),status=cox.loc[mask,'event'].astype(int),ties='efron')
    phr=ph.fit(groups=cox.loc[mask,'player_id'])
    i=list(X.columns).index('return_ratio25')
    b=float(phr.params[i]); se=float(phr.bse[i]); p=float(phr.pvalues[i])
    results.append(dict(label='P1 Cox time to next injury (reg season)',outcome='time_to_next_injury',exposure='return_ratio25',n=int(mask.sum()),events=int(cox.loc[mask,'event'].sum()),estimate=float(np.exp(b)),ci_low=float(np.exp(b-1.96*se)),ci_high=float(np.exp(b+1.96*se)),p=p,scale='HR',clusters=int(cox.loc[mask,'player_id'].nunique()),se_beta=se))
except Exception as e:
    results.append(dict(label='P1 Cox time to next injury (reg season)',outcome='time_to_next_injury',exposure='return_ratio25',n=int(mask.sum()),events=int(cox.loc[mask,'event'].sum()),estimate=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,scale='ERROR',clusters=int(cox.loc[mask,'player_id'].nunique()),error=str(e)))

print('PROGRESS 2: Cluster bootstrap main P1 30d adjusted logistic', flush=True)
# Cluster bootstrap main P1 30d adjusted logistic (player-cluster resampling, frequency-weight implementation)
boot_source=prim[prim.followup_reg_days>=30].copy()
boot_formula='subsequent_any_injury_absence_within_30d ~ return_ratio25 + age5 + baseline5 + log_games_missed + prior_injury_episodes + C(injury_region)+C(dx_adj)+C(season)+return_team_is_b2b+travel1000'
boot_source=boot_source.dropna(subset=['subsequent_any_injury_absence_within_30d','return_ratio25','age5','baseline5','log_games_missed','prior_injury_episodes','injury_region','dx_adj','season','return_team_is_b2b','travel1000'])
from patsy import dmatrices
yb,Xb=dmatrices(boot_formula,boot_source,return_type='dataframe')
yb=np.asarray(yb).ravel(); Xb=np.asarray(Xb); coef_names=list(dmatrices(boot_formula,boot_source.head(10),return_type='dataframe')[1].columns)
coef_idx=coef_names.index('return_ratio25')
players_arr=boot_source.player_id.to_numpy(); uniq_players=np.unique(players_arr); pid_to_pos={p:i for i,p in enumerate(uniq_players)}
row_pid_pos=np.array([pid_to_pos[p] for p in players_arr])
rng=np.random.default_rng(20260829); boots=[]
for b in range(500):
    samp_pos=rng.integers(0,len(uniq_players),size=len(uniq_players))
    mult=np.bincount(samp_pos,minlength=len(uniq_players))
    weights=mult[row_pid_pos].astype(float)
    try:
        f=sm.GLM(yb,Xb,family=Binomial(),freq_weights=weights).fit(maxiter=60,disp=0)
        boots.append(float(np.exp(f.params[coef_idx])))
    except: pass
pd.DataFrame({'OR_per_25pct':boots}).to_csv(f'{OUT}/paper1_cluster_bootstrap_500.csv',index=False)

print('PROGRESS 3: PAPER 2: B2B / schedule congestion', flush=True)
# PAPER 2: B2B / schedule congestion
p2_rows=[]
for horizon in [14,30,60]:
    d=prim[prim.followup_reg_days>=horizon]
    cov=base_cov+['return_ratio25','travel1000']
    r=glm_cluster(d,f'subsequent_any_injury_absence_within_{horizon}d','return_team_is_b2b',cov,f'P2 return-on-B2B any injury {horizon}d')
    if r:p2_rows.append(r[1])
# first5 schedule burden per B2B and 2+
for ex in ['b2b_count_first5_team_games_post','b2b2plus']:
    d=prim[prim.followup_reg_days>=30]
    r=glm_cluster(d,'subsequent_any_injury_absence_within_30d',ex,base_cov+['return_ratio25','travel1000'],f'P2 first5 schedule {ex} 30d')
    if r:p2_rows.append(r[1])
# B2B x return workload interaction
inter=prim[prim.followup_reg_days>=30].copy()
inter['b2b_x_ratio25']=inter.return_team_is_b2b*inter.return_ratio25
r=glm_cluster(inter,'subsequent_any_injury_absence_within_30d','b2b_x_ratio25',base_cov+['return_ratio25','return_team_is_b2b','travel1000'], 'P2 interaction B2B x return workload')
if r:p2_rows.append(r[1])
# recurrence outcome
r=glm_cluster(prim[prim.followup_reg_days>=30],'separated_same_body_event_proxy_30d_ge3_played','return_team_is_b2b',base_cov+['return_ratio25','travel1000'],'P2 B2B separated same-body 30d')
if r:p2_rows.append(r[1])

print('PROGRESS 4: PAPER 5: injury-specific RTP', flush=True)
# PAPER 5: injury-specific RTP
p5_rows=[]
prim['ankle_sprain']=((prim.injury_body_part=='ankle')&(prim.injury_diagnosis_class=='sprain')).astype(int)
prim['muscle_strain']=((prim.injury_body_part.isin(['hamstring','calf','adductor_groin','quadriceps']))&(prim.injury_diagnosis_class=='strain')).astype(int)
subgroups={'ankle_sprain':prim[prim.ankle_sprain==1], 'muscle_strain':prim[prim.muscle_strain==1], 'hamstring_all':prim[prim.injury_body_part=='hamstring'], 'knee_all':prim[prim.injury_body_part=='knee']}
for name,sd in subgroups.items():
    for out,h in [('subsequent_any_injury_absence_within_30d',30),('separated_same_body_event_proxy_30d_ge3_played',30),('subsequent_same_body_part_absence_within_60d',60)]:
        dd=sd[sd.followup_reg_days>=h]
        if len(dd)>=40 and dd[out].sum()>=8:
            r=glm_cluster(dd,out,'return_ratio25',['age5','baseline5','log_games_missed','prior_injury_episodes','C(season)','return_team_is_b2b'],f'P5 {name} {out}')
            if r:p5_rows.append(r[1])

# Age effect modification P1/P2/P5
age_rows=[]
for label,ex,covs in [
    ('AgeMod P1 return workload','return_ratio25',base_cov+['return_team_is_b2b','travel1000']),
    ('AgeMod P2 return B2B','return_team_is_b2b',base_cov+['return_ratio25','travel1000'])]:
    d=prim[prim.followup_reg_days>=30].copy(); intername='age_interaction'; d[intername]=d[ex]*d.age5
    cc=[c for c in covs if c!='age5']+[ex,'age5']
    r=glm_cluster(d,'subsequent_any_injury_absence_within_30d',intername,cc,label)
    if r: age_rows.append(r[1])
# age strata P1
prim['age_group']=pd.cut(prim.season_age,bins=[0,24,29,99],labels=['<=24','25-29','>=30'])
for g,sd in prim.groupby('age_group',observed=True):
    dd=sd[sd.followup_reg_days>=30]
    r=glm_cluster(dd,'subsequent_any_injury_absence_within_30d','return_ratio25',['baseline5','log_games_missed','prior_injury_episodes','C(injury_region)','C(dx_adj)','C(season)','return_team_is_b2b'],f'Age stratum P1 {g}')
    if r: age_rows.append(r[1])

print('PROGRESS 5: causal inference helpers', flush=True)
# causal inference helpers (cross-fitted AIPW)
def aipw_binary(data,treat,outcome,numeric,categorical,groups,seed=1):
    d=data[[treat,outcome,groups]+numeric+categorical].copy().dropna(subset=[treat,outcome,groups])
    T=d[treat].astype(int).to_numpy(); Y=d[outcome].astype(float).to_numpy(); G=d[groups].to_numpy()
    # pipelines allow missing values in covariates
    pre=ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),numeric),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),categorical)
    ])
    ps=np.zeros(len(d)); m1=np.zeros(len(d)); m0=np.zeros(len(d))
    gkf=GroupKFold(n_splits=5)
    for tr,te in gkf.split(d,T,G):
        # propensity
        pm=Pipeline([('pre',pre),('lr',LogisticRegression(max_iter=2000,C=10.0))])
        pm.fit(d.iloc[tr][numeric+categorical],T[tr]); ps[te]=pm.predict_proba(d.iloc[te][numeric+categorical])[:,1]
        # outcome model includes treatment as numeric
        num2=numeric+[treat]
        pre2=ColumnTransformer([
            ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num2),
            ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),categorical)
        ])
        om=Pipeline([('pre',pre2),('lr',LogisticRegression(max_iter=2000,C=10.0))])
        om.fit(d.iloc[tr][num2+categorical],Y[tr])
        x1=d.iloc[te][num2+categorical].copy(); x1[treat]=1
        x0=d.iloc[te][num2+categorical].copy(); x0[treat]=0
        m1[te]=om.predict_proba(x1)[:,1]; m0[te]=om.predict_proba(x0)[:,1]
    ps=np.clip(ps,.05,.95)
    phi1=m1+T*(Y-m1)/ps; phi0=m0+(1-T)*(Y-m0)/(1-ps)
    mu1=phi1.mean(); mu0=phi0.mean(); rd=mu1-mu0; rr=mu1/mu0 if mu0>0 else np.nan
    infl=(phi1-phi0)-rd; se=infl.std(ddof=1)/np.sqrt(len(d)); p=2*stats.norm.sf(abs(rd/se)) if se>0 else np.nan
    return {'n':len(d),'treated':int(T.sum()),'mu1':mu1,'mu0':mu0,'RD':rd,'RR':rr,'RD_se':se,'RD_ci_low':rd-1.96*se,'RD_ci_high':rd+1.96*se,'p':p,'ps_min':ps.min(),'ps_max':ps.max(),'auc_ps':roc_auc_score(T,ps)}, d, ps

causal=[]
causal_num=['season_age','pre_median_minutes_meaningful','log_games_missed','prior_injury_episodes','return_team_is_b2b','return_travel_km']
causal_cat=['injury_region','dx_adj','season','team']
d=prim[prim.followup_reg_days>=30].copy()
try:
    a,_,_=aipw_binary(d,'aggressive100','subsequent_any_injury_absence_within_30d',causal_num,causal_cat,'player_id')
    a['analysis']='P1 cross-fitted AIPW aggressive >=100% baseline'; causal.append(a)
except Exception as e: causal.append({'analysis':'P1 AIPW','error':str(e)})
try:
    nums=['season_age','pre_median_minutes_meaningful','log_games_missed','prior_injury_episodes','return_minutes_ratio','return_travel_km']
    a,_,_=aipw_binary(d,'return_team_is_b2b','subsequent_any_injury_absence_within_30d',nums,causal_cat,'player_id')
    a['analysis']='P2 cross-fitted AIPW return on B2B'; causal.append(a)
except Exception as e: causal.append({'analysis':'P2 AIPW','error':str(e)})
pd.DataFrame(causal).to_csv(f'{OUT}/causal_aipw_results.csv',index=False)

print('PROGRESS 6: PAPER 3: pre-injury workload patterns', flush=True)
# PAPER 3: pre-injury workload patterns from 10 played games before injury
# restrict to primary episodes with full 10 preinjury played games
wp=w[w.episode_id.isin(set(prim.episode_id)) & (w.window_phase=='preinjury_played')].copy()
# merge injury labels and age
lab=prim[['episode_id','player_id','injury_body_part','injury_region','injury_diagnosis_class','season_age','season','pre_median_minutes_meaningful']].drop_duplicates('episode_id')
wp=wp.merge(lab,on='episode_id',how='left',suffixes=('','_ep'))
# episode-level summaries
pre_summ=[]
for eid,g in wp.groupby('episode_id'):
    g=g.sort_values('relative_game')
    if set(range(-10,0)).issubset(set(g.relative_game.astype(int))):
        gg=g.set_index(g.relative_game.astype(int))
        first5=gg.loc[[-10,-9,-8,-7,-6]]; last5=gg.loc[[-5,-4,-3,-2,-1]]
        mins=g.sort_values('relative_game').minutes.astype(float).to_numpy()
        pre_summ.append({
            'episode_id':eid,'player_id':g.player_id.iloc[0],'season':g.season.iloc[0],
            'injury_body_part':g.injury_body_part.iloc[0],'injury_region':g.injury_region.iloc[0],'injury_diagnosis_class':g.injury_diagnosis_class.iloc[0],
            'season_age':g.season_age.iloc[0],
            'mean_first5':first5.minutes.mean(),'mean_last5':last5.minutes.mean(),'delta_last5_first5':last5.minutes.mean()-first5.minutes.mean(),
            'mean_last3':gg.loc[[-3,-2,-1]].minutes.mean(),'mean_prev3':gg.loc[[-6,-5,-4]].minutes.mean(),
            'delta_last3_prev3':gg.loc[[-3,-2,-1]].minutes.mean()-gg.loc[[-6,-5,-4]].minutes.mean(),
            'slope10':np.polyfit(np.arange(10),mins,1)[0],
            'b2b_first5':first5.team_is_b2b.sum(),'b2b_last5':last5.team_is_b2b.sum(),'b2b_delta':last5.team_is_b2b.sum()-first5.team_is_b2b.sum(),
            'travel_first5':first5.travel_km_since_prev.sum(),'travel_last5':last5.travel_km_since_prev.sum(),'travel_delta':last5.travel_km_since_prev.sum()-first5.travel_km_since_prev.sum(),
            'lastgame_prev5_mean':gg.loc[-1,'prev5_team_games_mean_minutes'] if -1 in gg.index else np.nan,
            'lastgame_prev20_mean':gg.loc[-1,'prev20_team_games_mean_minutes'] if -1 in gg.index else np.nan,
            'lastgame_acwr':gg.loc[-1,'acute_chronic_minutes_7d_28d'] if -1 in gg.index else np.nan,
        })
pre=pd.DataFrame(pre_summ)
pre.to_csv(f'{OUT}/paper3_preinjury_episode_features.csv',index=False)
# paired tests + cluster bootstrap means
p3=[]
for name,dd in {
    'all':pre,
    'lower_extremity':pre[pre.injury_region=='lower_extremity'],
    'muscle_soft_tissue':pre[pre.injury_body_part.isin(['hamstring','calf','adductor_groin','quadriceps'])],
    'hand_finger_negative_control':pre[pre.injury_body_part=='hand_finger'],
    'head_concussion_negative_control':pre[pre.injury_body_part=='head_concussion']
}.items():
    for metric in ['delta_last5_first5','delta_last3_prev3','b2b_delta','travel_delta']:
        v=dd[metric].dropna()
        if len(v)>=10:
            t,p=stats.ttest_1samp(v,0)
            wil=stats.wilcoxon(v) if (v!=0).any() else (np.nan,np.nan)
            p3.append({'group':name,'metric':metric,'n':len(v),'mean':v.mean(),'median':v.median(),'sd':v.std(),'t_p':p,'wilcoxon_p':wil.pvalue if hasattr(wil,'pvalue') else wil[1]})
# cluster bootstrap mean delta overall and soft tissue (player-level sufficient statistics)
for name,dd in [('all',pre),('muscle_soft_tissue',pre[pre.injury_body_part.isin(['hamstring','calf','adductor_groin','quadriceps'])])]:
    agg=dd.groupby('player_id').delta_last5_first5.agg(['sum','count'])
    sums=agg['sum'].to_numpy(float); cnts=agg['count'].to_numpy(float); npl=len(agg); vals=[]; rng2=np.random.default_rng(777)
    for b in range(1000):
        sp=rng2.integers(0,npl,size=npl); mult=np.bincount(sp,minlength=npl)
        vals.append(float((mult*sums).sum()/(mult*cnts).sum()))
    p3.append({'group':name,'metric':'delta_last5_first5_cluster_bootstrap','n':len(dd),'mean':dd.delta_last5_first5.mean(),'ci_low':np.quantile(vals,.025),'ci_high':np.quantile(vals,.975)})
pd.DataFrame(p3).to_csv(f'{OUT}/paper3_preinjury_tests.csv',index=False)
# specificity: soft-tissue vs traumatic negative controls as outcome
spec=pre[pre.injury_body_part.isin(['hamstring','calf','adductor_groin','quadriceps','hand_finger','head_concussion'])].copy()
spec['load_sensitive'] = spec.injury_body_part.isin(['hamstring','calf','adductor_groin','quadriceps']).astype(int)
spec['age5']=(spec.season_age-27)/5
if spec.load_sensitive.nunique()==2:
    rr=glm_cluster(spec,'load_sensitive','delta_last5_first5',['age5','C(season)'],'P3 workload-rise specificity soft tissue vs traumatic control')

print('PROGRESS 7: PAPER 4 performance / availability tradeoff', flush=True)
# PAPER 4 performance / availability tradeoff
post=w[w.episode_id.isin(set(prim.episode_id))].copy()
post=post.merge(prim[['episode_id','player_id','return_minutes_ratio','season_age','games_missed','pre_median_minutes_meaningful','injury_region','dx_adj','season','followup_reg_days']].drop_duplicates('episode_id'),on='episode_id',how='left',suffixes=('','_ep'))
# per36 only for played >5 min
for statc in ['points','assists','reboundsTotal','plusMinusPoints']:
    post[statc+'_per36']=np.where(post.minutes>=5,post[statc]/post.minutes*36,np.nan)
perf=[]
for eid,g in post.groupby('episode_id'):
    pre_g=g[(g.relative_game<0)&(g.played_actual==1)&(g.minutes>=5)]
    ret=g[(g.relative_game==0)&(g.played_actual==1)&(g.minutes>=5)]
    p15=g[(g.relative_game.between(1,5))&(g.played_actual==1)&(g.minutes>=5)]
    p610=g[(g.relative_game.between(6,10))&(g.played_actual==1)&(g.minutes>=5)]
    p010=g[g.relative_game.between(0,10)]
    if len(pre_g)>=5:
        row={'episode_id':eid,'player_id':g.player_id.iloc[0],'return_minutes_ratio':g.return_minutes_ratio.iloc[0], 'season_age':g.season_age.iloc[0], 'games_missed':g.games_missed.iloc[0], 'baseline_minutes':g.pre_median_minutes_meaningful.iloc[0], 'injury_region':g.injury_region.iloc[0], 'dx_adj':g.dx_adj.iloc[0], 'season':g.season.iloc[0]}
        for statc in ['points_per36','assists_per36','reboundsTotal_per36','plusMinusPoints_per36','trueShootingPercentage']:
            row['pre_'+statc]=pre_g[statc].mean()
            row['ret_'+statc]=ret[statc].mean() if len(ret) else np.nan
            row['post1_5_'+statc]=p15[statc].mean() if len(p15) else np.nan
            row['post6_10_'+statc]=p610[statc].mean() if len(p610) else np.nan
            row['delta1_5_'+statc]=row['post1_5_'+statc]-row['pre_'+statc] if pd.notna(row['post1_5_'+statc]) else np.nan
        row['team_games_played_0_10']=int((p010.played_actual==1).sum())
        row['injury_absences_0_10']=int(((p010.played_actual==0)&(p010.absence_reason_family=='injury')).sum())
        row['total_minutes_0_10']=p010.minutes.fillna(0).sum()
        row['total_points_0_10']=p010.points.fillna(0).sum()
        perf.append(row)
perf=pd.DataFrame(perf)
perf['age5']=(perf.season_age-27)/5; perf['baseline5']=(perf.baseline_minutes-27)/5; perf['log_games_missed']=np.log1p(perf.games_missed); perf['return_ratio25']=perf.return_minutes_ratio/.25
perf.to_csv(f'{OUT}/paper4_performance_episode_features.csv',index=False)
p4_rows=[]
for statc in ['points_per36','assists_per36','reboundsTotal_per36','plusMinusPoints_per36','trueShootingPercentage']:
    out='delta1_5_'+statc
    r=glm_cluster(perf,out,'return_ratio25',['age5','baseline5','log_games_missed','C(injury_region)','C(season)'],f'P4 performance recovery {statc}',family='gaussian')
    if r:p4_rows.append(r[1])
# availability/count proxy as continuous total minutes/points and played games over next 10 team games
for out in ['total_minutes_0_10','total_points_0_10','team_games_played_0_10','injury_absences_0_10']:
    r=glm_cluster(perf,out,'return_ratio25',['age5','baseline5','log_games_missed','C(injury_region)','C(season)'],f'P4 availability {out}',family='gaussian')
    if r:p4_rows.append(r[1])
# age modification of return workload on performance/availability
for out in ['delta1_5_points_per36','delta1_5_trueShootingPercentage','total_minutes_0_10']:
    dd=perf.copy(); dd['age_interaction']=dd.return_ratio25*dd.age5
    r=glm_cluster(dd,out,'age_interaction',['return_ratio25','age5','baseline5','log_games_missed','C(injury_region)','C(season)'],f'AgeMod P4 {out}',family='gaussian')
    if r:age_rows.append(r[1])

print('PROGRESS 8: target-trial-like first post-RTP B2B descriptive/adjusted', flush=True)
# target-trial-like first post-RTP B2B descriptive/adjusted
# identify first B2B second-leg after RTP in relative games 1..10, exclude if already injured at that B2B; compare played vs nonmedical sit
trial=[]
for eid,g in post.groupby('episode_id'):
    q=g[(g.relative_game>=1)&(g.relative_game<=10)&(g.team_is_b2b==1)].sort_values('relative_game')
    if q.empty: continue
    r0=q.iloc[0]
    if r0.absence_reason_family in ['injury','illness','g_league','roster_transaction']: continue
    # require previous team game row exists and player played it
    prev=g[g.relative_game==r0.relative_game-1]
    if prev.empty or int(prev.iloc[0].played_actual)!=1: continue
    treated=int(r0.played_actual==1) # played second leg
    # future injury in next 5 team games after the B2B decision
    fut=g[(g.relative_game>r0.relative_game)&(g.relative_game<=r0.relative_game+5)]
    y=int(((fut.played_actual==0)&(fut.absence_reason_family=='injury')).any())
    trial.append({'episode_id':eid,'player_id':g.player_id.iloc[0],'played_second_leg':treated,'future_injury_next5':y,'return_minutes_ratio':g.return_minutes_ratio.iloc[0],'season_age':g.season_age.iloc[0],'games_missed':g.games_missed.iloc[0],'baseline_minutes':g.pre_median_minutes_meaningful.iloc[0],'injury_region':g.injury_region.iloc[0],'season':g.season.iloc[0],'decision_relative_game':r0.relative_game,'second_leg_reason':r0.absence_reason_family})
trial=pd.DataFrame(trial)
trial.to_csv(f'{OUT}/paper2_target_trial_first_b2b.csv',index=False)
if len(trial)>20 and trial.played_second_leg.nunique()==2 and trial.future_injury_next5.sum()>5:
    trial['age5']=(trial.season_age-27)/5; trial['baseline5']=(trial.baseline_minutes-27)/5; trial['log_games_missed']=np.log1p(trial.games_missed); trial['return_ratio25']=trial.return_minutes_ratio/.25
    rr=glm_cluster(trial,'future_injury_next5','played_second_leg',['age5','baseline5','log_games_missed','return_ratio25','C(injury_region)','C(season)'],'P2 target-trial-like play second leg of first B2B')

print('PROGRESS 9: Multiple-testing q-values by paper family', flush=True)
# Multiple-testing q-values by paper family
res=pd.DataFrame(results)
def family(label):
    if str(label).startswith('P1'): return 'P1'
    if str(label).startswith('P2'): return 'P2'
    if str(label).startswith('P3'): return 'P3'
    if str(label).startswith('P4'): return 'P4'
    if str(label).startswith('P5'): return 'P5'
    if str(label).startswith('Age'): return 'AGE'
    return 'OTHER'
res['family']=res.label.map(family)
res['q_fdr']=np.nan
for fam,ix in res.groupby('family').groups.items():
    pvals=res.loc[ix,'p']
    good=pvals.notna()
    if good.sum(): res.loc[pvals[good].index,'q_fdr']=multipletests(pvals[good],method='fdr_bh')[1]
# approximate MDE OR/HR using design SE and alpha .05/power .80
z=stats.norm.ppf(.975)+stats.norm.ppf(.8)
res['mde_ratio_80pct']=np.where(res.scale.isin(['OR','HR']),np.exp(z*res.se_beta),np.nan)
res['convincing']=((res.p<.05)&(res.q_fdr<.05)&((res.ci_low>1)|(res.ci_high<1)))
res['nominal_positive']=((res.p<.05)&((res.ci_low>1)|(res.ci_high<1)))
res.to_csv(f'{OUT}/all_model_results.csv',index=False)

# Bootstrap summary
bs=pd.Series(boots)
boot_summary=pd.DataFrame([{'n_boot':len(bs),'median_OR':bs.median(),'ci_low':bs.quantile(.025),'ci_high':bs.quantile(.975),'pct_gt1':(bs>1).mean()}])
boot_summary.to_csv(f'{OUT}/paper1_bootstrap_summary.csv',index=False)

# Cohort/event counts & subgroup power context
counts=[]
for h in [14,30,60]:
    dd=prim[prim.followup_reg_days>=h]
    counts.append({'cohort':f'primary complete {h}d reg-season followup','n':len(dd),'players':dd.player_id.nunique(),'events_any':int(dd[f"subsequent_any_injury_absence_within_{h}d"].sum())})
for name,sd in subgroups.items():
    dd=sd[sd.followup_reg_days>=30]
    counts.append({'cohort':name,'n':len(dd),'players':dd.player_id.nunique(),'events_any':int(dd.subsequent_any_injury_absence_within_30d.sum()),'events_sep_samebody':int(dd.separated_same_body_event_proxy_30d_ge3_played.sum())})
pd.DataFrame(counts).to_csv(f'{OUT}/cohort_event_counts.csv',index=False)

# compact markdown report automatically generated
lines=[]
lines.append('# NBA RTP comprehensive statistical analysis\n')
lines.append(f'Primary eligible episodes: {len(prim)} across {prim.player_id.nunique()} players.\n')
lines.append('## Model results\n')
show=res.sort_values(['family','q_fdr','p']).copy()
for fam in ['P1','P2','P3','P4','P5','AGE']:
    q=show[show.family==fam]
    if q.empty: continue
    lines.append(f'### {fam}\n')
    for _,r in q.iterrows():
        est='NA' if pd.isna(r.estimate) else f"{r.estimate:.3f}"
        ci='NA' if pd.isna(r.ci_low) else f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
        lines.append(f"- {r.label}: {r.scale} {est} {ci}; p={r.p:.4g} q={r.q_fdr:.4g}; n={int(r.n)}" + (' **CONVINCING**' if r.convincing else '') + '\n')
lines.append('\n## AIPW causal sensitivity\n')
for a in causal:
    lines.append('- '+str(a)+'\n')
lines.append('\n## Bootstrap\n')
lines.append(boot_summary.to_markdown(index=False)+'\n')
lines.append('\n## Paper 3 pre-injury descriptive tests\n')
lines.append(pd.DataFrame(p3).to_markdown(index=False)+'\n')
open(f'{OUT}/SUMMARY.md','w').write('\n'.join(lines))

print('DONE',OUT)
print(res[['family','label','scale','estimate','ci_low','ci_high','p','q_fdr','n','events','convincing']].sort_values(['family','q_fdr','p']).to_string(index=False))
print('\nAIPW')
print(pd.DataFrame(causal).to_string(index=False))
print('\nBOOT',boot_summary.to_dict('records'))
print('\nP3')
print(pd.DataFrame(p3).to_string(index=False))
print('\nTRIAL',trial.shape, trial.played_second_leg.value_counts().to_dict() if len(trial) else {}, trial.future_injury_next5.value_counts().to_dict() if len(trial) else {})
