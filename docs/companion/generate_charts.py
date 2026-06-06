"""
Generate all companion charts and save to docs/companion/charts/.
Run from docs/companion/ directory.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy import stats
from scipy.stats import ttest_ind, norm, pearsonr, beta as beta_dist, mannwhitneyu, chisquare

CHARTS = "charts"
os.makedirs(CHARTS, exist_ok=True)

TUFTE = {
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'font.family': 'sans-serif',
    'axes.labelcolor': '#333', 'text.color': '#333',
    'xtick.color': '#333', 'ytick.color': '#333',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}
mpl.rcParams.update(TUFTE)

def save(name):
    plt.savefig(f"{CHARTS}/{name}", dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ {name}")

print("=== 00-foundations ===")

# ── counterfactual ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 10
users = [f"U{i+1}" for i in range(n)]
y_treatment = rng.normal(loc=120, scale=20, size=n)
y_control   = rng.normal(loc=100, scale=20, size=n)
observed    = np.where(np.arange(n) < 5, y_treatment, y_control)
counterfact = np.where(np.arange(n) < 5, y_control,   y_treatment)
in_treatment = np.arange(n) < 5

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(n)
ax.bar(x[in_treatment],  observed[in_treatment],  color='#4C72B0', label='Observed (treatment)', width=0.5)
ax.bar(x[~in_treatment], observed[~in_treatment], color='#DD8452', label='Observed (control)',   width=0.5)
ax.scatter(x, counterfact, color='#999', zorder=5, s=60, label='Counterfactual (never observed)', marker='x', linewidths=2)
ax.set_xticks(x); ax.set_xticklabels(users)
ax.set_ylabel('30-day revenue ($)')
ax.set_title('Observed outcomes vs. counterfactuals', loc='left', fontweight='bold')
ax.legend(frameon=False)
ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("counterfactual.png")

# ── positivity overlap ──────────────────────────────────────────────────────
rng = np.random.default_rng(0); n = 400
age_ctrl = rng.exponential(18, n); age_trt  = rng.exponential(20, n)
age_trt_poor = rng.exponential(6, n)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (trt, label) in zip(axes, [
    (age_trt,      'Good overlap  (positivity ✓)'),
    (age_trt_poor, 'Poor overlap  (positivity ✗)'),
]):
    ax.hist(age_ctrl, bins=30, alpha=0.55, color='#DD8452', label='Control')
    ax.hist(trt,      bins=30, alpha=0.55, color='#4C72B0', label='Treatment')
    ax.set_title(label, loc='left', fontweight='bold')
    ax.set_xlabel('Account age (months)'); ax.legend(frameon=False)
    ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
axes[0].set_ylabel('Users')
plt.suptitle('Positivity check: covariate overlap across variants', fontweight='bold')
plt.tight_layout(); save("positivity_overlap.png")

# ── dag examples ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
def draw_dag(ax, title, nodes, edges, red_edges=None):
    ax.set_title(title, loc='left', fontweight='bold'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    pos = dict(nodes)
    for (s,d) in edges:
        x0,y0=pos[s]; x1,y1=pos[d]
        c = '#e63946' if red_edges and (s,d) in red_edges else '#666'
        ax.annotate('', xy=(x1,y1), xytext=(x0,y0), arrowprops=dict(arrowstyle='->', color=c, lw=2))
    for name,(x,y) in pos.items():
        ax.text(x,y,name,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4',facecolor='#f5f5f5',edgecolor='#bbb'))

draw_dag(axes[0],'Confounding — account size biases comparison',
    [('T',(0.15,0.45)),('Y',(0.85,0.45)),('C',(0.5,0.82))],
    [('T','Y'),('C','T'),('C','Y')], red_edges={('C','T'),('C','Y')})
axes[0].text(0.15,0.30,'Feature\nexposure',ha='center',fontsize=8,color='#555')
axes[0].text(0.85,0.30,'Revenue',ha='center',fontsize=8,color='#555')
axes[0].text(0.50,0.95,'Account size\n(confounder)',ha='center',fontsize=8,color='#e63946')

draw_dag(axes[1],'Collider bias — filtering converted users opens spurious path',
    [('T',(0.12,0.50)),('Y',(0.88,0.50)),('Col',(0.50,0.18)),('Q',(0.50,0.82))],
    [('T','Y'),('Q','Y'),('T','Col'),('Q','Col')], red_edges={('T','Col'),('Q','Col')})
axes[1].text(0.12,0.35,'Feature',ha='center',fontsize=8,color='#555')
axes[1].text(0.88,0.35,'LTV',ha='center',fontsize=8,color='#555')
axes[1].text(0.50,0.04,'Converted\n(collider – do not filter)',ha='center',fontsize=8,color='#e63946')
axes[1].text(0.50,0.95,'User quality',ha='center',fontsize=8,color='#555')
plt.tight_layout(); save("dag_examples.png")

# ── association vs causation ────────────────────────────────────────────────
rng = np.random.default_rng(7); n=300
account_size = rng.exponential(10, n)
prob_adopt   = 1/(1+np.exp(-(-1+0.15*account_size)))
adopted      = rng.binomial(1, prob_adopt)
revenue      = 50*account_size + rng.normal(0,30,n)
fig, axes = plt.subplots(1,2,figsize=(11,4))
axes[0].hist(revenue[adopted==0],bins=30,alpha=0.6,color='#DD8452',label=f'Non-adopters  μ=${revenue[adopted==0].mean():.0f}')
axes[0].hist(revenue[adopted==1],bins=30,alpha=0.6,color='#4C72B0',label=f'Adopters       μ=${revenue[adopted==1].mean():.0f}')
axes[0].set_title('Naive comparison (association)',loc='left',fontweight='bold')
axes[0].set_xlabel('Revenue ($)'); axes[0].legend(frameon=False)
from numpy.polynomial.polynomial import polyfit
for grp,color,label in [(0,'#DD8452','Non-adopter'),(1,'#4C72B0','Adopter')]:
    mask = adopted==grp
    axes[1].scatter(account_size[mask],revenue[mask],alpha=0.25,s=10,color=color)
    c = polyfit(account_size[mask],revenue[mask],1)
    xl = np.linspace(account_size[mask].min(),account_size[mask].max(),100)
    axes[1].plot(xl,c[0]+c[1]*xl,color=color,lw=2,label=label)
axes[1].set_title('After conditioning on account size\n(effect disappears)',loc='left',fontweight='bold')
axes[1].set_xlabel('Account size'); axes[1].set_ylabel('Revenue ($)'); axes[1].legend(frameon=False)
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("association_vs_causation.png")

print("=== 01-design ===")

# ── randomization unit ──────────────────────────────────────────────────────
rng=np.random.default_rng(42); n_acc=200; ups=5; icc=0.15
acc_fx  = rng.normal(0,np.sqrt(icc),n_acc)
u_noise = rng.normal(0,np.sqrt(1-icc),(n_acc,ups))
y       = (acc_fx[:,None]+u_noise).flatten()
variant = np.repeat(rng.choice([0,1],size=n_acc),ups)
y_obs   = y + 0.2*variant
acc_ids = np.repeat(np.arange(n_acc),ups)
_,p_usr = ttest_ind(y_obs[variant==1],y_obs[variant==0])
acc_means = np.array([y_obs[acc_ids==i].mean() for i in range(n_acc)])
acc_var   = np.repeat([0,1],n_acc//2)
_,p_acc   = ttest_ind(acc_means[acc_var==1],acc_means[acc_var==0])
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,(p,label,note) in zip(axes,[
    (p_usr,f'User-level analysis\np={p_usr:.4f}  ← inflated',f'n={len(y_obs):,} users'),
    (p_acc,f'Account-level analysis\np={p_acc:.4f}  ← correct',f'n={n_acc} accounts'),
]):
    c='#e63946' if 'inflated' in label else '#2a9d8f'
    for grp,col,lab in [(0,'#DD8452','Control'),(1,'#4C72B0','Treatment')]:
        d=y_obs[variant==grp] if 'User' in label else acc_means[acc_var==grp]
        ax.hist(d,bins=25,alpha=0.55,color=col,density=True,label=lab)
    ax.set_title(label,loc='left',fontweight='bold',color=c)
    ax.text(0.97,0.95,note,ha='right',va='top',transform=ax.transAxes,fontsize=9,color='#888')
    ax.legend(frameon=False); ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.suptitle('Randomized by account — must aggregate before testing',fontweight='bold')
plt.tight_layout(); save("randomization_unit.png")

# ── time zero timeline ──────────────────────────────────────────────────────
import matplotlib.dates as mdates
from datetime import datetime
fig,ax=plt.subplots(figsize=(11,3.5)); ax.axis('off')
t0=datetime(2024,3,15); start=datetime(2024,3,1); end=datetime(2024,4,14)
def d2n(d): return mdates.date2num(d)
ax.barh(0.6,d2n(t0)-d2n(start),left=d2n(start),height=0.22,color='#a8dadc',label='Baseline window')
ax.barh(0.6,d2n(end)-d2n(t0),left=d2n(t0),height=0.22,color='#457b9d',label='Outcome window')
ax.axvline(d2n(t0),color='#e63946',lw=2.5,zorder=5)
ax.text(d2n(t0),0.90,'  Time zero\n  (assignment)',color='#e63946',va='top',fontsize=9,fontweight='bold')
ax.text((d2n(start)+d2n(t0))/2,0.58,'Baseline\n(pre-experiment)',ha='center',va='top',fontsize=9,color='#333')
ax.text((d2n(t0)+d2n(end))/2,0.58,'Outcome window',ha='center',va='top',fontsize=9,color='#fff')
contam=datetime(2024,3,10)
ax.axvline(d2n(contam),color='#f4a261',lw=1.5,linestyle='--')
ax.text(d2n(contam),0.76,'  ⚠ Soft launch\n  contaminates baseline',color='#f4a261',fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.set_yticks([]); ax.set_ylim(0.3,1.1)
ax.set_title('Experiment timeline: baseline, time zero, outcome window',loc='left',fontweight='bold')
ax.legend(frameon=False,loc='lower right'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("time_zero.png")

# ── metric taxonomy ─────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(10,5)); ax.axis('off')
levels = [
    ('Primary\n(ship/kill)',  '#2a9d8f', ['Trial conversion rate']),
    ('Secondary\n(diagnostic)','#457b9d',['Feature adoption','Session depth','Support tickets']),
    ('Guardrail\n(must not break)','#e9c46a',['7-day churn rate','p95 error rate','Revenue/account']),
    ('Proxy\n(leading indicator)','#f4a261',['D7 retention  →  proxy for D90 LTV']),
]
for i,(label,color,metrics) in enumerate(levels):
    y=0.80-i*0.20
    ax.add_patch(mpatches.FancyBboxPatch((0.04,y-0.08),0.92,0.14,
        boxstyle='round,pad=0.01',facecolor=color,alpha=0.15,edgecolor=color,lw=1.5))
    ax.text(0.09,y,label,va='center',fontsize=9,fontweight='bold',color=color)
    ax.text(0.38,y,' · '.join(metrics),va='center',fontsize=9,color='#444')
ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.set_title('Metric taxonomy — one primary, then the rest',loc='left',fontweight='bold')
plt.tight_layout(); save("metric_taxonomy.png")

# ── power analysis ──────────────────────────────────────────────────────────
def ss_binary(p,mde_rel,alpha=0.05,power=0.80):
    p1=p*(1+mde_rel); pb=(p+p1)/2; d=p1-p
    return int(np.ceil(2*pb*(1-pb)*(norm.ppf(1-alpha/2)+norm.ppf(power))**2/d**2))
mdes=np.linspace(0.02,0.30,50)
n_req=[ss_binary(0.05,m) for m in mdes]
sizes=np.arange(500,30000,200)
powers_curve=[]
pc,pt=0.05,0.05*1.15
for n in sizes:
    se=np.sqrt(pc*(1-pc)/n+pt*(1-pt)/n)
    z=(pt-pc)/se-norm.ppf(0.975)
    powers_curve.append(norm.cdf(z))
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(mdes*100,n_req,color='#4C72B0',lw=2)
for y_,lbl in [(5000,'5K'),(50000,'50K')]:
    axes[0].axhline(y_,color='#ccc',lw=1,linestyle='--')
    axes[0].text(28,y_*1.03,f'{lbl}/variant',ha='right',fontsize=8,color='#999')
axes[0].set_xlabel('Relative MDE (%)'); axes[0].set_ylabel('Required n per variant')
axes[0].set_title('Binary metric (base 5%)\nRequired n vs. MDE',loc='left',fontweight='bold')
axes[1].plot(sizes,powers_curve,color='#2a9d8f',lw=2)
axes[1].axhline(0.80,color='#e63946',lw=1.5,linestyle='--',label='80% power')
axes[1].axhline(0.90,color='#f4a261',lw=1.5,linestyle='--',label='90% power')
axes[1].set_xlabel('Sample size per variant'); axes[1].set_ylabel('Power')
axes[1].set_title('Power curve — 15% relative lift, base 5%',loc='left',fontweight='bold')
axes[1].legend(frameon=False)
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("power_analysis.png")

# ── mde business value ──────────────────────────────────────────────────────
mau=5000; arpa=200
days=np.arange(7,91,1)
mde_vals=[2*(norm.ppf(0.975)+norm.ppf(0.80))*np.sqrt(0.05*0.95/(d*mau/2)) for d in days]
arr_vals=[m*mau*12*arpa for m in mde_vals]
fig,ax1=plt.subplots(figsize=(10,4)); ax2=ax1.twinx()
ax1.plot(days,[m*100 for m in mde_vals],color='#4C72B0',lw=2,label='MDE (pp)')
ax2.plot(days,[a/1e6 for a in arr_vals],color='#e63946',lw=2,linestyle='--',label='ARR impact ($M)')
for d,lbl in [(14,'14d'),(30,'30d')]:
    ax1.axvline(d,color='#ccc',lw=1,linestyle=':')
    ax1.text(d+0.5,ax1.get_ylim()[1]*0.93,lbl,fontsize=8,color='#999')
ax1.set_xlabel('Experiment runtime (days)'); ax1.set_ylabel('MDE (pp)',color='#4C72B0')
ax2.set_ylabel('Min detectable ARR impact ($M)',color='#e63946')
ax1.set_title(f'MDE and ARR impact vs. runtime\n(base 5%, {mau:,} eligible/day, ARPA ${arpa})',loc='left',fontweight='bold')
l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2,lb1+lb2,frameon=False)
ax1.spines['bottom'].set_color('#ddd'); ax1.spines['left'].set_color('#ddd')
plt.tight_layout(); save("mde_business_value.png")

print("=== 02-data-quality ===")

# ── SRM ─────────────────────────────────────────────────────────────────────
def srm_test(obs,exp_pct):
    tot=sum(obs); exp=[tot*p for p in exp_pct]
    chi2,p=chisquare(obs,exp); return chi2,p,exp
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,(obs,title) in zip(axes,[
    ([10012,9988],'No SRM  (p > 0.05)'),
    ([10850,9150],'SRM detected  (p < 0.001) — STOP'),
]):
    chi2,p,exp=srm_test(obs,[0.5,0.5])
    x=np.arange(2)
    ax.bar(x,obs,width=0.4,color='#4C72B0',alpha=0.8,label='Observed')
    ax.bar(x+0.4,exp,width=0.4,color='#DD8452',alpha=0.8,label='Expected')
    for i,(o,e) in enumerate(zip(obs,exp)):
        ax.text(i,o+80,f'{o:,}',ha='center',fontsize=9)
        ax.text(i+0.4,e+80,f'{e:,.0f}',ha='center',fontsize=9,color='#DD8452')
    c='#2a9d8f' if p>0.05 else '#e63946'
    ax.set_title(f'{title}\nχ²={chi2:.1f}, p={p:.4f}',loc='left',fontweight='bold',color=c)
    ax.set_xticks(x+0.2); ax.set_xticklabels(['Control','Treatment'])
    ax.legend(frameon=False); ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("srm_detection.png")

# ── ICC / DEFF ───────────────────────────────────────────────────────────────
icc_vals=[0.0,0.05,0.10,0.20,0.30,0.40]; m_bar=5
deffs=[1+(m_bar-1)*v for v in icc_vals]; eff_n=[500/d for d in deffs]
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(range(6),deffs,color='#4C72B0',alpha=0.8)
axes[0].axhline(1,color='#ccc',lw=1,linestyle='--')
axes[0].set_xticks(range(6)); axes[0].set_xticklabels([f'ρ={v}' for v in icc_vals])
axes[0].set_ylabel('Design effect (DEFF)')
axes[0].set_title(f'DEFF = 1+(m̄−1)×ρ   (m̄={m_bar})',loc='left',fontweight='bold')
for i,d in enumerate(deffs): axes[0].text(i,d+0.02,f'{d:.2f}×',ha='center',fontsize=9)
axes[1].bar(range(6),eff_n,color='#2a9d8f',alpha=0.8)
axes[1].axhline(500,color='#e63946',lw=1.5,linestyle='--',label='Nominal n=500')
axes[1].set_xticks(range(6)); axes[1].set_xticklabels([f'ρ={v}' for v in icc_vals])
axes[1].set_ylabel('Effective sample size'); axes[1].legend(frameon=False)
axes[1].set_title('Effective n shrinks as ICC grows',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("icc_deff.png")

# ── network interference ────────────────────────────────────────────────────
rng=np.random.default_rng(99); n=500
nbrs=[rng.choice([j for j in range(n) if j!=i],size=3,replace=False) for i in range(n)]
treatment=rng.binomial(1,0.5,n)
neighbor_tx=np.array([treatment[nbrs[i]].mean() for i in range(n)])
y=10+2.0*treatment+0.8*neighbor_tx+rng.normal(0,1,n)
X=np.column_stack([np.ones(n),treatment,neighbor_tx])
beta=np.linalg.lstsq(X,y,rcond=None)[0]
fig,axes=plt.subplots(1,2,figsize=(11,4))
ctrl=treatment==0
axes[0].scatter(neighbor_tx[ctrl],y[ctrl],alpha=0.3,s=12,color='#DD8452')
xl=np.linspace(0,1,50)
axes[0].plot(xl,beta[0]+beta[2]*xl,color='#e63946',lw=2,label=f'β_spillover={beta[2]:.2f}')
axes[0].set_xlabel('Fraction of neighbors in treatment'); axes[0].set_ylabel('Outcome (control users)')
axes[0].set_title('Spillover in control arm',loc='left',fontweight='bold'); axes[0].legend(frameon=False)
naive=y[treatment==1].mean()-y[treatment==0].mean()
effs=[naive,beta[1],beta[2]]; lbls=['Naive ITT','Direct\n(OLS)','Spillover\n(OLS)']
axes[1].barh(lbls,effs,color=['#aaa','#4C72B0','#f4a261'],alpha=0.85)
axes[1].axvline(0,color='#333',lw=1)
axes[1].set_xlabel('Estimated effect'); axes[1].set_title('Decomposing direct vs spillover effects',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("network_interference.png")

# ── selection bias ──────────────────────────────────────────────────────────
rng=np.random.default_rng(3); n=2000
hi=rng.binomial(1,0.4,n); treat=rng.binomial(1,0.5,n)
rev=10+15*hi+3.0*treat+rng.normal(0,5,n)
p_trig=np.clip(0.2+0.4*hi+0.1*treat,0,1); triggered=rng.binomial(1,p_trig)
eff_trig=rev[(treat==1)&(triggered==1)].mean()-rev[(treat==0)&(triggered==1)].mean()
eff_itt =rev[treat==1].mean()-rev[treat==0].mean()
fig,ax=plt.subplots(figsize=(8,4))
bars=ax.bar(['Triggered only\n(biased)','ITT — all assigned\n(correct)'],[eff_trig,eff_itt],
            color=['#e63946','#2a9d8f'],width=0.4,alpha=0.85)
ax.axhline(3.0,color='#333',lw=1.5,linestyle='--',label='True effect = 3.0')
for b,v in zip(bars,[eff_trig,eff_itt]): ax.text(b.get_x()+b.get_width()/2,v+0.05,f'{v:.2f}',ha='center',fontsize=11,fontweight='bold')
ax.set_ylabel('Estimated treatment effect'); ax.legend(frameon=False)
ax.set_title('Trigger-based selection inflates the effect estimate',loc='left',fontweight='bold')
ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("selection_bias.png")

# ── novelty / primacy ───────────────────────────────────────────────────────
rng=np.random.default_rng(12); days=np.arange(1,29)
nov=3.0*np.exp(-0.3*days)+1.0+rng.normal(0,0.3,28)
prim=-1.5*np.exp(-0.25*days)+1.2+rng.normal(0,0.3,28)
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(days,nov,color='#4C72B0',lw=2,marker='o',markersize=3)
axes[0].axhline(1.0,color='#e63946',lw=1.5,linestyle='--',label='Steady-state = 1.0')
for d,lbl in [(7,'Day 7'),(14,'Day 14')]:
    axes[0].axvline(d,color='#ccc',lw=1,linestyle=':'); axes[0].text(d+0.2,nov.max()*0.97,lbl,fontsize=8,color='#999')
axes[0].set_title('Novelty effect — day-1 is inflated',loc='left',fontweight='bold')
axes[0].set_xlabel('Day in experiment'); axes[0].set_ylabel('Treatment − control'); axes[0].legend(frameon=False)
axes[1].plot(days,prim,color='#2a9d8f',lw=2,marker='o',markersize=3)
axes[1].axhline(1.2,color='#e63946',lw=1.5,linestyle='--',label='True long-run = 1.2')
axes[1].axhline(0,color='#ccc',lw=0.8)
axes[1].set_title('Primacy effect — day-1 is pessimistic',loc='left',fontweight='bold')
axes[1].set_xlabel('Day in experiment'); axes[1].legend(frameon=False)
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("novelty_primacy.png")

# ── survivor bias ───────────────────────────────────────────────────────────
rng=np.random.default_rng(55); n=3000
quality=rng.normal(0,1,n); treat=rng.binomial(1,0.5,n)
p_surv=1/(1+np.exp(-(0.5*quality-0.3*treat))); survived=rng.binomial(1,p_surv)
eng=5+2*quality+rng.normal(0,1,n)
surv_by_v=[survived[treat==0].mean(),survived[treat==1].mean()]
eng_surv_eff=eng[(treat==1)&(survived==1)].mean()-eng[(treat==0)&(survived==1)].mean()
eng_itt_eff=(eng*survived)[treat==1].mean()-(eng*survived)[treat==0].mean()
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(['Control','Treatment'],surv_by_v,color=['#DD8452','#4C72B0'],width=0.4,alpha=0.85)
axes[0].set_ylim(0,1); axes[0].set_ylabel('Fraction surviving to analysis window')
axes[0].set_title('Survival rates differ by variant',loc='left',fontweight='bold')
for i,v in enumerate(surv_by_v): axes[0].text(i,v+0.01,f'{v:.3f}',ha='center',fontsize=10)
axes[1].bar(['Survivor-filtered\n(biased)','ITT\n(correct)'],[eng_surv_eff,eng_itt_eff],
            color=['#e63946','#2a9d8f'],width=0.4,alpha=0.85)
axes[1].axhline(0,color='#333',lw=1,linestyle='--',label='True effect = 0')
axes[1].set_ylabel('Estimated effect on engagement'); axes[1].legend(frameon=False)
axes[1].set_title('Survivor filter inflates positive effect\n(true effect = 0)',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("survivor_bias.png")

print("=== 03-metric-pitfalls ===")

# ── Simpson's paradox ────────────────────────────────────────────────────────
data={'SMB':{'ctrl':(200,1000),'trt':(30,200)},'Enterprise':{'ctrl':(50,200),'trt':(400,1000)}}
ctrl_r=[data[s]['ctrl'][0]/data[s]['ctrl'][1] for s in data]
trt_r =[data[s]['trt'][0]/data[s]['trt'][1]  for s in data]
ctrl_agg=sum(d['ctrl'][0] for d in data.values())/sum(d['ctrl'][1] for d in data.values())
trt_agg =sum(d['trt'][0]  for d in data.values())/sum(d['trt'][1]  for d in data.values())
ctrl_r.append(ctrl_agg); trt_r.append(trt_agg); segs=list(data.keys())+['Aggregate']
fig,axes=plt.subplots(1,2,figsize=(11,4))
x=np.arange(3); w=0.35
axes[0].bar(x-w/2,[r*100 for r in ctrl_r],w,color='#DD8452',alpha=0.85,label='Control')
axes[0].bar(x+w/2,[r*100 for r in trt_r], w,color='#4C72B0',alpha=0.85,label='Treatment')
axes[0].axvline(1.5,color='#ddd',lw=1)
axes[0].text(2,max(max(ctrl_r),max(trt_r))*100*0.92,'Aggregate\nreversed ↑',ha='center',fontsize=8,color='#e63946')
axes[0].set_xticks(x); axes[0].set_xticklabels(segs); axes[0].set_ylabel('Conversion rate (%)')
axes[0].set_title("Simpson's paradox\nTreatment worse per segment, better overall",loc='left',fontweight='bold')
axes[0].legend(frameon=False)
cs_pct=data['SMB']['ctrl'][1]/sum(d['ctrl'][1] for d in data.values())
ts_pct=data['SMB']['trt'][1] /sum(d['trt'][1]  for d in data.values())
axes[1].bar(['Control','Treatment'],[cs_pct*100,ts_pct*100],color='#a8dadc',alpha=0.85,label='SMB')
axes[1].bar(['Control','Treatment'],[(1-cs_pct)*100,(1-ts_pct)*100],
            bottom=[cs_pct*100,ts_pct*100],color='#457b9d',alpha=0.85,label='Enterprise')
axes[1].set_ylabel('Composition (%)'); axes[1].legend(frameon=False)
axes[1].set_title('Treatment arm dominated by Enterprise\n(higher base rate → aggregate reversal)',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("simpsons_paradox.png")

# ── heavy tails ──────────────────────────────────────────────────────────────
rng=np.random.default_rng(7); n=1000
def sim_rev(n,seed):
    r=np.random.default_rng(seed)
    base=r.lognormal(5,1.5,n); w=r.binomial(1,0.02,n)*r.exponential(50000,n); return base+w
ctrl=sim_rev(n,1); trt=sim_rev(n,2)
trt_w=trt.copy(); trt_w[rng.integers(n)]+=300000
p99=np.percentile(np.concatenate([ctrl,trt_w]),99)
ctrl_w=np.clip(ctrl,0,p99); trtw=np.clip(trt_w,0,p99)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,(d,lbl,col) in zip(axes[:2],[(ctrl,'Control','#DD8452'),(trt_w,'Treatment','#4C72B0')]):
    ax.hist(np.log1p(d),bins=40,color=col,alpha=0.7)
    ax.set_xlabel('log(1+revenue)'); ax.set_title(f'{lbl}\nμ=${d.mean():,.0f}  p99=${np.percentile(d,99):,.0f}',loc='left',fontweight='bold')
    ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
raw_diff=trt_w.mean()-ctrl.mean(); wins_diff=trtw.mean()-ctrl_w.mean()
mw_stat,_=mannwhitneyu(trt_w,ctrl); mw_norm=(mw_stat/(n*n)-0.5)*ctrl.mean()
axes[2].barh(['Raw mean\ndiff','Winsorized\nmean diff','Mann-Whitney\n(robust)'],
             [raw_diff,wins_diff,mw_norm],color=['#e63946','#2a9d8f','#4C72B0'],alpha=0.85)
axes[2].axvline(0,color='#333',lw=1)
axes[2].set_title('One whale flips raw mean\nWinsorization is robust',loc='left',fontweight='bold')
axes[2].spines['left'].set_color('#ddd'); axes[2].spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("heavy_tails.png")

# ── ratio metrics ────────────────────────────────────────────────────────────
rng=np.random.default_rng(5); n=2000
treat=rng.binomial(1,0.5,n)
sess=rng.negative_binomial(2,0.3,n)+1; rev=2*sess+rng.exponential(5,n)
ul_ctrl=(rev[treat==0]/sess[treat==0]).mean(); ul_trt=(rev[treat==1]/sess[treat==1]).mean()
pl_ctrl=rev[treat==0].sum()/sess[treat==0].sum(); pl_trt=rev[treat==1].sum()/sess[treat==1].sum()
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].scatter(sess[treat==0],rev[treat==0],alpha=0.15,s=8,color='#DD8452',label='Control')
axes[0].scatter(sess[treat==1],rev[treat==1],alpha=0.15,s=8,color='#4C72B0',label='Treatment')
axes[0].set_xlabel('Sessions'); axes[0].set_ylabel('Revenue ($)')
axes[0].set_title('Heavy users dominate pop-level ratio\n(more sessions → more revenue)',loc='left',fontweight='bold')
axes[0].legend(frameon=False)
ests=['User-level\nmean(rev/sess)','Pop-level\ntotal_rev/total_sess']
x=np.arange(2)
axes[1].bar(x-0.2,[ul_ctrl,pl_ctrl],0.35,color='#DD8452',alpha=0.85,label='Control')
axes[1].bar(x+0.2,[ul_trt, pl_trt], 0.35,color='#4C72B0',alpha=0.85,label='Treatment')
axes[1].set_xticks(x); axes[1].set_xticklabels(ests)
axes[1].set_ylabel('ARPU ($/session)'); axes[1].legend(frameon=False)
axes[1].set_title('Two estimators give different answers\n— choose one before launch',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("ratio_metrics.png")

# ── aggregation bias ─────────────────────────────────────────────────────────
rng=np.random.default_rng(44); n_u=500
heavy=(rng.binomial(1,0.20,n_u)).astype(bool)
sess_ctrl=np.where(heavy,rng.integers(20,50,n_u),rng.integers(1,5,n_u))
treat=rng.binomial(1,0.5,n_u)
effect=np.where(heavy,-0.3,0.5)*treat
sv=3.0+effect+rng.normal(0,0.5,n_u)
ev_ctrl=(sv[treat==0]*sess_ctrl[treat==0]).sum()/sess_ctrl[treat==0].sum()
ev_trt =(sv[treat==1]*sess_ctrl[treat==1]).sum()/sess_ctrl[treat==1].sum()
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,(groups,title) in zip(axes,[
    ([(sv[treat==0],'Control'),(sv[treat==1],'Treatment')],'User-level (each user equal weight)'),
    ([(np.repeat(sv[treat==0],sess_ctrl[treat==0]),'Control'),
      (np.repeat(sv[treat==1],sess_ctrl[treat==1]),'Treatment')],'Event-level (heavy users dominate)'),
]):
    for d,lbl in groups:
        c='#DD8452' if 'Ctrl' in lbl or lbl=='Control' else '#4C72B0'
        ax.hist(d,bins=30,alpha=0.6,color=c,density=True,label=f'{lbl}  μ={d.mean():.2f}')
    ax.set_title(title,loc='left',fontweight='bold'); ax.legend(frameon=False)
    ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("aggregation_bias.png")

# ── regression to mean ────────────────────────────────────────────────────────
rng=np.random.default_rng(21); n=5000
q=rng.normal(5,2,n); p1=q+rng.normal(0,2,n); p2=q+rng.normal(0,2,n)
thresh=np.percentile(p1,80); top=p1>thresh
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].scatter(p1[~top],p2[~top],alpha=0.1,s=5,color='#ddd',label='Other users')
axes[0].scatter(p1[top], p2[top], alpha=0.4,s=8,color='#4C72B0',label=f'Top users (P1>{thresh:.1f})')
xl=np.linspace(thresh,p1.max(),50)
axes[0].plot(xl,xl,color='#333',lw=1,linestyle='--',label='P2=P1 (no RTM)')
axes[0].set_xlabel('Period 1 score'); axes[0].set_ylabel('Period 2 score')
axes[0].set_title('Top users drift toward mean in P2\nwith no intervention',loc='left',fontweight='bold')
axes[0].legend(frameon=False,fontsize=8)
axes[1].hist(p2,bins=30,color='#ccc',alpha=0.6,density=True,label=f'All users  μ={p2.mean():.2f}')
axes[1].hist(p2[top],bins=20,color='#4C72B0',alpha=0.7,density=True,label=f'Top users  μ={p2[top].mean():.2f}')
axes[1].axvline(thresh,color='#e63946',lw=1.5,linestyle='--',label=f'P1 threshold={thresh:.1f}')
axes[1].set_xlabel('Period 2 score')
axes[1].set_title('Top users regress toward population mean',loc='left',fontweight='bold')
axes[1].legend(frameon=False,fontsize=8)
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("regression_to_mean.png")

print("=== 04-variance-reduction ===")

# ── CUPED ─────────────────────────────────────────────────────────────────────
rng=np.random.default_rng(42); n=1000
quality=rng.normal(0,1,n); x_pre=quality+rng.normal(0,0.8,n)
treat=rng.binomial(1,0.5,n); y=quality+0.5*treat+rng.normal(0,0.8,n)
theta=np.cov(y,x_pre)[0,1]/np.var(x_pre); y_adj=y-theta*(x_pre-x_pre.mean())
rho,_=pearsonr(y,x_pre)
fig,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].scatter(x_pre,y,alpha=0.2,s=8,color='#4C72B0')
axes[0].set_xlabel('Pre-experiment metric (X)'); axes[0].set_ylabel('Post metric (Y)')
axes[0].set_title(f'Pre–post correlation ρ={rho:.2f}\nVariance reduction ≈{(1-rho**2)*100:.0f}%',loc='left',fontweight='bold')
for ax,(yy,lbl,c) in zip(axes[1:],[(y,'Raw outcome Y','#DD8452'),(y_adj,'CUPED-adjusted Y','#2a9d8f')]):
    _,p_=ttest_ind(yy[treat==1],yy[treat==0])
    ax.hist(yy[treat==0],bins=30,alpha=0.6,color='#DD8452',density=True,label='Control')
    ax.hist(yy[treat==1],bins=30,alpha=0.6,color='#4C72B0',density=True,label='Treatment')
    ax.set_title(f'{lbl}\np={p_:.4f}',loc='left',fontweight='bold',color=c)
    ax.legend(frameon=False); ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
axes[0].spines['left'].set_color('#ddd'); axes[0].spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("cuped.png")

# ── stratification ───────────────────────────────────────────────────────────
import statsmodels.formula.api as smf
rng=np.random.default_rng(33)
rows=[]
for s,se in [('SMB',0.5),('Mid-market',1.0),('Enterprise',2.0)]:
    for _ in range(100):
        t=rng.binomial(1,0.5); y=se*10+2.0*t+rng.normal(0,se*5)
        rows.append({'stratum':s,'treatment':t,'y':y})
df=pd.DataFrame(rows)
_,p_n=ttest_ind(df[df.treatment==1].y,df[df.treatment==0].y)
m=smf.ols('y ~ treatment + C(stratum)',data=df).fit()
p_a=m.pvalues['treatment']; b_a=m.params['treatment']
sm=df.groupby(['stratum','treatment'])['y'].mean().unstack()
fig,axes=plt.subplots(1,2,figsize=(11,4))
x=np.arange(len(sm)); w=0.35
axes[0].bar(x-w/2,sm[0],w,color='#DD8452',alpha=0.85,label='Control')
axes[0].bar(x+w/2,sm[1],w,color='#4C72B0',alpha=0.85,label='Treatment')
axes[0].set_xticks(x); axes[0].set_xticklabels(sm.index)
axes[0].set_ylabel('Mean outcome'); axes[0].legend(frameon=False)
axes[0].set_title('Outcome by stratum',loc='left',fontweight='bold')
axes[1].bar([f'Naive\np={p_n:.4f}',f'ANCOVA\np={p_a:.4f}'],
            [abs(ttest_ind(df[df.treatment==1].y,df[df.treatment==0].y)[0]),abs(m.tvalues['treatment'])],
            color=['#DD8452','#2a9d8f'],width=0.4,alpha=0.85)
axes[1].set_ylabel('|t-statistic|'); axes[1].set_title('ANCOVA is more sensitive',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("stratification.png")

# ── winsorization ─────────────────────────────────────────────────────────────
rng=np.random.default_rng(77); n=2000
base=rng.lognormal(4,1.5,n); w=rng.binomial(1,0.01,n)*rng.exponential(100000,n)
treat=rng.binomial(1,0.5,n); y=base+w+10.0*treat
p99=np.percentile(y,99); yw=np.clip(y,0,p99)
fig,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].hist(np.log1p(y[treat==0]),bins=40,alpha=0.6,color='#DD8452',density=True,label='Control')
axes[0].hist(np.log1p(y[treat==1]),bins=40,alpha=0.6,color='#4C72B0',density=True,label='Treatment')
axes[0].set_title('Raw distribution',loc='left',fontweight='bold'); axes[0].set_xlabel('log(1+revenue)'); axes[0].legend(frameon=False)
axes[1].hist(np.log1p(yw[treat==0]),bins=40,alpha=0.6,color='#DD8452',density=True,label='Control')
axes[1].hist(np.log1p(yw[treat==1]),bins=40,alpha=0.6,color='#4C72B0',density=True,label='Treatment')
axes[1].set_title(f'Winsorized at p99=${p99:,.0f}',loc='left',fontweight='bold'); axes[1].legend(frameon=False)
r=y[treat==1].mean()-y[treat==0].mean(); rw=yw[treat==1].mean()-yw[treat==0].mean()
axes[2].bar(['Raw mean\ndiff','Winsorized\nmean diff'],[r,rw],color=['#DD8452','#2a9d8f'],width=0.4,alpha=0.85)
axes[2].axhline(10,color='#333',lw=1.5,linestyle='--',label='True effect = 10')
axes[2].legend(frameon=False); axes[2].set_title('Both valid — winsorized less noisy',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("winsorization.png")

# ── delta method ─────────────────────────────────────────────────────────────
rng=np.random.default_rng(88); n=2000
treat=rng.binomial(1,0.5,n)
sess=rng.negative_binomial(3,0.3,n)+1; rev=5*sess+2*treat+rng.normal(0,10,n)
R=rev.mean()/sess.mean(); z=rev-R*sess
_,p_d=ttest_ind(z[treat==1],z[treat==0])
ur=rev/sess; _,p_u=ttest_ind(ur[treat==1],ur[treat==0])
se_wrong=(rev.std()/np.sqrt(n/2))/sess.mean()
se_delta=z.std()/np.sqrt(n/2)/sess.mean()
se_user =ur.std()/np.sqrt(n/2)
fig,ax=plt.subplots(figsize=(9,4))
methods=['Naive SE\n(wrong)','User-level\nmean(a/b)','Delta method\n(correct)']
ses=[se_wrong,se_user,se_delta]; colors=['#e63946','#4C72B0','#2a9d8f']
bars=ax.bar(methods,ses,color=colors,width=0.4,alpha=0.85)
for b,v in zip(bars,ses): ax.text(b.get_x()+b.get_width()/2,v+0.0002,f'{v:.4f}',ha='center',fontsize=9)
ax.set_ylabel('Standard error of ARPU estimate')
ax.set_title('SE comparison for ratio metric (ARPU)\nDelta method gives the correct SE',loc='left',fontweight='bold')
ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("delta_method.png")

print("=== 05-causal-methods ===")

# ── PSM ───────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
rng=np.random.default_rng(42); n=2000
age=rng.exponential(12,n); usage=rng.exponential(5,n)
lo=-2+0.05*age+0.15*usage; pt=1/(1+np.exp(-lo)); treat=rng.binomial(1,pt)
rev=10*age+8*usage+5.0*treat+rng.normal(0,20,n)
df2=pd.DataFrame({'treat':treat,'age':age,'usage':usage,'rev':rev})
lr=LogisticRegression().fit(df2[['age','usage']].values,treat)
df2['ps']=lr.predict_proba(df2[['age','usage']].values)[:,1]
ti=df2.index[df2.treat==1].tolist(); ci=df2.index[df2.treat==0].tolist()
pt_=df2.loc[ti,'ps'].values.reshape(-1,1); pc_=df2.loc[ci,'ps'].values.reshape(-1,1)
nbrs=NearestNeighbors(n_neighbors=1).fit(pc_)
_,idxs=nbrs.kneighbors(pt_)
mc=[ci[i[0]] for i in idxs]
mdf=pd.concat([df2.loc[ti].assign(r='treated'),df2.loc[mc].assign(r='ctrl_m')])
att=mdf[mdf.r=='treated']['rev'].mean()-mdf[mdf.r=='ctrl_m']['rev'].mean()
naive=df2[df2.treat==1]['rev'].mean()-df2[df2.treat==0]['rev'].mean()
def smd(a,b): return (a.mean()-b.mean())/np.sqrt((a.std()**2+b.std()**2)/2)
fig,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].hist(df2[df2.treat==0]['ps'],bins=30,alpha=0.6,color='#DD8452',density=True,label='Control')
axes[0].hist(df2[df2.treat==1]['ps'],bins=30,alpha=0.6,color='#4C72B0',density=True,label='Treated')
axes[0].set_xlabel('Propensity score'); axes[0].set_title('Propensity overlap\n(common support check)',loc='left',fontweight='bold'); axes[0].legend(frameon=False)
covs=['age','usage']
sb=[smd(df2[df2.treat==1][c],df2[df2.treat==0][c]) for c in covs]
sa=[smd(mdf[mdf.r=='treated'][c],mdf[mdf.r=='ctrl_m'][c]) for c in covs]
y2=np.arange(2)
axes[1].barh(y2+0.2,sb,0.35,color='#e63946',alpha=0.8,label='Before matching')
axes[1].barh(y2-0.2,sa,0.35,color='#2a9d8f',alpha=0.8,label='After matching')
axes[1].axvline(0.1,color='#ccc',lw=1,linestyle='--')
axes[1].set_yticks(y2); axes[1].set_yticklabels(covs)
axes[1].set_xlabel('Standardized mean difference'); axes[1].set_title('Balance: SMD < 0.1 after matching ✓',loc='left',fontweight='bold'); axes[1].legend(frameon=False)
axes[2].bar(['Naive\n(confounded)','PSM ATT'],[naive,att],color=['#e63946','#2a9d8f'],width=0.4,alpha=0.85)
axes[2].axhline(5.0,color='#333',lw=1.5,linestyle='--',label='True effect = 5')
axes[2].legend(frameon=False); axes[2].set_ylabel('Estimated ATT')
axes[2].set_title('PSM corrects for confounding',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("psm.png")

# ── DiD ───────────────────────────────────────────────────────────────────────
rng=np.random.default_rng(11); weeks=np.arange(-8,9)
rows2=[]
for a in range(100):
    treated=a<50; ai=rng.normal(100,20)
    for w in weeks:
        post=w>=0; eff=15.0*treated*post
        y=ai+1.5*w+eff+rng.normal(0,5)
        rows2.append({'account':a,'week':w,'treated':int(treated),'post':int(post),'y':y})
df3=pd.DataFrame(rows2); wk=df3.groupby(['week','treated'])['y'].mean().unstack()
m_did=smf.ols('y ~ treated + post + treated:post',data=df3).fit(); did_e=m_did.params['treated:post']
ev_eff=[]; ev_se=[]
pre_m=None
for w in weeks:
    s=df3[df3.week==w]
    te=s[s.treated==1]['y'].mean()-s[s.treated==0]['y'].mean()
    ts=np.sqrt(s[s.treated==1]['y'].std()**2/(s.treated==1).sum()+s[s.treated==0]['y'].std()**2/(s.treated==0).sum())
    ev_eff.append(te); ev_se.append(ts)
pm=np.mean([ev_eff[i] for i,w in enumerate(weeks) if w<0])
adj=[e-pm for e in ev_eff]
fig,axes=plt.subplots(1,2,figsize=(11,4))
for g,c,l in [(1,'#4C72B0','Treated'),(0,'#DD8452','Control')]:
    axes[0].plot(wk.index,wk[g],color=c,lw=2,label=l,marker='o',markersize=3)
axes[0].axvline(0,color='#e63946',lw=2,linestyle='--')
axes[0].text(0.3,wk.values.max()*0.97,'Rollout',color='#e63946',fontsize=9)
axes[0].set_xlabel('Week'); axes[0].set_ylabel('Mean revenue')
axes[0].set_title('Parallel pre-trends ✓  |  Effect visible post-rollout',loc='left',fontweight='bold'); axes[0].legend(frameon=False)
axes[1].fill_between(weeks,[a-1.96*s for a,s in zip(adj,ev_se)],[a+1.96*s for a,s in zip(adj,ev_se)],alpha=0.2,color='#4C72B0')
axes[1].plot(weeks,adj,color='#4C72B0',lw=2,marker='o',markersize=3)
axes[1].axhline(0,color='#333',lw=0.8); axes[1].axvline(0,color='#e63946',lw=1.5,linestyle='--')
axes[1].set_xlabel('Week'); axes[1].set_ylabel('Treatment − control (pre-adjusted)')
axes[1].set_title(f'Event study  |  DiD = {did_e:.1f}  (true = 15)',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("did.png")

# ── IV ────────────────────────────────────────────────────────────────────────
rng=np.random.default_rng(55); n=3000
Z=rng.binomial(1,0.5,n)
never=rng.binomial(1,0.20,n); always=rng.binomial(1,0.10,n)*(1-never)
comp=1-never-always; T=always+comp*Z
rev2=50+20.0*T+rng.normal(0,30,n)
itt_y=rev2[Z==1].mean()-rev2[Z==0].mean()
itt_t=T[Z==1].mean()-T[Z==0].mean(); late=itt_y/itt_t
naive2=rev2[T==1].mean()-rev2[T==0].mean()
X_fs=np.column_stack([np.ones(n),Z]); b_fs=np.linalg.lstsq(X_fs,T,rcond=None)[0]
T_hat=X_fs@b_fs; ss_r=np.sum((T-T_hat)**2); ss_m=np.sum((T_hat-T.mean())**2)
F=(ss_m/1)/(ss_r/(n-2))
fig,axes=plt.subplots(1,2,figsize=(11,4))
acts=[T[Z==0].mean(),T[Z==1].mean()]
axes[0].bar(['No prompt','Prompt'],acts,color=['#DD8452','#4C72B0'],width=0.4,alpha=0.85)
axes[0].set_ylabel('Activation rate'); axes[0].set_title(f'First stage: prompt drives activation\nITT_T={itt_t:.3f}  F={F:.0f}',loc='left',fontweight='bold')
for i,v in enumerate(acts): axes[0].text(i,v+0.003,f'{v:.3f}',ha='center',fontsize=10)
axes[1].bar(['Naive\n(biased)','ITT\n(prompt→rev)','IV / LATE\n(activation→rev)'],
            [naive2,itt_y,late],color=['#e63946','#4C72B0','#2a9d8f'],width=0.4,alpha=0.85)
axes[1].axhline(20,color='#333',lw=1.5,linestyle='--',label='True LATE = 20'); axes[1].legend(frameon=False)
axes[1].set_ylabel('Estimated effect'); axes[1].set_title('IV recovers LATE',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("iv.png")

# ── AIPW ─────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
rng=np.random.default_rng(99); n=2000
X4=rng.normal(0,1,(n,3)); lo4=-0.5+0.7*X4[:,0]+0.4*X4[:,1]
T4=rng.binomial(1,1/(1+np.exp(-lo4))); Y4=5+2*X4[:,0]+3.0*T4+rng.normal(0,2,n)
kf=KFold(n_splits=2,shuffle=True,random_state=0); scores=np.zeros(n)
for tr,ev in kf.split(X4):
    ps=LogisticRegression().fit(X4[tr],T4[tr])
    e_hat=np.clip(ps.predict_proba(X4[ev])[:,1],0.05,0.95)
    om=Ridge().fit(np.column_stack([X4[tr],T4[tr]]),Y4[tr])
    mu1=om.predict(np.column_stack([X4[ev],np.ones(len(X4[ev]))]));mu0=om.predict(np.column_stack([X4[ev],np.zeros(len(X4[ev]))]))
    scores[ev]=mu1-mu0+T4[ev]/e_hat*(Y4[ev]-mu1)-(1-T4[ev])/(1-e_hat)*(Y4[ev]-mu0)
ate=scores.mean(); se_a=scores.std()/np.sqrt(n)
naive3=Y4[T4==1].mean()-Y4[T4==0].mean()
ps2=LogisticRegression().fit(X4,T4); e2=np.clip(ps2.predict_proba(X4)[:,1],0.05,0.95)
ipw=(Y4*T4/e2-Y4*(1-T4)/(1-e2)).mean()
fig,ax=plt.subplots(figsize=(9,4))
errs=[Y4[T4==1].std()/np.sqrt(T4.sum())+Y4[T4==0].std()/np.sqrt((1-T4).sum()),0,1.96*se_a]
ax.bar(['Naive','IPW only','AIPW\n(doubly robust)'],[naive3,ipw,ate],color=['#e63946','#4C72B0','#2a9d8f'],width=0.4,alpha=0.85,yerr=errs,capsize=5)
ax.axhline(3.0,color='#333',lw=1.5,linestyle='--',label='True ATE = 3.0'); ax.legend(frameon=False)
ax.set_ylabel('Estimated ATE'); ax.set_title('AIPW closest to truth\nDoubled protection against model misspecification',loc='left',fontweight='bold')
ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("aipw.png")

# ── target trial ─────────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(12,5)); ax.axis('off')
comps=[
    ('1. Eligibility','Accounts ≥30 days old,\nnot yet on Enterprise',0.85),
    ('2. Treatments','A: Upgrade prompt\nB: No prompt (control)',0.65),
    ('3. Assignment','Emulated via PSM on:\nplan, age, usage',0.45),
    ('4. Time zero','Date account became eligible\n(NOT first feature use)',0.25),
]
outs=[('5. Follow-up','T0 to T0+90 days',0.65),('6. Outcome','Enterprise upgrade\nwithin 90 days',0.45),
      ('7. Assumptions','Conditional exchangeability\non measured covariates',0.25)]
for lbl,txt,y in comps:
    ax.add_patch(plt.Rectangle((0.02,y-0.09),0.44,0.16,fill=True,facecolor='#e8f4f8',edgecolor='#457b9d',lw=1.2))
    ax.text(0.06,y,lbl,fontsize=9,fontweight='bold',va='center',color='#457b9d')
    ax.text(0.25,y,txt,fontsize=9,va='center',color='#333')
for lbl,txt,y in outs:
    ax.add_patch(plt.Rectangle((0.54,y-0.09),0.44,0.16,fill=True,facecolor='#fef9e7',edgecolor='#e9c46a',lw=1.2))
    ax.text(0.58,y,lbl,fontsize=9,fontweight='bold',va='center',color='#c49a00')
    ax.text(0.77,y,txt,fontsize=9,va='center',color='#333')
ax.text(0.5,0.97,'Target Trial Emulation Spec',ha='center',va='top',fontsize=12,fontweight='bold')
ax.text(0.5,0.92,'"What RCT would we have run?"',ha='center',va='top',fontsize=10,color='#666',style='italic')
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout(); save("target_trial.png")

print("=== 06-readout ===")

# ── HTE forest ───────────────────────────────────────────────────────────────
rng=np.random.default_rng(7); n=3000
segs=rng.choice(['SMB','Mid-market','Enterprise'],n,p=[0.5,0.3,0.2])
treat=rng.binomial(1,0.5,n); eff_map={'SMB':3.0,'Mid-market':1.0,'Enterprise':-2.0}
y5=np.array([20+eff_map[s]*t+rng.normal(0,8) for s,t in zip(segs,treat)])
df4=pd.DataFrame({'segment':segs,'treatment':treat,'y':y5})
res=[]
for seg in ['SMB','Mid-market','Enterprise','Overall']:
    s=df4 if seg=='Overall' else df4[df4.segment==seg]
    t_=ttest_ind(s[s.treatment==1].y,s[s.treatment==0].y)
    d=s[s.treatment==1].y.mean()-s[s.treatment==0].y.mean()
    se_=np.sqrt(s[s.treatment==1].y.var()/(s.treatment==1).sum()+s[s.treatment==0].y.var()/(s.treatment==0).sum())
    res.append({'segment':seg,'effect':d,'se':se_,'n':len(s),'p':t_.pvalue})
res=pd.DataFrame(res)
fig,ax=plt.subplots(figsize=(9,5))
y_p=np.arange(len(res)); cols=['#4C72B0' if p<0.05 else '#aaa' for p in res.p]
ax.barh(y_p,res.effect,xerr=1.96*res.se,color=cols,height=0.5,alpha=0.85,error_kw={'ecolor':'#666','capsize':4})
ax.axvline(0,color='#333',lw=1); ax.axhline(len(res)-1.5,color='#ddd',lw=1,linestyle='--')
for i,(_,row) in enumerate(res.iterrows()):
    ax.text(row.effect+1.96*row.se+0.1,i,f"n={row.n:,}  β={row.effect:.1f}  p={row.p:.3f}",va='center',fontsize=8,color='#666')
ax.set_yticks(y_p); ax.set_yticklabels(res.segment)
ax.set_xlabel('Treatment effect (95% CI)')
ax.set_title('Forest plot: HTE across segments\nPositive for SMB, negative for Enterprise',loc='left',fontweight='bold')
ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("hte_forest.png")

# ── multiple testing ─────────────────────────────────────────────────────────
from statsmodels.stats.multitest import multipletests
rng=np.random.default_rng(42); m=20
pv=np.concatenate([rng.uniform(0.0001,0.04,5),rng.uniform(0,1,15)]); rng.shuffle(pv)
rj_b,_,_,_=multipletests(pv,0.05,'bonferroni'); rj_bh,_,_,_=multipletests(pv,0.05,'fdr_bh'); rj_n=pv<0.05
si=np.argsort(pv); mnames=[f'M{i+1:02d}' for i in range(m)]
bh_th=0.05*(np.arange(1,m+1))/m
fig,axes=plt.subplots(1,2,figsize=(13,5))
cols=np.where(rj_n,'#e63946','#aaa')
axes[0].scatter(range(m),pv[si],color=cols[si],s=60,zorder=3)
axes[0].axhline(0.05,color='#e63946',lw=1.5,linestyle='--',label='α=0.05')
axes[0].set_xticks(range(m)); axes[0].set_xticklabels(np.array(mnames)[si],rotation=45,ha='right',fontsize=7)
axes[0].set_title(f'Uncorrected: {rj_n.sum()} significant (includes FP)',loc='left',fontweight='bold'); axes[0].legend(frameon=False)
axes[1].scatter(range(m),pv[si],color='#4C72B0',s=60,zorder=3,label='p-value')
axes[1].plot(range(m),bh_th,color='#2a9d8f',lw=2,label='BH threshold')
axes[1].axhline(0.05/m,color='#f4a261',lw=1.5,linestyle='--',label='Bonferroni')
axes[1].set_xticks(range(m)); axes[1].set_xticklabels(np.array(mnames)[si],rotation=45,ha='right',fontsize=7)
axes[1].set_title(f'BH: {rj_bh.sum()} significant  |  Bonferroni: {rj_b.sum()} significant',loc='left',fontweight='bold'); axes[1].legend(frameon=False)
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("multiple_testing.png")

# ── ITT vs PP ────────────────────────────────────────────────────────────────
rng=np.random.default_rng(13); n=4000
hi2=rng.binomial(1,0.35,n); asgn=rng.binomial(1,0.5,n); act=hi2*asgn
rev3=50+rng.normal(0,1,n)*20+20.0*act+rng.normal(0,20,n)
itt_y2=rev3[asgn==1].mean()-rev3[asgn==0].mean(); comp2=act[asgn==1].mean(); pp_iv2=itt_y2/comp2
pp_n2=rev3[act==1].mean()-rev3[act==0].mean()
fig,axes=plt.subplots(1,2,figsize=(11,4))
ax=axes[0]; ax.axis('off')
for y_,txt,col in [(0.82,f'Assigned to treatment\nn={asgn.sum():,}','#4C72B0'),
                   (0.52,f'Activated  n={act.sum():,}  ({comp2*100:.0f}% compliance)','#2a9d8f'),
                   (0.22,f'Did NOT activate  n={(asgn-act).sum():,}  (still in ITT)','#DD8452')]:
    ax.add_patch(plt.Rectangle((0.08,y_-0.09),0.84,0.16,fill=True,facecolor=col,alpha=0.12,edgecolor=col,lw=1.5))
    ax.text(0.5,y_,txt,ha='center',va='center',fontsize=9,color=col)
ax.set_title('Enrollment flow: ITT includes all assigned',loc='left',fontweight='bold')
axes[1].bar(['Naive PP\n(biased)','IV / Wald PP','ITT'],[pp_n2,pp_iv2,itt_y2],
            color=['#e63946','#2a9d8f','#4C72B0'],width=0.4,alpha=0.85)
axes[1].axhline(20,color='#333',lw=1.5,linestyle='--',label='True LATE = 20'); axes[1].legend(frameon=False)
axes[1].set_ylabel('Estimated effect'); axes[1].set_title('ITT < IV/PP < Naive PP',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("itt_vs_pp.png")

# ── mediation ────────────────────────────────────────────────────────────────
rng=np.random.default_rng(34); n=2000
T5=rng.binomial(1,0.5,n); M5=0.5+0.4*T5+rng.normal(0,0.3,n); Y5=20+10.0*M5+5.0*T5+rng.normal(0,8,n)
df5=pd.DataFrame({'T':T5,'M':M5,'Y':Y5})
m1_=smf.ols('M~T',data=df5).fit(); m2_=smf.ols('Y~T+M',data=df5).fit()
ind=m1_.params['T']*m2_.params['M']; drct=m2_.params['T']; tot=ind+drct
rng2=np.random.default_rng(0); bi=[]
for _ in range(200):
    idx=rng2.integers(n,size=n); df_b=df5.iloc[idx]
    bi.append(smf.ols('M~T',data=df_b).fit().params['T']*smf.ols('Y~T+M',data=df_b).fit().params['M'])
cil,cih=np.percentile(bi,[2.5,97.5])
fig,axes=plt.subplots(1,2,figsize=(11,4))
ax=axes[0]; ax.axis('off')
pos={'T':(0.10,0.50),'M':(0.50,0.85),'Y':(0.90,0.50)}
for (s,d),lbl,c in [(('T','M'),f'β={m1_.params["T"]:.2f}','#2a9d8f'),(('M','Y'),f'β={m2_.params["M"]:.2f}','#2a9d8f'),(('T','Y'),f'β={drct:.2f}','#457b9d')]:
    x0,y0=pos[s];x1,y1=pos[d]
    ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='->',color=c,lw=2.5))
    ax.text((x0+x1)/2,(y0+y1)/2+0.06,lbl,ha='center',fontsize=9,color=c)
for nm,(x,y) in pos.items():
    lbls={'T':'New\nonboarding','M':'Onboarding\ncompletion','Y':'Revenue'}
    ax.text(x,y,lbls[nm],ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=0.4',facecolor='#f5f5f5',edgecolor='#bbb'))
ax.set_title('Mediation path diagram',loc='left',fontweight='bold')
axes[1].bar(['Direct\n(T→Y)','Indirect\n(T→M→Y)','Total'],[drct,ind,tot],color=['#457b9d','#2a9d8f','#4C72B0'],width=0.4,alpha=0.85)
axes[1].errorbar([1],[ind],yerr=[[ind-cil],[cih-ind]],fmt='none',color='#333',capsize=5)
axes[1].set_ylabel('Effect on revenue ($)'); axes[1].set_title(f'{ind/tot*100:.0f}% mediated through onboarding',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("mediation.png")

# ── SPRT ─────────────────────────────────────────────────────────────────────
rng=np.random.default_rng(99); p0,p1_=0.05,0.08; alpha2,beta2=0.05,0.20
A=np.log((1-beta2)/alpha2); B_=np.log(beta2/(1-alpha2))
fig,ax=plt.subplots(figsize=(11,5))
for sim,(true_p,col) in enumerate([(p1_,'#4C72B0'),(p1_,'#2a9d8f'),(p0,'#DD8452')]):
    obs=rng.binomial(1,true_p,3000); llr=0; lrs=[0]; stop=None
    for i,x in enumerate(obs):
        llr+=x*np.log(p1_/p0)+(1-x)*np.log((1-p1_)/(1-p0)); lrs.append(llr)
        if llr>=A: stop=(i+1,'reject H₀'); break
        if llr<=B_: stop=(i+1,'accept H₀'); break
    ax.plot(lrs,color=col,alpha=0.8,lw=1.5,label=f'Sim {sim+1}: p={true_p}')
    if stop: ax.scatter([stop[0]],[lrs[stop[0]]],color=col,s=100,zorder=5); ax.text(stop[0]+20,lrs[stop[0]],stop[1],fontsize=8,color=col)
ax.axhline(A,color='#2a9d8f',lw=2,linestyle='--',label=f'Reject boundary A={A:.2f}')
ax.axhline(B_,color='#e63946',lw=2,linestyle='--',label=f'Accept boundary B={B_:.2f}')
ax.axhline(0,color='#ccc',lw=0.8); ax.set_xlabel('Observations'); ax.set_ylabel('Log-likelihood ratio')
ax.set_title(f'SPRT: stop when LLR crosses a boundary\np₀={p0}, p₁={p1_}, α={alpha2}, β={beta2}',loc='left',fontweight='bold')
ax.legend(frameon=False,fontsize=8); ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("sprt.png")

# ── Bayesian readout ─────────────────────────────────────────────────────────
rng=np.random.default_rng(7)
nc,sc=1500,72; nt,st=1500,91
pc_=beta_dist(1+sc,1+nc-sc); pt_=beta_dist(1+st,1+nt-st)
cs_=pc_.rvs(100000,random_state=rng); ts_=pt_.rvs(100000,random_state=rng)
pb=(ts_>cs_).mean(); lift=(ts_-cs_)/cs_*100
fig,axes=plt.subplots(1,2,figsize=(11,4))
x6=np.linspace(0.02,0.12,300)
axes[0].plot(x6,pc_.pdf(x6),color='#DD8452',lw=2.5,label=f'Control  {sc/nc:.1%}')
axes[0].plot(x6,pt_.pdf(x6),color='#4C72B0',lw=2.5,label=f'Treatment  {st/nt:.1%}')
axes[0].fill_between(x6,pc_.pdf(x6),alpha=0.15,color='#DD8452'); axes[0].fill_between(x6,pt_.pdf(x6),alpha=0.15,color='#4C72B0')
axes[0].set_xlabel('Conversion rate'); axes[0].set_title(f'Posterior distributions\nP(treatment>control) = {pb:.1%}',loc='left',fontweight='bold'); axes[0].legend(frameon=False)
axes[1].hist(lift,bins=80,color='#4C72B0',alpha=0.7,density=True)
axes[1].axvline(0,color='#e63946',lw=2,linestyle='--')
axes[1].axvline(np.percentile(lift,2.5),color='#aaa',lw=1.5,linestyle=':')
axes[1].axvline(np.percentile(lift,97.5),color='#aaa',lw=1.5,linestyle=':')
axes[1].text(np.percentile(lift,2.5)-0.3,axes[1].get_ylim()[1]*0.7,f'{np.percentile(lift,2.5):.1f}%',ha='right',fontsize=8,color='#888')
axes[1].text(np.percentile(lift,97.5)+0.3,axes[1].get_ylim()[1]*0.7,f'{np.percentile(lift,97.5):.1f}%',fontsize=8,color='#888')
axes[1].set_xlabel('Relative lift (%)'); axes[1].set_title('Posterior lift distribution\n95% credible interval',loc='left',fontweight='bold')
for ax in axes: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("bayesian_readout.png")

print("=== 07-communication ===")

# ── CI business value ────────────────────────────────────────────────────────
def exp_res(n,eff,sig,seed):
    r=np.random.default_rng(seed); c=r.normal(100,sig,n); t=r.normal(100+eff,sig,n)
    _,p=ttest_ind(t,c); d=t.mean()-c.mean(); se=np.sqrt(t.var()/n+c.var()/n); return d,se,p
d1,s1,p1_=exp_res(200000,0.10,30,1); d2,s2,p2_=exp_res(2000,8.0,30,2)
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,(d,s,p,n,label) in zip(axes,[
    (d1,s1,p1_,200000,f'Exp A: n=200K  effect={d1:.2f}'),
    (d2,s2,p2_,2000,  f'Exp B: n=2K    effect={d2:.2f}'),
]):
    ci_lo,ci_hi=d-1.96*s,d+1.96*s
    ax.barh(['Effect'],[d],color='#4C72B0',height=0.3,alpha=0.85,xerr=[[d-ci_lo],[ci_hi-d]],capsize=8,error_kw={'ecolor':'#666','capsize':6})
    ax.axvline(0,color='#aaa',lw=1); ax.axvline(2.0,color='#e63946',lw=1.5,linestyle='--',label='MDE=2')
    arr_lo=ci_lo*n*12*500/100
    ax.set_title(f'{label}\np={p:.4f}  CI:[{ci_lo:.2f},{ci_hi:.2f}]\nARR pessimistic: ${arr_lo:,.0f}',loc='left',fontweight='bold')
    ax.legend(frameon=False); ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.suptitle('Same p-value, very different business implications',fontweight='bold')
plt.tight_layout(); save("ci_business.png")

# ── causal language ───────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(11,5)); ax.axis('off')
rows_=[('RCT + all checks pass','Caused\n(full causal claim)','#2a9d8f','✓ Randomized\n✓ SRM passed\n✓ 14+ days'),
       ('RCT + minor issues','Likely caused\n(qualified)','#f4a261','✓ Randomized\n⚠ Short runtime'),
       ('Quasi-experiment (DiD/PSM)','Associated with\n(after adjustment)','#457b9d','~ Observational\n✓ Parallel trends'),
       ('Observational (no controls)','Correlated with\n(descriptive only)','#e63946','✗ Self-selected\n✗ Confounders')]
for i,(sc,cl,col,ch) in enumerate(rows_):
    y=0.78-i*0.20
    ax.add_patch(mpatches.FancyBboxPatch((0.02,y-0.07),0.40,0.13,boxstyle='round,pad=0.01',facecolor=col,alpha=0.10,edgecolor=col,lw=1.5))
    ax.add_patch(mpatches.FancyBboxPatch((0.54,y-0.07),0.44,0.13,boxstyle='round,pad=0.01',facecolor=col,alpha=0.07,edgecolor=col,lw=1))
    ax.text(0.22,y,sc,ha='center',va='center',fontsize=9,color='#333')
    ax.text(0.76,y,cl,ha='center',va='center',fontsize=10,fontweight='bold',color=col)
    ax.text(0.48,y,ch,ha='center',va='center',fontsize=7.5,color='#777')
ax.text(0.22,0.97,'Study design',ha='center',fontsize=10,fontweight='bold',color='#555')
ax.text(0.76,0.97,'Language to use',ha='center',fontsize=10,fontweight='bold',color='#555')
ax.set_title('Causal language ladder',loc='left',fontsize=12,fontweight='bold')
ax.set_xlim(0,1);ax.set_ylim(0,1)
plt.tight_layout(); save("causal_language.png")

# ── experiment narrative ─────────────────────────────────────────────────────
rng=np.random.default_rng(7); nc2,sc2=3500,168; nt2,st2=3500,213
pc2=sc2/nc2; pt2=st2/nt2; d6=pt2-pc2; se6=np.sqrt(pc2*(1-pc2)/nc2+pt2*(1-pt2)/nt2)
ci6l,ci6h=d6-1.96*se6,d6+1.96*se6; z6=d6/se6; pv6=2*(1-norm.cdf(abs(z6)))
p_ctrl6=beta_dist(1+sc2,1+nc2-sc2); p_trt6=beta_dist(1+st2,1+nt2-st2)
s6c=p_ctrl6.rvs(50000,random_state=rng); s6t=p_trt6.rvs(50000,random_state=rng)
pb6=(s6t>s6c).mean(); lft6=(s6t-s6c)/s6c*100
arr6=ci6l*50000*12*200
fig,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].barh(['Conversion\nrate lift'],[d6*100],color='#4C72B0',height=0.3,alpha=0.85,xerr=[[(d6-ci6l)*100]],capsize=8,error_kw={'ecolor':'#666'})
axes[0].axvline(0,color='#aaa',lw=1); axes[0].set_xlabel('pp')
axes[0].set_title(f'Primary metric\np={pv6:.4f}  CI:[{ci6l*100:.2f},{ci6h*100:.2f}pp]\nARR pessimistic: ${arr6:,.0f}',loc='left',fontweight='bold')
x7=np.linspace(0.02,0.10,300)
axes[1].plot(x7,p_ctrl6.pdf(x7),color='#DD8452',lw=2,label=f'Control {pc2:.1%}')
axes[1].plot(x7,p_trt6.pdf(x7),color='#4C72B0',lw=2,label=f'Treatment {pt2:.1%}')
axes[1].fill_between(x7,p_ctrl6.pdf(x7),alpha=0.15,color='#DD8452'); axes[1].fill_between(x7,p_trt6.pdf(x7),alpha=0.15,color='#4C72B0')
axes[1].set_title(f'Posteriors\nP(better)={pb6:.1%}',loc='left',fontweight='bold'); axes[1].legend(frameon=False)
ax=axes[2]; ax.axis('off')
dc='#2a9d8f' if ci6l>0 else '#e63946'
ax.text(0.5,0.88,'✓ SHIP',ha='center',fontsize=22,fontweight='bold',color=dc,transform=ax.transAxes)
ax.text(0.5,0.58,f'Effect: +{d6*100:.2f}pp\nARR pessimistic: ${arr6:,.0f}\nP(better): {pb6:.1%}',ha='center',fontsize=10,transform=ax.transAxes)
ax.text(0.5,0.25,'Guardrails: PASS\nCaveats: novelty possible',ha='center',fontsize=9,color='#888',style='italic',transform=ax.transAxes)
ax.set_title('Decision summary',loc='left',fontweight='bold')
for ax in axes[:2]: ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("experiment_narrative.png")

# ── null results ──────────────────────────────────────────────────────────────
mde2=0.8
cases_=[('True null\n(well-powered)',0.1,0.3,'#2a9d8f'),
        ('Inconclusive\n(underpowered)',0.4,1.5,'#f4a261'),
        ('Below MDE\n(well-powered)',0.3,0.6,'#4C72B0')]
fig,ax=plt.subplots(figsize=(10,5))
ax.axvline(0,color='#aaa',lw=1); ax.axvline(mde2,color='#e63946',lw=2,linestyle='--',label=f'MDE={mde2}pp')
ax.axvline(-mde2,color='#e63946',lw=2,linestyle='--')
ypos=[0.75,0.50,0.25]
for (lbl,eff,hci,c),y in zip(cases_,ypos):
    lo,hi=eff-hci,eff+hci
    ax.plot([lo,hi],[y,y],color=c,lw=5,alpha=0.75,solid_capstyle='round')
    ax.scatter([eff],[y],color=c,s=100,zorder=5)
    ax.text(-2.3,y,lbl,va='center',ha='right',fontsize=9,color=c,fontweight='bold')
ax.set_xlabel('Estimated effect (pp)'); ax.set_xlim(-2.5,3.5); ax.set_yticks([])
ax.set_title('Types of null results — not all nulls are equal',loc='left',fontweight='bold')
ax.legend(frameon=False); ax.spines['left'].set_color('#ddd'); ax.spines['bottom'].set_color('#ddd')
plt.tight_layout(); save("null_results.png")

print("\nAll charts generated.")
chart_files = os.listdir(CHARTS)
print(f"Total: {len(chart_files)} files in {CHARTS}/")
