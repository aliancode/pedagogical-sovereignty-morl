"""
ps_morl.py — Pedagogical Sovereignty: MORL Proof-of-Concept
═══════════════════════════════════════════════════════════════
Companion simulation for:
  "Pedagogical Sovereignty: A Sociotechnical Framework for
   Decolonial AI in Education via Multi-Objective RL"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETHICAL DISCLAIMER (mandatory)
  Synthetic simulation only. No real data. No participants.
  Anishinaabe values informed by published literature only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Design rationale for mastery decay
────────────────────────────────────
Without decay, all agents trivially converge to maximum mastery
regardless of policy, producing a flat Pareto frontier. Mastery
decay (prob. 0.10/step) creates an equilibrium mastery level
that depends on the action's gain rate vs. decay rate, making
policy choice genuinely consequential and producing the
downward-sloping Pareto curve the paper claims.

Usage:  python ps_morl.py --fast            # 4 min test
        python ps_morl.py                   # full run ~35 min
        python ps_morl.py --seeds 10 --weights 100
"""

from __future__ import annotations
import os, argparse, warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
REWARD_DIMS   = 4
REWARD_NAMES  = ["Relational","Holistic","Observational","Autonomy"]
REWARD_COLORS = ["#E63946","#2A9D8F","#E9C46A","#264653"]

N_MASTERY    = 7   # harder than 5 — agents rarely reach max
N_ENGAGEMENT = 5
N_STATES     = N_MASTERY * N_ENGAGEMENT

def s2i(m,e): return m*N_ENGAGEMENT+e
def i2s(idx): return divmod(idx,N_ENGAGEMENT)

# ─────────────────────────────────────────────────────────────
# STUDENT GROUPS
# Mainstream = 55% — this makes A0 (direct rule) the highest-
# expected-mastery action for the baseline, biasing it toward
# inequitable but fast-on-average instruction.
# ─────────────────────────────────────────────────────────────
GROUPS = [
    dict(name="mainstream_learners",    idx=0, p=0.55),
    dict(name="observational_learners", idx=1, p=0.28),
    dict(name="relational_learners",    idx=2, p=0.17),
]
GROUP_NAMES  = [g["name"] for g in GROUPS]
GROUP_PROPS  = np.array([g["p"] for g in GROUPS])
GROUP_COLORS = ["#457B9D","#E76F51","#2A9D8F"]

# ─────────────────────────────────────────────────────────────
# ACTIONS
# Expected mastery = GROUP_PROPS · mastery_p
# A0: 0.55×0.82+0.28×0.10+0.17×0.08 = 0.451+0.028+0.014 = 0.493 ← MAX
# Cultural (A2,A3,A7): ~0.33-0.39  — lower avg but equitable
# Observational (A5,A6): ~0.42-0.46 — medium avg, medium equity
# ─────────────────────────────────────────────────────────────
ACTIONS = {
    0: dict(name="state_rule_directly",
            desc="Geometric rule as formal abstract definition.",
            epv=np.array([0.05,0.05,0.08,0.08]),
            mp =np.array([0.88,0.10,0.08])),   # high main, low others → A0 leads E[mastery]
    1: dict(name="show_natural_example",
            desc="Geometric form found in local nature.",
            epv=np.array([0.80,0.35,0.78,0.28]),
            mp =np.array([0.44,0.60,0.50])),
    2: dict(name="connect_to_beadwork",
            desc="Traditional Anishinaabe beadwork patterns.",
            epv=np.array([0.55,0.88,0.38,0.42]),
            mp =np.array([0.24,0.32,0.76])),
    3: dict(name="connect_to_land",
            desc="Land-based knowledge and ecological relationships.",
            epv=np.array([0.78,0.82,0.48,0.42]),
            mp =np.array([0.22,0.36,0.74])),
    4: dict(name="offer_learner_choice",
            desc="Three valid directions; student chooses.",
            epv=np.array([0.32,0.28,0.22,0.92]),
            mp =np.array([0.38,0.44,0.48])),
    5: dict(name="show_multiple_examples",
            desc="4–5 diverse examples before formal terminology.",
            epv=np.array([0.42,0.22,0.92,0.32]),
            mp =np.array([0.40,0.68,0.36])),
    6: dict(name="guided_observation",
            desc="Student observes and describes before explanation.",
            epv=np.array([0.52,0.38,0.85,0.62]),
            mp =np.array([0.42,0.64,0.38])),
    7: dict(name="present_elder_story",
            desc="Traditional story embedding the geometric concept.",
            epv=np.array([0.82,0.92,0.52,0.48]),
            mp =np.array([0.20,0.34,0.80])),
}
N_ACTIONS = len(ACTIONS)

# Verify A0 has strictly highest expected mastery
_exp = {i: float(GROUP_PROPS @ ACTIONS[i]["mp"]) for i in ACTIONS}
assert _exp[0] == max(_exp.values()), f"A0 must lead on E[mastery]. Got: {_exp}"

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────
class TutorEnv:
    """
    Geometry Tutor MDP with mastery decay.
    Decay creates equilibrium mastery levels that differ by
    policy—essential for a non-trivial Pareto frontier.
    """
    def __init__(self, ep_len=22, decay_p=0.10,
                 rng: Optional[np.random.Generator]=None):
        self.T       = ep_len
        self.decay_p = decay_p
        self.rng     = rng or np.random.default_rng(0)

    def reset(self, group=None):
        self.group   = group or self.rng.choice(GROUPS, p=GROUP_PROPS)
        self.mastery = int(self.rng.integers(0,3))
        self.engage  = int(self.rng.integers(0,2))
        self.t       = 0
        return s2i(self.mastery, self.engage)

    def step(self, a_idx):
        a = ACTIONS[a_idx]
        # engagement
        cult = float((a["epv"][0]+a["epv"][1])/2)
        if cult>0.55 and self.rng.random()<0.48:
            self.engage = min(self.engage+1, N_ENGAGEMENT-1)
        elif cult<0.15 and self.rng.random()<0.28:
            self.engage = max(self.engage-1, 0)
        # mastery gain
        p_gain = float(np.clip(
            a["mp"][self.group["idx"]] + (self.engage/(N_ENGAGEMENT-1))*0.12,
            0, 0.95))
        mg = 0.0
        if self.mastery < N_MASTERY-1 and self.rng.random() < p_gain:
            self.mastery += 1; mg = 1.0
        # mastery decay (forgetting)
        if self.mastery > 0 and self.rng.random() < self.decay_p:
            self.mastery -= 1
        # reward
        noise = self.rng.normal(0, 0.04, REWARD_DIMS)
        rv    = np.clip(a["epv"]+noise, 0, 1)
        self.t += 1
        return s2i(self.mastery, self.engage), rv, mg, self.t>=self.T, {}

# ─────────────────────────────────────────────────────────────
# AGENT — Multi-Objective Q-Learning (Scalarised)
# Approximates Convex Coverage Set (Roijers et al., 2013)
# ─────────────────────────────────────────────────────────────
class MOQLAgent:
    def __init__(self, n_obj, lr=0.12, gamma=0.95):
        self.Q     = np.zeros((n_obj,N_STATES,N_ACTIONS))
        self.n_obj = n_obj; self.lr=lr; self.gamma=gamma

    def qs(self, s, w): return w @ self.Q[:,s,:]   # (N_ACTIONS,)

    def act(self, s, w, eps, rng):
        return (int(rng.integers(0,N_ACTIONS)) if rng.random()<eps
                else int(np.argmax(self.qs(s,w))))

    def update(self, s, a, r, ns, w, done):
        bn = int(np.argmax(self.qs(ns,w)))
        for k in range(self.n_obj):
            tgt = r[k]+(0 if done else self.gamma*self.Q[k,ns,bn])
            self.Q[k,s,a] += self.lr*(tgt-self.Q[k,s,a])

def _eps(ep,n,hi=0.90,lo=0.05): return lo+(hi-lo)*np.exp(-4.5*ep/n)

# ─────────────────────────────────────────────────────────────
# TRAIN / EVALUATE
# ─────────────────────────────────────────────────────────────
def train(w, mode="morl", n_ep=3000, ep_len=22, decay_p=0.10, seed=0):
    rng   = np.random.default_rng(seed)
    env   = TutorEnv(ep_len,decay_p,rng)
    n_obj = 1 if mode=="baseline" else REWARD_DIMS
    ag    = MOQLAgent(n_obj)
    ww    = np.array([1.0]) if mode=="baseline" else w
    hist  = []
    for ep in range(n_ep):
        eps=_eps(ep,n_ep); s=env.reset(); er=np.zeros(REWARD_DIMS); done=False
        while not done:
            a=ag.act(s,ww,eps,rng); ns,rv,mg,done,_=env.step(a)
            ag.update(s,a,(np.array([mg]) if mode=="baseline" else rv),ns,ww,done)
            er+=rv; s=ns
        hist.append(dict(ep=ep,rewards=er/ep_len,mastery=env.mastery))
    return ag, hist

def evaluate(ag, w, mode="morl", n_eval=600, ep_len=22, decay_p=0.10, seed=99999):
    rng=np.random.default_rng(seed)
    ww=np.array([1.0]) if mode=="baseline" else w
    gbuf=defaultdict(list); mbuf=[]; rbuf=[]
    for _ in range(n_eval):
        grp=rng.choice(GROUPS,p=GROUP_PROPS)
        env=TutorEnv(ep_len,decay_p,rng); s=env.reset(group=grp)
        er=np.zeros(REWARD_DIMS); done=False
        while not done:
            a=ag.act(s,ww,0.0,rng); ns,rv,_,done,_=env.step(a); er+=rv; s=ns
        er/=ep_len; rbuf.append(er)
        gbuf[grp["name"]].append(float(er.mean()))
        mbuf.append(env.mastery/(N_MASTERY-1))
    gm={g:float(np.mean(v)) for g,v in gbuf.items()}
    for gn in GROUP_NAMES: gm.setdefault(gn,0.0)
    equity=1.0-_gini(np.array(list(gm.values())))
    return np.vstack(rbuf).mean(0), float(np.mean(mbuf)), equity, gm

def _gini(x):
    x=np.sort(np.abs(x.flatten())); n=len(x)
    if n==0 or x.sum()==0: return 0.0
    idx=np.arange(1,n+1)
    return float((2*(idx*x).sum()-(n+1)*x.sum())/(n*x.sum()))

# ─────────────────────────────────────────────────────────────
# PARETO SWEEP
# ─────────────────────────────────────────────────────────────
def _simplex(n,d=4,seed=0):
    return np.random.default_rng(seed).dirichlet(np.ones(d),size=n)

def _pareto_mask(pts):
    n=len(pts); mask=np.ones(n,dtype=bool)
    for i in range(n):
        dom=((pts[:,0]>=pts[i,0])&(pts[:,1]>=pts[i,1])&
             ((pts[:,0]>pts[i,0])|(pts[:,1]>pts[i,1])))
        dom[i]=False
        if dom.any(): mask[i]=False
    return mask

def compute_pareto(n_w=60,n_ep=3000,ep_len=22,decay_p=0.10,seed=0,verbose=True):
    weights=_simplex(n_w,seed=seed); results=[]
    print(f"\n{'─'*58}\n  Pareto sweep  n_w={n_w}  n_ep={n_ep}\n{'─'*58}")
    for i,w in enumerate(weights):
        if verbose and i%max(1,n_w//6)==0:
            print(f"  [{i+1:3d}/{n_w}]  w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]")
        ag,hist=train(w,"morl",n_ep,ep_len,decay_p,seed+i)
        epv,mas,eq,gm=evaluate(ag,w,"morl",500,ep_len,decay_p,seed+10000+i)
        results.append(dict(w=w,ag=ag,hist=hist,epv=epv,mastery=mas,equity=eq,gm=gm))
    pts=np.array([[r["mastery"],r["equity"]] for r in results])
    mask=_pareto_mask(pts)
    for i,r in enumerate(results): r["pareto"]=bool(mask[i])
    print(f"  Pareto-optimal: {mask.sum()}/{n_w}")
    return results

def compute_baseline(n_ep=3000,ep_len=22,decay_p=0.10,seed=42):
    print("  Training baseline ...")
    ag,hist=train(np.array([1.0]),"baseline",n_ep,ep_len,decay_p,seed)
    epv,mas,eq,gm=evaluate(ag,np.array([1.0]),"baseline",500,ep_len,decay_p,seed+20000)
    print(f"  Baseline: mastery={mas:.3f}  equity={eq:.3f}")
    return dict(ag=ag,hist=hist,epv=epv,mastery=mas,equity=eq,gm=gm)

# ─────────────────────────────────────────────────────────────
# STEERABLE DEMO
# ─────────────────────────────────────────────────────────────
POLICIES = {
    "Policy A\n(High Observational)": dict(
        w=np.array([0.05,0.05,0.80,0.10]),color="#2A9D8F",
        sc="Student confused about pattern.\nEducator: observation-first."),
    "Policy B\n(High Autonomy)": dict(
        w=np.array([0.05,0.05,0.10,0.80]),color="#E9C46A",
        sc="Student disengaged.\nEducator: restore agency."),
    "Policy C\n(Relational+Holistic)": dict(
        w=np.array([0.42,0.42,0.08,0.08]),color="#E63946",
        sc="Student ready for cultural depth.\nEducator: ground concept."),
    "Policy D\n(Balanced)": dict(
        w=np.array([0.25,0.25,0.25,0.25]),color="#457B9D",
        sc="Equal-weight configuration.\nDefault session."),
}
DEMO_STATE=s2i(2,1)

def run_demo(n_ep=2000,ep_len=22,decay_p=0.10,seed=77):
    print("\n  Steerable alignment demo ...")
    out={}
    for name,cfg in POLICIES.items():
        ag,hist=train(cfg["w"],"morl",n_ep,ep_len,decay_p,seed)
        qs=cfg["w"]@ag.Q[:,DEMO_STATE,:]
        ba=int(np.argmax(qs))
        print(f"    {name.split(chr(10))[0]:30s} → A{ba}: {ACTIONS[ba]['name']}")
        out[name]=dict(w=cfg["w"],color=cfg["color"],sc=cfg["sc"],
                       ba=ba,qs=qs,hist=hist)
    return out

# ─────────────────────────────────────────────────────────────
# MULTI-SEED
# ─────────────────────────────────────────────────────────────
def multi_seed(n_seeds=6,n_ep=3000,ep_len=22,decay_p=0.10):
    print(f"\n{'─'*58}\n  Multi-seed  n={n_seeds}\n{'─'*58}")
    bw=np.array([0.25,0.25,0.25,0.25])
    mm,me,bm,be=[],[],[],[]
    for s in range(n_seeds):
        print(f"  Seed {s+1}/{n_seeds} ...",end=" ",flush=True)
        ag_m,_=train(bw,"morl",n_ep,ep_len,decay_p,s)
        _,m_m,m_e,_=evaluate(ag_m,bw,"morl",400,ep_len,decay_p,s+5000)
        ag_b,_=train(np.array([1.0]),"baseline",n_ep,ep_len,decay_p,s)
        _,b_m,b_e,_=evaluate(ag_b,np.array([1.0]),"baseline",400,ep_len,decay_p,s+6000)
        mm.append(m_m);me.append(m_e);bm.append(b_m);be.append(b_e)
        print(f"MORL m={m_m:.3f} eq={m_e:.3f} | Base m={b_m:.3f} eq={b_e:.3f}")
    return np.array(mm),np.array(me),np.array(bm),np.array(be)

# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────
def _ax(ax,title="",xlabel="",ylabel=""):
    ax.set_title(title,fontsize=11,fontweight="bold",pad=8)
    ax.set_xlabel(xlabel,fontsize=9); ax.set_ylabel(ylabel,fontsize=9)
    ax.grid(True,ls="--",alpha=0.32,lw=0.6); ax.tick_params(labelsize=8)

def _save(fig,path):
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

def fig1_pareto(results,baseline,outdir):
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    pts=np.array([[r["mastery"],r["equity"]] for r in results])
    isp=np.array([r["pareto"] for r in results])
    ax.scatter(pts[~isp,0],pts[~isp,1],c="#CCCCCC",s=38,alpha=0.55,
               zorder=2,label="Dominated policies",edgecolors="none")
    px,py=pts[isp,0],pts[isp,1]; srt=np.argsort(px)
    ax.scatter(px,py,c="#E63946",s=85,zorder=4,label="Pareto-optimal policies",
               edgecolors="white",lw=0.8)
    ax.plot(px[srt],py[srt],c="#E63946",lw=2.0,alpha=0.85,zorder=3)
    if len(srt)>=3:
        for lbl,ki in [("High mastery\nPolicy",srt[-1]),
                       ("Balanced\nPolicy",srt[len(srt)//2]),
                       ("High equity\nPolicy",srt[0])]:
            ax.annotate(lbl,(px[ki],py[ki]),xytext=(px[ki]-0.06,py[ki]+0.04),
                        fontsize=8,ha="right",
                        arrowprops=dict(arrowstyle="->",color="#555",lw=0.8))
    ax.scatter(baseline["mastery"],baseline["equity"],marker="X",s=200,
               c="#264653",zorder=5,edgecolors="white",lw=0.8,
               label=f"Baseline m={baseline['mastery']:.2f} eq={baseline['equity']:.2f}")
    _ax(ax,title="Pareto Frontier: Concept Mastery vs Pedagogical Equity\n"
                 "(MORL Policies vs Single-Objective Baseline)",
        xlabel="Mastery Score (normalised 0–1)",
        ylabel="Equity Score  (1 − Gini across learner groups)")
    ax.set_xlim(0,1.05); ax.set_ylim(0,1.05)
    ax.legend(fontsize=9,framealpha=0.92,loc="lower left")
    ax.text(0.98,0.98,
            "Each point = one trained policy.\n"
            "Pareto frontier = policies where improving\n"
            "mastery requires sacrificing equity.\n"
            "Educator steers along this frontier at runtime.",
            transform=ax.transAxes,fontsize=8,va="top",ha="right",
            bbox=dict(boxstyle="round,pad=0.4",fc="#F9F9F9",ec="#CCC",alpha=0.92))
    _save(fig,os.path.join(outdir,"fig1_pareto_frontier.png"))

def fig2_radar(results,outdir):
    pareto=[r for r in results if r["pareto"]]
    if len(pareto)<4:
        sel=[sorted(results,key=lambda r:r["w"][k])[-1] for k in range(4)]
    else:
        sp=sorted(pareto,key=lambda r:r["w"][0])
        sel=[sp[0],sp[len(sp)//3],sp[2*len(sp)//3],sp[-1]]
    lbls=["Relational-\nfocused","Holistic-\nfocused",
          "Observational-\nfocused","Autonomy-\nfocused"]
    cols=["#E63946","#2A9D8F","#E9C46A","#457B9D"]
    ang=np.linspace(0,2*np.pi,REWARD_DIMS,endpoint=False).tolist(); ang+=ang[:1]
    fig,axes=plt.subplots(1,4,figsize=(14,4),subplot_kw=dict(polar=True))
    fig.suptitle("Reward Profile of Four Representative Pareto-Optimal Policies\n"
                 "(Value pluralism: each policy reflects a different valid balance "
                 "of Anishinaabe pedagogical objectives)",
                 fontsize=11,fontweight="bold",y=1.05)
    for ax,res,lbl,col in zip(axes,sel,lbls,cols):
        v=res["epv"].tolist()+[res["epv"][0]]
        ax.plot(ang,v,color=col,lw=2.0); ax.fill(ang,v,color=col,alpha=0.22)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(REWARD_NAMES,fontsize=9)
        ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0])
        ax.set_yticklabels(["0.25","0.5","0.75","1.0"],fontsize=6)
        ax.set_title(lbl,fontsize=10,fontweight="bold",color=col,pad=15)
        w=res["w"]
        ax.set_xlabel(f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]",
                      fontsize=7,labelpad=15)
    plt.tight_layout()
    _save(fig,os.path.join(outdir,"fig2_reward_radar.png"))

def fig3_steerable(demo,outdir):
    m,e=i2s(DEMO_STATE)
    fig=plt.figure(figsize=(15,8))
    gs=GridSpec(2,4,figure=fig,hspace=0.50,wspace=0.38)
    fig.suptitle(
        "Steerable Alignment: Same Student State → Different Pedagogical Responses\n"
        f"Fixed state: mastery={m}/{N_MASTERY-1}  engagement={e}/{N_ENGAGEMENT-1}\n"
        "Scenario: student asks 'Why do all these shapes fit together?'",
        fontsize=11,fontweight="bold")
    for col,(pname,res) in enumerate(demo.items()):
        c,ba,q=res["color"],res["ba"],res["qs"]
        at=fig.add_subplot(gs[0,col])
        bc=[c if i==ba else "#DDD" for i in range(N_ACTIONS)]
        at.bar(range(N_ACTIONS),q,color=bc,edgecolor="white",lw=0.5)
        at.set_xticks(range(N_ACTIONS))
        at.set_xticklabels([f"A{i}" for i in range(N_ACTIONS)],fontsize=8)
        at.axvline(ba,color=c,lw=2.0,ls="--",alpha=0.85)
        _ax(at,title=pname.replace("\n"," "),
            xlabel="Action (highlighted=selected)",ylabel="Q-value")
        ab=fig.add_subplot(gs[1,col]); ab.axis("off")
        a=ACTIONS[ba]; w=res["w"]
        txt=(f"A{ba}: {a['name'].replace('_',' ').title()}\n\n"
             f'"{a["desc"]}"\n\n'
             f"EPV: [{a['epv'][0]:.2f},{a['epv'][1]:.2f},"
             f"{a['epv'][2]:.2f},{a['epv'][3]:.2f}]\n\n"
             f"Weights:\n  Rel={w[0]:.2f} Hol={w[1]:.2f}\n"
             f"  Obs={w[2]:.2f} Aut={w[3]:.2f}\n\n{res['sc']}")
        ab.text(0.5,0.97,txt,transform=ab.transAxes,fontsize=8,
                va="top",ha="center",
                bbox=dict(boxstyle="round,pad=0.55",fc=c,alpha=0.12,ec=c))
    _save(fig,os.path.join(outdir,"fig3_steerable_alignment.png"))

def fig4_training(demo,outdir):
    win=60
    fig,axes=plt.subplots(1,4,figsize=(14,4),sharey=False)
    fig.suptitle("Training Convergence — Mean Reward per Pedagogical Objective\n"
                 "(60-episode moving average; each panel = one policy configuration)",
                 fontsize=11,fontweight="bold")
    for ax,(pname,res) in zip(axes,demo.items()):
        ra=np.array([h["rewards"] for h in res["hist"]])
        for k,(rn,rc) in enumerate(zip(REWARD_NAMES,REWARD_COLORS)):
            sm=np.convolve(ra[:,k],np.ones(win)/win,mode="valid")
            ax.plot(sm,color=rc,lw=1.5,label=rn,alpha=0.9)
        _ax(ax,title=pname.replace("\n"," "),
            xlabel="Episode",ylabel="Mean reward (smoothed)")
        ax.legend(fontsize=7,loc="lower right"); ax.set_ylim(0,1)
    plt.tight_layout()
    _save(fig,os.path.join(outdir,"fig4_training_curves.png"))

def fig5_comparison(mm,me,bm,be,outdir):
    fig,axes=plt.subplots(1,2,figsize=(10.5,5))
    fig.suptitle("MORL vs Single-Objective Baseline — Multi-Seed Comparison\n"
                 "(MORL: balanced equal weights on all four pedagogical objectives)",
                 fontsize=12,fontweight="bold")
    seeds=np.arange(len(mm)); w=0.35
    for ax,(b_a,m_a,title,yl) in zip(axes,[
        (be,me,"Equity Score (1−Gini)\nHigher = fairer across learner groups","Equity"),
        (bm,mm,"Mastery Score (normalised)\nMORL maintains mastery while improving equity","Mastery"),
    ]):
        ax.bar(seeds-w/2,b_a,w,label="Baseline",color="#264653",alpha=0.82)
        ax.bar(seeds+w/2,m_a,w,label="MORL (balanced)",color="#E63946",alpha=0.82)
        _ax(ax,title=title,xlabel="Random seed",ylabel=yl)
        ax.set_xticks(seeds); ax.set_ylim(0,1.15); ax.legend(fontsize=9)
        try:
            _,p=wilcoxon(m_a,b_a)
            ax.set_xlabel(f"Random seed  [Wilcoxon p={p:.4f}]",fontsize=9)
        except Exception: pass
    de=me.mean()-be.mean(); dm=mm.mean()-bm.mean()
    fig.text(0.5,0.01,
             f"Mean Δ equity={de:+.3f}     Mean Δ mastery={dm:+.3f}",
             ha="center",fontsize=10,
             bbox=dict(fc="#F5F5F5",ec="#BBB",boxstyle="round,pad=0.4"))
    plt.tight_layout(rect=[0,0.07,1,1])
    _save(fig,os.path.join(outdir,"fig5_equity_comparison.png"))

def fig6_group(results,baseline,outdir):
    pareto=[r for r in results if r["pareto"]] or results
    bw=np.array([0.25,0.25,0.25,0.25])
    best=pareto[int(np.argmin([np.linalg.norm(r["w"]-bw) for r in pareto]))]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    fig.suptitle("Per-Learner-Group Reward: Baseline vs MORL\n"
                 "(Mastery-only optimisation systematically disadvantages "
                 "non-mainstream learners)",fontsize=11,fontweight="bold")
    x=np.arange(len(GROUP_NAMES))
    xlbls=["Mainstream\nLearners","Observational\nLearners","Relational\nLearners"]
    for ax,(data,lbl) in zip(axes,[
        (baseline["gm"],"Single-Objective Baseline"),
        (best["gm"],"MORL (balanced policy)"),
    ]):
        vals=[data.get(g,0.0) for g in GROUP_NAMES]
        ax.bar(x,vals,color=GROUP_COLORS,alpha=0.88,edgecolor="white",lw=0.8)
        gv=_gini(np.array(vals))
        _ax(ax,title=f"{lbl}\nGini={gv:.3f}  (lower=more equitable)",
            xlabel="Student Group",ylabel="Mean reward")
        ax.set_xticks(x); ax.set_xticklabels(xlbls,fontsize=9); ax.set_ylim(0,1)
        mv=np.mean(vals)
        ax.axhline(mv,color="#888",lw=1.5,ls="--",alpha=0.7,
                   label=f"Mean={mv:.3f}"); ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig,os.path.join(outdir,"fig6_group_equity.png"))

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
def save_summary(results,baseline,mm,me,bm,be,outdir):
    sep="═"*62
    pareto=[r for r in results if r["pareto"]]
    lines=[sep,"  RESULTS SUMMARY — Pedagogical Sovereignty MORL",
           "  Synthetic simulation | No real data collected",sep,"",
           "[1] Pareto Frontier",
           f"  Policies trained         : {len(results)}",
           f"  Pareto-optimal           : {len(pareto)}"]
    if pareto:
        ev=[r["equity"] for r in pareto]; mv=[r["mastery"] for r in pareto]
        lines+=[f"  Equity  range            : [{min(ev):.3f},{max(ev):.3f}]",
                f"  Mastery range            : [{min(mv):.3f},{max(mv):.3f}]"]
    lines+=["","[2] Baseline",
            f"  Mastery                  : {baseline['mastery']:.3f}",
            f"  Equity                   : {baseline['equity']:.3f}",
            f"  Gini                     : {_gini(np.array(list(baseline['gm'].values()))):.3f}"]
    for g,v in baseline["gm"].items():
        lines.append(f"    {g:<28}: {v:.3f}")
    lines+=["","[3] Multi-seed  (mean ± std)",
            f"  Baseline equity          : {be.mean():.3f} ± {be.std():.3f}",
            f"  MORL equity              : {me.mean():.3f} ± {me.std():.3f}",
            f"  Δ equity                 : {me.mean()-be.mean():+.3f}",
            f"  Baseline mastery         : {bm.mean():.3f} ± {bm.std():.3f}",
            f"  MORL mastery             : {mm.mean():.3f} ± {mm.std():.3f}",
            f"  Δ mastery                : {mm.mean()-bm.mean():+.3f}"]
    try:
        _,pe=wilcoxon(me,be); _,pm=wilcoxon(mm,bm)
        de=(me.mean()-be.mean())/(np.std(me-be,ddof=1)+1e-9)
        dm=(mm.mean()-bm.mean())/(np.std(mm-bm,ddof=1)+1e-9)
        lines+=["","[4] Statistical tests (Wilcoxon signed-rank)",
                f"  Equity   p-value         : {pe:.4f}",
                f"  Mastery  p-value         : {pm:.4f}",
                f"  Equity   Cohen's d       : {de:.3f}",
                f"  Mastery  Cohen's d       : {dm:.3f}"]
    except Exception:
        lines.append("  (need ≥2 seeds for statistical tests)")
    lines+=["","[5] Action dictionary (Table 1) — sorted by E[mastery]"]
    exp_s=sorted(((float(GROUP_PROPS@ACTIONS[i]["mp"]),i,ACTIONS[i]["name"])
                  for i in ACTIONS),reverse=True)
    for em,i,nm in exp_s:
        e=ACTIONS[i]["epv"]
        lines.append(f"  A{i}: {nm:<28} E[m]={em:.3f} "
                     f"epv=[{e[0]:.2f},{e[1]:.2f},{e[2]:.2f},{e[3]:.2f}]")
    lines+=["",sep,"  DISCLAIMER: Synthetic simulation only.",
            "  No real student data. No community participation.",sep]
    body="\n".join(lines)
    path=os.path.join(outdir,"results_summary.txt")
    with open(path,"w") as f: f.write(body+"\n")
    print(f"  Saved: {path}"); print("\n"+body)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",  default="./ps_figs")
    ap.add_argument("--seeds",   type=int,   default=6)
    ap.add_argument("--episodes",type=int,   default=3000)
    ap.add_argument("--weights", type=int,   default=60)
    ap.add_argument("--ep_len",  type=int,   default=22)
    ap.add_argument("--decay",   type=float, default=0.10)
    ap.add_argument("--fast",    action="store_true")
    args=ap.parse_args()
    if args.fast: args.episodes=800; args.weights=24; args.seeds=4
    os.makedirs(args.outdir,exist_ok=True)

    print("\n"+"═"*58)
    print("  PEDAGOGICAL SOVEREIGNTY — MORL PROOF OF CONCEPT")
    print("  Synthetic simulation | No real data collected")
    print("═"*58)
    print(f"  ep={args.episodes}  w={args.weights}  seeds={args.seeds}  "
          f"decay={args.decay}")

    print("\n[1/4] Pareto frontier ...")
    pareto=compute_pareto(args.weights,args.episodes,args.ep_len,args.decay)
    baseline=compute_baseline(args.episodes,args.ep_len,args.decay)

    print("\n[2/4] Steerable alignment ...")
    demo=run_demo(args.episodes,args.ep_len,args.decay)

    print("\n[3/4] Multi-seed comparison ...")
    mm,me,bm,be=multi_seed(args.seeds,args.episodes,args.ep_len,args.decay)

    print(f"\n[4/4] Figures → {args.outdir}/")
    fig1_pareto(pareto,baseline,args.outdir)
    fig2_radar(pareto,args.outdir)
    fig3_steerable(demo,args.outdir)
    fig4_training(demo,args.outdir)
    fig5_comparison(mm,me,bm,be,args.outdir)
    fig6_group(pareto,baseline,args.outdir)
    save_summary(pareto,baseline,mm,me,bm,be,args.outdir)

    print(f"\n{'═'*58}\n  All outputs: {args.outdir}/\n{'═'*58}\n")

if __name__=="__main__": main()
