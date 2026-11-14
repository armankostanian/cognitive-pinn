"""Разная трудность форм: w_n = beta_i*k_n + gamma_i + eta_{i,f}, eta_{F01}=0 (якорная форма).
Популяционный режим: параметры общие, стартовые уровни учеников различаются."""
import numpy as np
from scipy.optimize import brentq
rng=np.random.default_rng(11)
def step(k,u,Lam,h): s=u+Lam; ks=u/s; return ks+(k-ks)*np.exp(-s*h)
H=lambda r,x:(1-np.exp(-r*x))/(1-np.exp(-x))

HS=1/24
# план эксперимента (K=2, порядок S1S2): (день, что до следующего замера)
def build_plan(mid_rest=True):
    ev=[]  # список замеров: (day, form)
    rot=["F01","F02","F03","F04","F05","F06","F07","F08","F09","F10","F11","F12",
         "F01","F02","F03","F04","F05","F06","F07","F08"]
    probes=[("P00",0.0)]
    for d in (7,10,14,17): probes+= [(f"pre",d),(f"post",d+HS)]
    probes.append(("P09",24.0))
    for d in (35,38,42,45): probes+= [(f"pre",d),(f"post",d+HS)]
    probes.append(("P18",49.0)); probes.append(("P19",63.0))
    ints=[10,4,4,10]
    plan=[]  # (t, form, practice-вектор ДО следующего замера на промежутке)
    for idx,((name,t),f) in enumerate(zip(probes,rot)):
        plan.append([name,t,f])
    # практика: между pre и post соответствующего занятия
    pr=[]
    for idx in range(len(plan)-1):
        a,b=plan[idx],plan[idx+1]
        p=np.zeros(2)
        if a[0]=="pre":
            blk=0 if a[1]<30 else 1
            n=[1,3,5,7] if blk==0 else [10,12,14,16]
            p[blk]=ints[n.index(idx)]
        pr.append(p)
    if not mid_rest:   # как у группы REDUCED: без P09 и P18
        keep=[i for i,x in enumerate(plan) if x[0] not in ("P09","P18")]
        # пересобрать промежутки: практика на сохранённых промежутках не меняется (P09/P18 внутри пауз)
        plan2=[plan[i] for i in keep]; pr2=[]
        for a,b in zip(keep[:-1],keep[1:]): pr2.append(sum(pr[a:b],np.zeros(2)) if b-a==1 else np.zeros(2))
        return plan2,pr2
    return plan,pr

def simulate(plan,pr,A,Lam,al,beta,gam,eta,N=300):
    K=2; k=rng.uniform(0.15,0.85,(N,K)); W=[]
    for m,(name,t,f) in enumerate(plan):
        W.append(beta*k+gam+np.array([eta[i].get(f,0.0) for i in range(K)]))
        if m<len(plan)-1:
            h=plan[m+1][1]-t; u=A@(1-np.exp(-al*pr[m]))
            k=step(k,u,Lam,h)
    return np.array(W)   # (M,N,K)

def reconstruct(plan,pr,W):
    K=2; M=len(plan)
    slope=np.zeros((M-1,K)); icpt=np.zeros((M-1,K))
    for m in range(M-1):
        for i in range(K):
            X=np.c_[W[m][:,i],np.ones(W.shape[1])]; c=np.linalg.lstsq(X,W[m+1][:,i],rcond=None)[0]
            slope[m,i],icpt[m,i]=c
    dt=np.array([plan[m+1][1]-plan[m][1] for m in range(M-1)])
    rest=[m for m in range(M-1) if not pr[m].any()]
    Lam=np.array([np.mean([-np.log(slope[m,i])/dt[m] for m in rest]) for i in range(K)])
    s=-np.log(slope)/dt[:,None]; u=np.clip(s-Lam,0,None)
    # A, alpha из откликов изолированных занятий
    A=np.zeros((K,K)); al=np.zeros(K)
    for j in range(K):
        ua=np.mean([u[m] for m in range(M-1) if pr[m][j]==4],0)
        ub=np.mean([u[m] for m in range(M-1) if pr[m][j]==10],0)
        al[j]=brentq(lambda x:H(2.5,x)-ub[j]/ua[j],1e-12,500,xtol=1e-15)/4
        A[:,j]=ua/(1-np.exp(-al[j]*4))
    # линейная система для (beta, gamma, eta_F02..): C = (1-M)(beta*u/s + gamma) + eta_next - M*eta_prev
    forms=sorted({x[2] for x in plan}-{"F01"}); col={f:2+q for q,f in enumerate(forms)}
    res={}
    for i in range(K):
        rows=[];rhs=[]
        for m in range(M-1):
            r=np.zeros(2+len(forms)); Mk=slope[m,i]; sk=s[m,i]; uk=u[m,i]
            r[0]=(1-Mk)*(uk/sk if uk>1e-12 else 0.0); r[1]=1-Mk
            fn,fp=plan[m+1][2],plan[m][2]
            if fn in col: r[col[fn]]+=1
            if fp in col: r[col[fp]]-=Mk
            rows.append(r); rhs.append(icpt[m,i])
        Amat=np.array(rows); x=np.linalg.lstsq(Amat,np.array(rhs),rcond=None)[0]
        res[i]=(x,np.linalg.matrix_rank(Amat,tol=1e-9),Amat.shape[1],dict(zip(forms,x[2:])))
    return A,Lam,al,res

A=np.array([[5.0,0.8],[1.5,4.0]]); Lam=np.array([0.05,0.07]); al=np.array([0.12,0.10])
beta=np.array([5.0,4.0]); gam=np.array([-2.0,-1.5])
eta=[{f:rng.normal(0,0.3) for f in [f"F{q:02d}" for q in range(2,13)]} for _ in range(2)]
for mid in (True,False):
    plan,pr=build_plan(mid); W=simulate(plan,pr,A,Lam,al,beta,gam,eta)
    A_,L_,a_,res=reconstruct(plan,pr,W)
    print(f"\n=== {'полный план' if mid else 'без средних замеров пауз (REDUCED)'} ===")
    print(f"  динамика по наклонам: отн. ошибка A {np.abs(A_-A).max()/A.max():.1e}, "
          f"Lambda {np.abs(L_-Lam).max()/Lam.max():.1e}, alpha {np.abs(a_-al).max()/al.max():.1e}")
    for i in range(2):
        x,rank,n,etah=res[i]
        err_eta=max(abs(etah[f]-eta[i][f]) for f in etah)
        print(f"  тема {i+1}: ранг системы {rank} из {n};  beta {x[0]:.6f} (ист. {beta[i]}), "
              f"gamma {x[1]:.6f} (ист. {gam[i]}), макс. ошибка сдвигов форм {err_eta:.1e}")
