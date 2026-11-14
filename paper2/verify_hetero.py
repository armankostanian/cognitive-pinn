"""Разные скорости забывания у учеников + общие неизвестные сдвиги форм на паузе.
w_n(t_j) = a_n * exp(-Lam_n t_j) + c_j,  c_j = gamma + eta_{form(j)}."""
import numpy as np
from scipy.optimize import least_squares
rng=np.random.default_rng(21)
t=np.array([0.0,7.0,18.0]); c_true=np.array([-1.7,-2.1,-1.9])
N=60; Lam=0.06*np.exp(rng.normal(0,0.3,N)); a=rng.uniform(1.0,4.0,N)
W=a[:,None]*np.exp(-Lam[:,None]*t[None,:])+c_true[None,:]

def fit(idx,starts=8):
    def res(x):
        c=x[:3]; aa=x[3:3+N]; L=np.exp(x[3+N:])
        return (aa[:,None]*np.exp(-L[:,None]*t[None,idx])+c[None,idx]-W[:,idx]).ravel()
    out=[]
    for s in range(starts):
        c0=rng.normal(-2,0.6,3)
        x0=np.r_[c0, W[:,0]-c0[0], np.log(0.06)+rng.normal(0,0.4,N)]
        r=least_squares(res,x0,x_scale='jac',xtol=1e-15,ftol=1e-15,gtol=1e-15,max_nfev=50000)
        out.append((np.sqrt(np.mean(r.fun**2)), r.x[:3], np.exp(r.x[3+N:])))
    return out

for name,idx in (("три замера на паузе",[0,1,2]),("два замера на паузе",[0,2])):
    out=fit(idx)
    rms=np.array([o[0] for o in out]); ok=[o for o in out if o[0]<1e-8]
    used=sorted(set(idx))
    if ok:
        cs=np.array([o[1][used] for o in ok]); Ls=np.array([o[2] for o in ok])
        print(f"{name}: точных подгонок {len(ok)} из {len(out)}; "
              f"разброс найденных сдвигов между подгонками {np.ptp(cs,axis=0).max():.1e}; "
              f"макс. ошибка Lambda_n {np.abs(Ls-Lam).max():.1e}")
    else:
        print(f"{name}: точных подгонок нет, rms {rms.min():.1e}")
