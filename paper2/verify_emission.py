"""Идентифицируемость при логистической модели наблюдения:
наблюдается w_i(t) = logit P(верно) = beta_i*k_i(t) + gamma_i, (beta_i>0, gamma_i) неизвестны."""
import numpy as np
from scipy.optimize import brentq
rng = np.random.default_rng(7)
def step(k,u,Lam,h): s=u+Lam; ks=u/s; return ks+(k-ks)*np.exp(-s*h)
def uof(A,al,p): return A@(1-np.exp(-al*p))
H=lambda r,x:(1-np.exp(-r*x))/(1-np.exp(-x))

def design(A,Lam,al,k0,beta,gam,hs=1/24):
    """Для каждой темы j: пауза с ТРЕМЯ замерами (0, 7, 18 сут), затем 4 изолированных занятия
    темы j (10,4,4,10 задач) с замерами до/после, между занятиями 3-4 суток. Замер = w всех тем."""
    K=len(Lam); k=k0.copy(); W=lambda k: beta*k+gam; rest3={}; sess=[]
    for j in range(K):
        w0=W(k); k=step(k,0*k,Lam,7.0); w1=W(k); k=step(k,0*k,Lam,11.0); w2=W(k)
        rest3[j]=(w0,w1,w2,7.0,18.0)
        for p,gap in ((10,3.0),(4,4.0),(4,3.0),(10,4.0)):
            pv=np.zeros(K); pv[j]=p; wa=W(k); k=step(k,uof(A,al,pv),Lam,hs); wb=W(k)
            sess.append((j,p,hs,wa,wb)); k=step(k,0*k,Lam,gap)
    return rest3,sess

def reconstruct(rest3,sess,K):
    Lam=np.zeros(K); gam=np.zeros(K); beta=np.zeros(K)
    for i in range(K):                        # (1) Lambda_i, gamma_i по трём замерам паузы
        w0,w1,w2,t1,t2=(x[i] if hasattr(x,'__len__') else x for x in rest3[i])
        r=t2/t1; R=(w2-w0)/(w1-w0)
        x=brentq(lambda x:H(r,x)-R,1e-12,500,xtol=1e-15); Lam[i]=x/t1
        gam[i]=w0-(w1-w0)/(np.exp(-Lam[i]*t1)-1)
    for i in range(K):                        # (2) beta_i по двум занятиям темы i с одной интенсивностью
        pair=[s for s in sess if s[0]==i and s[1]==10]
        (_,_,h,a1,b1),(_,_,_,a2,b2)=pair
        m=(b1[i]-b2[i])/(a1[i]-a2[i]); sr=-np.log(m)/h; u=sr-Lam[i]
        C=(b1[i]-gam[i])-m*(a1[i]-gam[i]); beta[i]=C*sr/(u*(1-m))
    # (3) переход к уровню освоения и теорема 6
    U={}
    for j,p,h,wa,wb in sess:
        ka=(wa-gam)/beta; kb=(wb-gam)/beta; u=np.zeros(K)
        for i in range(K):
            f=lambda x: step(np.array([ka[i]]),np.array([x]),np.array([Lam[i]]),h)[0]-kb[i]
            u[i]=0.0 if f(0.0)>=-1e-10 else brentq(f,0.0,1e4,xtol=1e-15)
        U.setdefault((j,p),[]).append(u)
    A=np.zeros((K,K)); al=np.zeros(K)
    for j in range(K):
        ua=np.mean(U[(j,4)],0); ub=np.mean(U[(j,10)],0); i=int(np.argmax(ua)); r=10/4
        al[j]=brentq(lambda x:H(r,x)-ub[i]/ua[i],1e-12,500,xtol=1e-15)/4
        A[:,j]=ua/(1-np.exp(-al[j]*4))
    return A,Lam,al,beta,gam

worst=0
for K in (2,3,5):
    for _ in range(20):
        A=np.abs(rng.normal(3,1.5,(K,K)))*(rng.random((K,K))<0.7); np.fill_diagonal(A,np.abs(rng.normal(4,1,K)))
        Lam=rng.uniform(0.02,0.3,K); al=rng.uniform(0.05,0.3,K); k0=rng.uniform(0.3,0.7,K)
        beta=rng.uniform(2,8,K); gam=rng.uniform(-4,1,K)
        A_,L_,a_,b_,g_=reconstruct(*design(A,Lam,al,k0,beta,gam),K)
        e=max(np.abs(A_-A).max()/A.max(),np.abs(L_-Lam).max()/Lam.max(),np.abs(a_-al).max()/al.max(),
              np.abs(b_-beta).max()/beta.max(),np.abs(g_-gam).max()/np.abs(gam).max())
        worst=max(worst,e)
print(f"восстановление (A, Lambda, alpha, beta, gamma) по логитам, K=2,3,5: макс. отн. ошибка {worst:.1e}")

# Два замера на паузе недостаточно: семейство (Lambda', gamma') точно воспроизводит оба
L,g,b,k0,tau=0.1,-1.0,5.0,0.6,18.0
w0=b*k0+g; w2=b*k0*np.exp(-L*tau)+g
for g2 in (-3.0,-1.5,-1.0,-0.6):
    L2=-np.log((w2-g2)/(w0-g2))/tau
    print(f"  gamma'={g2:+.1f}, Lambda'={L2:.4f}: воспроизводит оба замера паузы точно; истинное Lambda={L}")
