import numpy as np
from scipy.optimize import linprog
def lp_day(P,power,energy,eff=np.sqrt(0.9),soc0=0.5):
    T=len(P)
    cost=np.concatenate([P,-P])
    A=[];b=[]
    for t in range(T):
        row=np.zeros(2*T); row[:t+1]=eff; row[T:T+t+1]=-1/eff
        A.append(row); b.append(energy-soc0*energy)
        A.append(-row); b.append(soc0*energy)
    row=np.concatenate([np.full(T,-eff),np.full(T,1/eff)])
    A.append(row); b.append(0.0)
    res=linprog(cost,A_ub=np.array(A),b_ub=b,bounds=[(0,power)]*(2*T),method='highs')
    return -res.fun, res.x[T:].sum()
for seed in [43,47]:
    base=np.load(f'outputs/bess_sweep/full/baseline_s{seed}/dispatch.npz')['market_prices']
    print('seed',seed,'baseline price day0',np.round(base[0],0))
    for p in [5,25,75]:
        for dur in [1,2,4,8]:
            E=p*dur
            lp=[lp_day(base[i],p,E) for i in range(20)]
            rev=np.mean([x[0] for x in lp]); dis=np.mean([x[1] for x in lp])
            d=np.load(f'outputs/bess_sweep/full/p{p}_d{dur}_s{seed}/dispatch.npz')
            P=d['market_prices']; learned=(P*(d['bess_discharge']-d['bess_charge'])).sum(1).mean(); ld=d['bess_discharge'].sum(1).mean()
            pen=np.maximum(0,0.5*E-d['soc'][:,-1]).mean()*80
            print(f"p{p:2d}_d{dur} LP(price-taker): rev={rev:7.0f} dis={dis:6.1f} cyc={dis/E:.2f} | learned: rev={learned:7.0f} dis={ld:6.1f} cyc={ld/E:.2f} pen={pen:5.0f}")
