import json, numpy as np, pandas as pd, pingouin as pg
df=pd.read_csv("chaticu_long_clean.csv"); df["question"]=df["question"].astype(str)
R=json.load(open("results.json"))
LIKERT=["Accuracy","Relevance","Clarity","Trust","Comparison","Confidence"]

# ---- ICC with correct column name 'CI95' ----
icc={}
for dom in LIKERT+["Safety"]:
    sub=df[df.domain==dom]
    long=sub.assign(target=sub.question+"|"+sub.model)[["target","expert","score"]]
    ic=pg.intraclass_corr(data=long,targets="target",raters="expert",ratings="score",nan_policy="omit").set_index("Type")
    def row(t):
        r=ic.loc[t]; ci=r["CI95"]
        return {"ICC":round(float(r["ICC"]),3),"ci":[round(float(ci[0]),3),round(float(ci[1]),3)]}
    icc[dom]={"ICC2_1":row("ICC2"),"ICC2_k":row("ICC2k")}
R["icc"]=icc

# ---- Gwet AC1 (unweighted) & AC2 (quadratic weights), Gwet (2014) ----
def gwet(matrix, weighted=False):
    cats=sorted({v for r in matrix for v in r if not (isinstance(v,float) and np.isnan(v))})
    q=len(cats); idx={c:i for i,c in enumerate(cats)}
    if q<2: return np.nan
    W=np.eye(q) if not weighted else np.array(
        [[1-((i-j)**2)/((q-1)**2) for j in range(q)] for i in range(q)])
    Tw=W.sum()
    pa_num=0.0; nprime=0; pik=np.zeros(q)
    for r in matrix:
        vals=[v for v in r if not (isinstance(v,float) and np.isnan(v))]
        ni=len(vals)
        if ni<2: continue
        nprime+=1
        c=np.zeros(q)
        for v in vals: c[idx[v]]+=1
        agree=sum(W[k,l]*c[k]*(c[l]-(1 if k==l else 0)) for k in range(q) for l in range(q))
        pa_num+=agree/(ni*(ni-1))
        pik+=c/ni
    pa=pa_num/nprime
    pik=pik/nprime
    pe=(Tw/(q*(q-1)))*sum(pik[k]*(1-pik[k]) for k in range(q))
    return (pa-pe)/(1-pe)

# unit test
assert abs(gwet([[1,1,1],[2,2,2],[1,1,1]])-1.0)<1e-9, "perfect agreement should be 1"
g={}
for dom in LIKERT+["Safety"]:
    sub=df[df.domain==dom]
    wide=sub.assign(target=sub.question+"|"+sub.model).pivot_table(
        index="target",columns="expert",values="score",aggfunc="first")
    mat=wide.values.tolist()
    if dom=="Safety":
        g[dom]={"AC1":round(float(gwet(mat,False)),3)}
    else:
        g[dom]={"AC1":round(float(gwet(mat,False)),3),"AC2_quadratic":round(float(gwet(mat,True)),3)}
R["gwet_ac"]=g

json.dump(R,open("results.json","w"),ensure_ascii=False,indent=1)
print("ICC:");  [print(" ",k,v["ICC2_k"]) for k,v in icc.items()]
print("GWET:"); [print(" ",k,v) for k,v in g.items()]
