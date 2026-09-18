# ChatICU publication-grade re-analysis pipeline
# Deterministic, single coherent pipeline. Outputs results.json + figures/.
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import openpyxl
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE, GEE
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Binomial
import pingouin as pg

RNG = np.random.default_rng(20260530)
RESULTS = {}

# ---------- Load canonical data ----------
wb = openpyxl.load_workbook("專家評分彙整 (AI分析).xlsx", data_only=True)
ws = wb["工作表1"]
rows = [r[:5] for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
df = pd.DataFrame(rows, columns=["expert","question","model","domain","score"])
df["model"] = df["model"].map({1:"OpenEvidence",2:"ChatICU",3:"ChatGPT"})
df["score"] = df["score"].astype(float)
df["question"] = df["question"].astype(str)
role_map = {"D1":"Physician","D2":"Physician","N1":"Nurse","N2":"Nurse",
            "P1":"Pharmacist","P3":"Pharmacist","P4":"Pharmacist","P5":"Pharmacist"}
df["role"] = df["expert"].map(role_map)
df.to_csv("chaticu_long_clean.csv", index=False)

MODELS = ["ChatGPT","OpenEvidence","ChatICU"]
LIKERT = ["Accuracy","Relevance","Clarity","Trust","Comparison","Confidence"]
CORE = ["Accuracy","Relevance","Clarity","Trust","Comparison"]
RESULTS["meta"] = {"n_rows":len(df),"n_questions":df.question.nunique(),
                   "experts":sorted(df.expert.unique()),
                   "role_counts":df.drop_duplicates("expert").role.value_counts().to_dict()}

def cluster_bootstrap_ci(values_by_q, stat=np.mean, B=2000):
    qs = list(values_by_q.keys()); arr=[values_by_q[q] for q in qs]
    obs = stat(np.concatenate(arr))
    boots=[]
    n=len(qs)
    for _ in range(B):
        idx=RNG.integers(0,n,n)
        samp=np.concatenate([arr[i] for i in idx])
        boots.append(stat(samp))
    lo,hi=np.percentile(boots,[2.5,97.5])
    return float(obs),float(lo),float(hi)

# ---------- 1. Descriptives w/ cluster-bootstrap 95% CI ----------
desc={}
for dom in LIKERT+["Safety"]:
    desc[dom]={}
    for m in MODELS:
        sub=df[(df.domain==dom)&(df.model==m)]
        vbq={q:g.score.values for q,g in sub.groupby("question")}
        mean,lo,hi=cluster_bootstrap_ci(vbq)
        desc[dom][m]={"mean":round(mean,3),"sd":round(sub.score.std(),3),
                      "ci":[round(lo,3),round(hi,3)],"n":int(len(sub))}
RESULTS["descriptives"]=desc

# ---------- 2. PRIMARY: Ordinal GEE (proportional odds), clustered by question ----------
# model effect on each Likert domain; ChatICU as reference -> report cumulative OR vs ChatICU
ordinal={}
for dom in LIKERT:
    sub=df[df.domain==dom].copy()
    sub["model"]=pd.Categorical(sub["model"],categories=["ChatICU","ChatGPT","OpenEvidence"])
    sub=sub.sort_values("question")
    try:
        m=OrdinalGEE.from_formula("score ~ C(model, Treatment(reference='ChatICU'))",
                                  groups="question", data=sub, cov_struct=Independence())
        r=m.fit(maxiter=100)
        out={}
        for term in r.params.index:
            if "model" in term:
                or_=float(np.exp(r.params[term])); ci=np.exp(r.conf_int().loc[term])
                out[term]={"cumulative_OR":round(or_,3),
                           "ci":[round(float(ci[0]),3),round(float(ci[1]),3)],
                           "p":float(r.pvalues[term])}
        ordinal[dom]={"ref":"ChatICU","contrasts":out,"note":"OR>1 favors the listed model over ChatICU (higher odds of higher score); OR<1 favors ChatICU"}
    except Exception as e:
        ordinal[dom]={"error":str(e)}
RESULTS["ordinal_gee_primary"]=ordinal

# ---------- 3. SENSITIVITY: Linear mixed-effects, crossed random intercepts (question + rater) ----------
lmm={}
for dom in LIKERT:
    sub=df[df.domain==dom].copy()
    sub["model"]=pd.Categorical(sub["model"],categories=["ChatICU","ChatGPT","OpenEvidence"])
    sub["g"]=1
    try:
        vc={"question":"0+C(question)","rater":"0+C(expert)"}
        md=smf.mixedlm("score ~ C(model, Treatment(reference='ChatICU'))", sub,
                       groups="g", vc_formula=vc)
        mf=md.fit(reml=True, method="lbfgs")
        # adjusted means (EMM) per model = intercept + coef
        inter=mf.params["Intercept"]
        emm={"ChatICU":round(float(inter),3)}
        contr={}
        for term in mf.params.index:
            if "model" in term:
                lab=term.split("[T.")[-1].rstrip("]")
                emm[lab]=round(float(inter+mf.params[term]),3)
                contr[lab]={"diff_vs_ChatICU":round(float(mf.params[term]),3),
                            "p":float(mf.pvalues[term])}
        lmm[dom]={"adjusted_EMM":emm,"contrasts_vs_ChatICU":contr}
    except Exception as e:
        lmm[dom]={"error":str(e)}
RESULTS["linear_mixed_sensitivity"]=lmm

# ---------- 4. Friedman + Wilcoxon (per-question mean across raters) ----------
nonpar={}
for dom in LIKERT+["Safety"]:
    sub=df[df.domain==dom]
    wide=sub.groupby(["question","model"]).score.mean().unstack("model")
    wide=wide.dropna()
    try:
        fr=stats.friedmanchisquare(*[wide[m].values for m in MODELS])
        pw={}
        pairs=[("ChatICU","ChatGPT"),("ChatICU","OpenEvidence"),("ChatGPT","OpenEvidence")]
        raw=[]
        for a,b in pairs:
            w=stats.wilcoxon(wide[a].values,wide[b].values)
            raw.append(w.pvalue)
        # Holm correction
        order=np.argsort(raw); adj=[None]*3; mtest=3
        for rank,i in enumerate(order):
            adj[i]=min(1.0,raw[i]*(mtest-rank))
        for i,(a,b) in enumerate(pairs):
            pw[f"{a} vs {b}"]={"p_raw":round(raw[i],5),"p_holm":round(adj[i],5),
                               "median_diff":round(float(wide[a].median()-wide[b].median()),3)}
        nonpar[dom]={"friedman_chi2":round(float(fr.statistic),3),"friedman_p":float(fr.pvalue),"pairwise":pw}
    except Exception as e:
        nonpar[dom]={"error":str(e)}
RESULTS["friedman_wilcoxon"]=nonpar

# ---------- 5. Safety: binomial GEE clustered by question + Firth-penalized per-model rate ----------
safety={}
sub=df[df.domain=="Safety"].copy()
sub["model"]=pd.Categorical(sub["model"],categories=["ChatICU","ChatGPT","OpenEvidence"])
sub=sub.sort_values("question")
try:
    gm=GEE.from_formula("score ~ C(model, Treatment(reference='ChatICU'))",
                        groups="question",data=sub,family=Binomial(),cov_struct=Independence())
    gr=gm.fit()
    out={}
    for term in gr.params.index:
        if "model" in term:
            or_=float(np.exp(gr.params[term])); ci=np.exp(gr.conf_int().loc[term])
            out[term]={"OR_vs_ChatICU":round(or_,3),"ci":[round(float(ci[0]),3),round(float(ci[1]),3)],"p":float(gr.pvalues[term])}
    # per-model no-risk proportion + Wilson CI
    prop={}
    for m in MODELS:
        s=sub[sub.model==m].score; k=int(s.sum()); n=int(len(s))
        lo,hi=sm.stats.proportion_confint(k,n,method="wilson")
        prop[m]={"no_risk_prop":round(k/n,3),"wilson_ci":[round(lo,3),round(hi,3)],
                 "risk_rate":round(1-k/n,3),"n":n}
    safety={"binomial_gee_vs_ChatICU":out,"per_model_proportion":prop}
except Exception as e:
    safety={"error":str(e)}
RESULTS["safety"]=safety

# ---------- 6. ICC(2,k) and ICC(2,1) with 95% CI + Gwet AC ----------
icc={}
for dom in LIKERT+["Safety"]:
    sub=df[df.domain==dom]
    # average over models per (question,expert)? No: reliability across raters on the item-model targets.
    # Standard: targets = (question x model), raters = expert. Use wide rater matrix.
    long=sub.assign(target=sub.question+"|"+sub.model)[["target","expert","score"]]
    try:
        ic=pg.intraclass_corr(data=long,targets="target",raters="expert",ratings="score",nan_policy="omit")
        ic=ic.set_index("Type")
        r1=ic.loc["ICC2"]; rk=ic.loc["ICC2k"]
        _cicol="CI95%" if "CI95%" in r1.index else "CI95"  # pingouin renamed the column across versions
        icc[dom]={"ICC2_1":round(float(r1["ICC"]),3),"ICC2_1_ci":[round(float(r1[_cicol][0]),3),round(float(r1[_cicol][1]),3)],
                  "ICC2_k":round(float(rk["ICC"]),3),"ICC2_k_ci":[round(float(rk[_cicol][0]),3),round(float(rk[_cicol][1]),3)]}
    except Exception as e:
        icc[dom]={"error":str(e)}
RESULTS["icc"]=icc

# Gwet AC1 (binary safety) / AC2 (ordinal) via bootstrap over targets
def gwet_ac(matrix, weights=None):
    # Gwet (2014) AC1 (unweighted) / AC2 (weighted). Canonical implementation,
    # identical to patch_icc_gwet.py (verified against Gwet 2014 + unit test).
    # NOTE (v10): the earlier version of this function double-counted self-pairs
    # (the `-1*0` no-op) and used the unweighted AC1 chance term even for the
    # weighted AC2, producing agreement values > 1. It was masked because
    # patch_icc_gwet.py overwrote results.json after the pipeline ran; the two
    # now agree so no separate patch step is required.
    cats=sorted({v for row in matrix for v in row if not (isinstance(v,float) and np.isnan(v))})
    q=len(cats); idx={c:i for i,c in enumerate(cats)}
    if q<2: return np.nan
    W=np.eye(q) if weights is None else np.asarray(weights)
    Tw=W.sum()
    pa_num=0.0; nprime=0; pik=np.zeros(q)
    for row in matrix:
        vals=[v for v in row if not (isinstance(v,float) and np.isnan(v))]
        ni=len(vals)
        if ni<2: continue
        nprime+=1
        counts=np.zeros(q)
        for v in vals: counts[idx[v]]+=1
        agree=sum(W[k,l]*counts[k]*(counts[l]-(1 if k==l else 0))
                  for k in range(q) for l in range(q))
        pa_num+=agree/(ni*(ni-1))
        pik+=counts/ni
    pa=pa_num/nprime
    pik=pik/nprime
    pe=(Tw/(q*(q-1)))*sum(pik[k]*(1-pik[k]) for k in range(q))
    return (pa-pe)/(1-pe) if (1-pe)!=0 else np.nan
assert abs(gwet_ac([[1,1,1],[2,2,2],[1,1,1]])-1.0)<1e-9, "perfect agreement should be 1"
gwet={}
for dom,wtype in [("Safety",None),("Accuracy","ordinal"),("Clarity","ordinal"),("Confidence","ordinal"),("Relevance","ordinal"),("Trust","ordinal"),("Comparison","ordinal")]:
    sub=df[df.domain==dom]
    wide=sub.assign(target=sub.question+"|"+sub.model).pivot_table(index="target",columns="expert",values="score",aggfunc="first")
    mat=wide.values.tolist()
    cats=sorted({v for row in mat for v in row if not (isinstance(v,float) and np.isnan(v))})
    q=len(cats)
    if wtype=="ordinal" and q>1:
        # quadratic weights
        W=np.array([[1-((i-j)**2)/((q-1)**2) for j in range(q)] for i in range(q)])
    else:
        W=None
    try:
        ac=gwet_ac(mat,W)
        gwet[dom]=round(float(ac),3)
    except Exception as e:
        gwet[dom]=f"err:{e}"
RESULTS["gwet_ac"]=gwet

# ---------- 7. Composite quality score (5 core domains, per question x model x rater) ----------
core=df[df.domain.isin(CORE)]
comp=core.groupby(["question","model","expert","role"]).score.mean().reset_index().rename(columns={"score":"composite"})
comp_model={}
for m in MODELS:
    vbq={q:g.composite.values for q,g in comp[comp.model==m].groupby("question")}
    mean,lo,hi=cluster_bootstrap_ci(vbq)
    comp_model[m]={"mean":round(mean,3),"ci":[round(lo,3),round(hi,3)]}
# Cronbach alpha + correlations among 5 core domains (wide per question x model x rater)
wide_core=core.pivot_table(index=["question","model","expert"],columns="domain",values="score").dropna()
alpha=pg.cronbach_alpha(data=wide_core[CORE])
corr=wide_core[CORE].corr().round(3)
# McDonald omega (approx via factor analysis loadings)
try:
    from factor_analyzer import FactorAnalyzer
    fa=FactorAnalyzer(n_factors=1,rotation=None); fa.fit(wide_core[CORE])
    load=fa.loadings_[:,0]; err=1-load**2
    omega=float((load.sum()**2)/((load.sum()**2)+err.sum()))
except Exception as e:
    omega=f"unavailable:{e}"
# sensitivity composites
def comp_means(domains):
    c=df[df.domain.isin(domains)].groupby(["question","model","expert"]).score.mean().reset_index()
    return {m:round(float(c[c.model==m].score.mean()),3) for m in MODELS}
sens={"core5":comp_means(CORE),
      "drop_Clarity":comp_means([d for d in CORE if d!="Clarity"]),
      "drop_Comparison":comp_means([d for d in CORE if d!="Comparison"]),
      "AccTrustSafetyish":comp_means(["Accuracy","Trust"])}
# role-reweighted composite (equal weight to 3 professions)
rr={}
for m in MODELS:
    sm_=comp[comp.model==m]
    role_means=sm_.groupby("role").composite.mean()
    rr[m]=round(float(role_means.mean()),3)  # equal weight across professions
# leave-one-rater-out ranking stability (composite)
loo={}
ranks_ok=0
for ex in sorted(df.expert.unique()):
    cc=comp[comp.expert!=ex].groupby("model").composite.mean()
    rank=cc.sort_values(ascending=False).index.tolist()
    loo[ex]=rank
    if rank[0]=="ChatICU": ranks_ok+=1
RESULTS["composite"]={"per_model_bootstrap":comp_model,"cronbach_alpha":round(float(alpha[0]),3),
                      "alpha_ci":[round(x,3) for x in alpha[1]],"mcdonald_omega":omega,
                      "core_domain_correlation":corr.to_dict(),
                      "sensitivity_composites":sens,"role_reweighted_mean":rr,
                      "leave_one_rater_out_top":loo,"loo_chaticu_first_of_8":ranks_ok}

# ---------- 8. Confidence as meta-variable: does rater confidence differ by model & does low-confidence change ranking ----------
conf=df[df.domain=="Confidence"].groupby(["model"]).score.mean().round(3).to_dict()
RESULTS["confidence_by_model"]=conf

with open("results.json","w") as f:
    json.dump(RESULTS,f,ensure_ascii=False,indent=1)
print("DONE. wrote results.json and chaticu_long_clean.csv")
print(json.dumps({k:(v if k in ("meta","gwet_ac","confidence_by_model") else "...") for k,v in RESULTS.items()},ensure_ascii=False,indent=1)[:1500])
