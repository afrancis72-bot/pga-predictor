"""Normalize RickRunGood CSV exports without trusting model scores or weights."""
from __future__ import annotations
import re
import pandas as pd

EXCLUDED={"rank","score","min%","max%","country"}
KNOWN={
"SG Off-the-Tee L36":"sg_ott","SG Approach L36":"sg_approach",
"SG Around-the-Green L36":"sg_arg","SG Putting L36":"sg_putting",
"SG Tee-to-Green L36":"sg_tee_to_green",
"Greens in Regulation % L36":"gir_pct","Scrambling % L36":"scrambling_pct",
"SG Bentgrass L36":"rrg_sg_bentgrass_l36","SG Bermuda L36":"rrg_sg_bermuda_l36",
"SG Paspalum L36":"rrg_sg_paspalum_l36","SG Poa Annua L36":"rrg_sg_poa_annua_l36",
}
def _key(x):
    return re.sub(r"\s+"," ",str(x).strip()).casefold()

def normalize_rrg_exports(exports):
    if not exports: raise ValueError("Upload RickRunGood CSV exports")
    merged=None
    for n,raw in enumerate(exports,1):
        d=raw.copy()
        d.columns=[str(c).replace("\ufeff","").strip() for c in d.columns]
        if not {"PlayerId","Player","Salary"}.issubset(d):
            raise ValueError(f"RRG export {n} missing PlayerId/Player/Salary")
        d["PlayerId"]=pd.to_numeric(d.PlayerId,errors="raise").astype(int)
        d["Player"]=d.Player.astype(str).str.strip()
        d["Salary"]=pd.to_numeric(d.Salary,errors="raise").astype(int)
        if d.PlayerId.duplicated().any(): raise ValueError("Duplicate RRG PlayerId")
        cols=[c for c in d if c not in {"PlayerId","Player","Salary"} and c.casefold() not in EXCLUDED]
        for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce")
        d=d[["PlayerId","Player","Salary"]+cols]
        if merged is None: merged=d;continue
        j=merged.merge(d,on="PlayerId",how="outer",suffixes=("","__new"),validate="one_to_one")
        for c in ["Player","Salary"]+cols:
            new=c+"__new"
            if new not in j: continue
            if c in j:
                a,b=j[c],j[new]
                conflict=a.notna()&b.notna()&(a!=b)
                if c=="Player": conflict=a.notna()&b.notna()&(a.map(_key)!=b.map(_key))
                if conflict.any(): raise ValueError(f"Conflicting RRG exports: {c}")
                j[c]=a.combine_first(b)
            else: j[c]=j[new]
            j=j.drop(columns=new)
        merged=j
    merged=merged.rename(columns={"PlayerId":"rrg_player_id","Player":"player","Salary":"salary",**KNOWN})
    rename={c:"rrg_"+re.sub(r"[^a-z0-9]+","_",c.lower()).strip("_") for c in merged
            if c not in {"rrg_player_id","player","salary"} and c not in KNOWN.values()}
    merged=merged.rename(columns=rename)
    if merged.player.map(_key).duplicated().any(): raise ValueError("Duplicate RRG golfer names")
    metrics=[c for c in merged if c not in {"rrg_player_id","player","salary"}]
    audit={"golfers":len(merged),"exports":len(exports),"metrics":metrics,
           "missing_by_metric":{c:int(merged[c].isna().sum()) for c in metrics},
           "excluded_model_columns":sorted(EXCLUDED)}
    return merged.reset_index(drop=True),audit

def align_rrg_to_field(field,rrg,min_coverage=.70):
    if not {"player","salary"}.issubset(field): raise ValueError("Field needs player,salary")
    a=field[["player","salary"]].copy();a["_key"]=a.player.map(_key)
    b=rrg.copy();b["_key"]=b.player.map(_key)
    if a._key.duplicated().any() or b._key.duplicated().any():
        raise ValueError("Ambiguous golfer names")
    b=b.rename(columns={"player":"rrg_source_name","salary":"rrg_source_salary"})
    j=a.merge(b,on="_key",how="left",validate="one_to_one")
    matched=j.rrg_player_id.notna()
    bad=matched&(pd.to_numeric(j.salary)!=pd.to_numeric(j.rrg_source_salary))
    if bad.any(): raise ValueError("RRG/DK salary mismatch: "+", ".join(j.loc[bad,"player"].tolist()[:5]))
    coverage=float(matched.mean()) if len(a) else 0.
    if coverage<min_coverage: raise ValueError(f"RRG/DK coverage {coverage:.1%} below {min_coverage:.0%}")
    metrics=[c for c in rrg if c not in {"player","salary","rrg_player_id"}]
    return j[["player"]+metrics],{"matched":int(matched.sum()),"field_size":len(a),"coverage":coverage}
