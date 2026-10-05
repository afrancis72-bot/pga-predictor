from __future__ import annotations
import pandas as pd
import streamlit as st
from showdown_model import ShowdownSettings, APPROACH_BUCKETS, build_showdown_projections
from showdown_optimizer import ShowdownPortfolioSettings, optimize_showdown

HIST_COLS=["player","birdies_or_better_gained","bogey_avoidance_gained"]+[f"sg_app_{b}" for b in APPROACH_BUCKETS]
for r in range(1,5): HIST_COLS += [f"r{r}_sg_total",f"r{r}_rounds"]
CURRENT_COLS=["player","week_sg_app","week_sg_ott","week_sg_arg","week_sg_putt","tee_time_weather_edge","start_position"]

def render_showdown(base:pd.DataFrame):
    st.header("PGA Showdown V1.0")
    st.caption("Round-specific Monte Carlo: historical round affinity, Birdies-or-Better, bogey avoidance, course-weighted approach buckets, live underlying play, and Round-4 finishing position.")
    round_no=st.sidebar.selectbox("Showdown round",[1,2,3,4],index=0)
    sims=st.sidebar.selectbox("Showdown simulations",[25000,50000,100000],index=0, help="25K is the fast working default. Use 100K for final lineup generation once inputs are set.")
    seed=st.sidebar.number_input("Showdown seed",1,999999,42)
    st.sidebar.subheader("Showdown Data")
    hist=st.sidebar.file_uploader("Historical round / scoring / approach CSV",type="csv",key="sd_hist")
    current=st.sidebar.file_uploader("Current-week round data (R2-R4)",type="csv",key="sd_current",disabled=round_no==1)
    template=pd.DataFrame({"player":base.player.astype(str)})
    for c in HIST_COLS[1:]: template[c]=""
    st.sidebar.download_button("Download historical template",template.to_csv(index=False),"pga_showdown_history_template.csv","text/csv")
    ctemplate=pd.DataFrame({"player":base.player.astype(str)})
    for c in CURRENT_COLS[1:]: ctemplate[c]=""
    st.sidebar.download_button("Download current-week template",ctemplate.to_csv(index=False),"pga_showdown_current_week_template.csv","text/csv")

    hdf=pd.read_csv(hist) if hist is not None else pd.DataFrame()
    cdf=pd.read_csv(current) if current is not None else pd.DataFrame()
    # Keep the base table clean. Historical/current files are merged exactly once
    # inside the model so round-affinity columns cannot be duplicated or shadowed.
    enriched=base.copy()
    st.sidebar.subheader("Course approach mix")
    st.sidebar.caption("Enter expected share of approach shots by distance. Values are normalized automatically.")
    defaults={"lt100":5,"100_125":10,"125_150":20,"150_175":25,"175_200":25,"200_plus":15}
    mix={b:st.sidebar.number_input(b.replace("_","–")+" %",0,100,defaults[b],1,key=f"mix_{b}") for b in APPROACH_BUCKETS}
    if sum(mix.values())==0: st.error("Course approach mix cannot total zero."); return

    missing_core=[c for c in ["birdies_or_better_gained","bogey_avoidance_gained"] if c not in enriched]
    bucket_coverage=sum(c in enriched for c in [f"sg_app_{b}" for b in APPROACH_BUCKETS])
    if missing_core or bucket_coverage==0:
        st.warning("Showdown V1.0 can run, but full scoring/course-fit power requires the historical template. Missing inputs shrink to neutral; they are never invented.")
    if round_no>1 and cdf.empty:
        st.warning("R2-R4 current-week file is not loaded. Live ball-striking/regression information will be neutral.")

    settings=ShowdownSettings(round_number=int(round_no),sims=int(sims),seed=int(seed))
    with st.spinner(f"Running {sims:,} Round {round_no} simulations..."):
        proj=build_showdown_projections(enriched,hdf,cdf,mix,settings)
    st.session_state["showdown_proj"]=proj
    tabs=st.tabs(["Round Projections","Showdown Optimizer","Diagnostics"])
    with tabs[0]:
        cols=["player","salary","showdown_mean","showdown_p90","showdown_p95","round_birdies_mean","round_bogeys_mean","round_score_mean","round_affinity_z","course_approach_z","course_approach_coverage","scoring_style_z","current_week_z"]
        st.dataframe(proj[[c for c in cols if c in proj]].sort_values("showdown_p90",ascending=False),use_container_width=True,hide_index=True)
        st.download_button("Download Showdown projections",proj.to_csv(index=False),f"pga_showdown_r{round_no}_projections.csv","text/csv")
    with tabs[1]:
        a,b,c,d=st.columns(4)
        n=a.number_input("Lineups",1,150,20,key="sd_n")
        exp=b.slider("Max exposure",.10,1.0,.50,.05,key="sd_exp")
        uniq=c.number_input("Minimum unique golfers",1,5,2,key="sd_uniq")
        floor=d.number_input("Minimum lineup salary",40000,50000,45000,100,key="sd_floor")
        names=sorted(proj.player.dropna().astype(str).unique())
        locks=st.multiselect("Lock golfers",names,key="sd_locks"); exc=st.multiselect("Exclude golfers",[x for x in names if x not in locks],key="sd_exc")
        if st.button("Generate Showdown portfolio",type="primary"):
            ps=ShowdownPortfolioSettings(int(n),50000,int(floor),6,float(exp),int(uniq),int(seed))
            try:
                port,summ,expo=optimize_showdown(proj,ps,locks,exc)
                st.success(f"Generated {len(summ)} Round {round_no} lineups")
                st.dataframe(summ,use_container_width=True,hide_index=True); st.subheader("Exposure"); st.dataframe(expo,use_container_width=True,hide_index=True)
                st.download_button("Download Showdown lineups",port.to_csv(index=False),f"pga_showdown_r{round_no}_lineups.csv","text/csv")
            except Exception as e: st.error(str(e))
    with tabs[2]:
        st.write({"round":round_no,"simulations":sims,"historical_file":hist is not None,"current_week_file":current is not None,"approach_mix_total":sum(mix.values())})
        st.caption("Round affinity is sample-size shrunk with a 20-round prior. Missing specialty stats are neutral. Current-week putting is deliberately weighted below approach/OTT to reduce score-chasing. R4 adds simulated finishing-position points.")
