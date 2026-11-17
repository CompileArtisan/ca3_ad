"""Streamlit entry point: thin; all model logic lives in ca3model/, all UI in ui/."""
import streamlit as st

st.set_page_config(page_title="RECALL4 NEUROLAB", page_icon=":material/neurology:", layout="wide",
                   initial_sidebar_state="collapsed")

from ui.neurolab import render_neurolab  # noqa: E402
from ui.settings import render_settings  # noqa: E402
from ui.style import CSS, HERO  # noqa: E402
from ui.tabs import (render_about, render_combined, render_demo, render_learning,  # noqa: E402
                     render_load, render_network, render_summary, render_sweeps)
from ui.text import DISCLAIMER  # noqa: E402

st.session_state["_chart_n"] = 0
st.html(CSS)
st.html(HERO)

sim_tab, adv_tab = st.tabs([":material/neurology: Simulator", ":material/tune: Advanced"])

# Advanced settings are rendered first (they feed the simulator), into the second tab.
with adv_tab:
    st.markdown("#### Model settings")
    settings = render_settings()
    st.markdown("#### Detailed analysis")
    names = ["Memory demo", "Network connectivity", "Damage sweeps", "Combined damage",
             "New learning", "Memory load", "Summary & export", "About & limitations"]
    tabs = st.tabs(names)
    for tab, fn in zip(tabs, [render_demo, render_network, render_sweeps, render_combined,
                              render_learning, render_load, render_summary]):
        with tab:
            fn(settings)
    with tabs[7]:
        render_about()

with sim_tab:
    render_neurolab(settings)
    st.info(DISCLAIMER, icon=":material/info:")
