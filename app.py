"""Streamlit entry point: thin; all model logic lives in ca3model/, all UI in ui/."""
import streamlit as st

st.set_page_config(page_title="CA3-inspired Hopfield model of memory deterioration", layout="wide")

from ui.sidebar import render_sidebar  # noqa: E402
from ui.tabs import (render_about, render_combined, render_demo, render_learning,  # noqa: E402
                     render_load, render_network, render_summary, render_sweeps)
from ui.text import DISCLAIMER  # noqa: E402

st.session_state["_chart_n"] = 0
st.title("CA3-inspired Hopfield model of memory deterioration")
st.info(DISCLAIMER)

settings = render_sidebar()
names = ["Memory demo", "Network connectivity", "Damage sweeps", "Combined damage",
         "New learning", "Memory load", "Summary & export", "About & limitations"]
tabs = st.tabs(names)
with tabs[0]:
    render_demo(settings)
with tabs[1]:
    render_network(settings)
with tabs[2]:
    render_sweeps(settings)
with tabs[3]:
    render_combined(settings)
with tabs[4]:
    render_learning(settings)
with tabs[5]:
    render_load(settings)
with tabs[6]:
    render_summary(settings)
with tabs[7]:
    render_about()
