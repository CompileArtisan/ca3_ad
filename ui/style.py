"""Page-level styling: header and CSS overrides on top of the dark theme in .streamlit/config.toml."""

HERO = """
<div class="nl-hero">
  <div class="nl-eyebrow">CA3 attractor-network model</div>
  <h1>Neuro<span>lab</span></h1>
  <p>Show a healthy brain and an Alzheimer's-inspired brain the same scrap of a memory, and watch
  whether each one can complete it.</p>
</div>
"""

CSS = """
<style>
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1400px}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebarCollapsedControl"]{display:none}
.nl-hero{text-align:center;margin:0 auto 1.2rem;max-width:760px}
.nl-hero h1{font-size:3rem;font-weight:700;letter-spacing:-.03em;margin:.1rem 0 .3rem;padding:0;
  color:#ece9ff}
.nl-hero h1 span{background:linear-gradient(90deg,#a78bfa,#ffc94a);-webkit-background-clip:text;
  background-clip:text;color:transparent}
.nl-hero p{color:#9a94c7;font-size:1.02rem;margin:0}
.nl-eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#a78bfa}
.nl-label{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#9a94c7;
  margin:.2rem 0 .5rem}
.st-key-nl_controls{background:linear-gradient(180deg,#191536,#141129);border-color:#2a2650!important;
  padding:1.1rem 1.4rem .6rem!important;margin-bottom:.8rem}
.stTabs [data-baseweb="tab-list"]{gap:.4rem}
.stTabs [data-baseweb="tab"]{height:2.6rem;padding:0 1rem;border-radius:.6rem .6rem 0 0}
.nl-rel{background:#15122b;border:1px solid #2a2650;border-radius:14px;padding:.7rem 1rem;
  display:grid;grid-template-columns:1fr auto;gap:.45rem;align-items:center}
.nl-rel span{color:#c9c4ef}.nl-rel b{font-variant-numeric:tabular-nums}
.nl-track{grid-column:1/-1;height:6px;border-radius:99px;background:#241f45;overflow:hidden}
.nl-track i{display:block;height:100%;border-radius:99px}
.nl-healthy .nl-track i{background:#2dd4bf}.nl-damaged .nl-track i{background:#fb7185}
iframe{border-radius:16px}
</style>
"""
