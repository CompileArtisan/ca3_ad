"""Page-level styling: header and CSS overrides on top of the dark theme in .streamlit/config.toml."""

HERO = """
<div class="nl-hero">
  <h1>RECALL4 <span>NEUROLAB</span></h1>
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
.nl-track i{background:var(--acc,#a78bfa)}
.nl-rel small{grid-column:1/-1;color:#9a94c7;font-size:.8rem}
.nl-scs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem;margin-bottom:.4rem}
@media (max-width:900px){.nl-scs{grid-template-columns:repeat(2,minmax(0,1fr))}}
.nl-sc{background:#15122b;border:1px solid #2a2650;border-radius:14px;padding:.6rem .8rem;
  display:grid;grid-template-columns:1fr auto;gap:.3rem .5rem;align-items:baseline}
.nl-sc.on{border-color:var(--acc,#a78bfa);box-shadow:0 0 0 1px var(--acc,#a78bfa) inset}
.nl-sc span{color:#c9c4ef;grid-column:1/-1;font-size:.85rem}
.nl-sc b{font-size:1.5rem;font-variant-numeric:tabular-nums;color:#ece9ff}
.nl-sc em{font-style:normal;color:#34d399;font-size:.85rem;font-variant-numeric:tabular-nums}
.nl-sc small{grid-column:1/-1;color:#9a94c7;font-size:.72rem}
.nl-sc .nl-track{height:5px}
iframe{border-radius:16px}
</style>
"""
