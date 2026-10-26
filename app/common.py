"""Shared, cached loaders, page styling and chart helpers for every dashboard page."""
from __future__ import annotations

import base64
import functools
import os
import random
import sys
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hotelsight import results                      # noqa: E402
from hotelsight.contract import FRIENDLY_NAME, country_names, input_ranges   # noqa: E402
from hotelsight.pipeline import DeployedModel       # noqa: E402

ASSETS = ROOT / "assets"

# Data colours (validated with the dataviz palette validator, light surface). They only ever carry meaning:
BLUE = "#2a78d6"      # "lowers cancellation risk" / likely to stay / primary series
RED = "#e34948"       # "raises cancellation risk" / likely to cancel
GREY = "#c3c2b7"      # de-emphasised series
NAVY = "#0b1d33"      # brand ink / secondary series
GOLD = "#c5a059"      # brand accent only, never used for data
INK = "#1c2533"
MUTED = "#5f6470"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Charts use installed fonts only: Vega measures label widths at render time, before a web font such as
# the page's Inter has loaded, which clips long axis labels and overlaps legend entries.
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"

CSS = """
<style>
:root { --hs-navy:#0b1d33; --hs-navy-2:#14304f; --hs-gold:#c5a059; --hs-gold-ink:#8a6a2b; --hs-gold-soft:#f6eedc;
        --hs-ink:#1c2533; --hs-muted:#5f6470; --hs-line:#e6e0d4; --hs-card:#ffffff;
        --hs-red:#b8322b; --hs-red-soft:#fbeceb; --hs-blue:#1f5fbf; --hs-blue-soft:#e8f0fb;
        --hs-serif:"Playfair Display", Georgia, serif; }
[data-testid="stMainBlockContainer"] { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1320px; }
[data-testid="stHeader"] { background: transparent; }
.hs-ico { font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal; font-size: 20px;
          line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
          direction: ltr; font-feature-settings: "liga"; -webkit-font-smoothing: antialiased; vertical-align: middle; }

/* page header with the skyline artwork */
.hs-hero { position: relative; display: flex; align-items: flex-end; justify-content: space-between; gap: 24px;
           background: radial-gradient(120% 140% at 100% 0%, rgba(197,160,89,.24) 0%, rgba(197,160,89,0) 42%),
                       linear-gradient(118deg, #0b1d33 0%, #122b48 58%, #1a3d63 100%);
           border-radius: 18px; padding: 26px 30px 0; margin-bottom: 6px; overflow: hidden;
           box-shadow: 0 10px 28px rgba(11,29,51,.18), inset 0 -3px 0 #c5a059; }
.hs-hero-text { padding-bottom: 24px; min-width: 0; }
.hs-hero .hs-eyebrow { display: flex; align-items: center; gap: 8px; text-transform: uppercase; letter-spacing: .14em;
                       font-size: 11.5px; font-weight: 600; color: #e2c88f; }
.hs-hero .hs-eyebrow .hs-ico { font-size: 18px; }
.hs-hero h1 { font-family: var(--hs-serif); color: #fff; font-size: 34px; font-weight: 600; margin: 8px 0 8px;
              padding: 0; line-height: 1.15; }
.hs-hero.lg { padding-top: 38px; }
.hs-hero.lg h1 { font-size: 42px; max-width: 760px; }
.hs-hero p { margin: 0; color: rgba(255,255,255,.84); font-size: 15px; max-width: 780px; line-height: 1.6; }
.hs-hero p b { color: #fff; font-weight: 600; }
.hs-chips { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.hs-chip { background: rgba(255,255,255,.07); border: 1px solid rgba(226,200,143,.38); border-radius: 999px;
           padding: 3px 12px; font-size: 12.5px; color: #f3ead6; }
.hs-hero-art { flex: 0 0 auto; line-height: 0; align-self: flex-end; }
.hs-hero-art img { display: block; }
@media (max-width: 1100px) { .hs-hero-art { display: none; } }

/* section titles and form steps */
.hs-section { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 12px; margin: 22px 0 2px;
              padding-bottom: 7px; border-bottom: 1px solid var(--hs-line); }
.hs-section h3 { font-family: var(--hs-serif); font-size: 21px; font-weight: 600; color: var(--hs-navy); margin: 0; padding: 0; }
.hs-section span { font-size: 13px; color: var(--hs-muted); }
.hs-step { display: flex; align-items: center; gap: 10px; font-weight: 650; font-size: 14.5px; color: var(--hs-navy);
           margin: 6px 0 -2px; }
.hs-step .hs-ico { width: 30px; height: 30px; border-radius: 9px; background: var(--hs-gold-soft); color: var(--hs-gold-ink);
                   display: inline-flex; align-items: center; justify-content: center; font-size: 18px; }
.hs-step small { color: var(--hs-muted); font-weight: 500; font-size: 12px; }

/* cards: metrics, forms and keyed containers */
[data-testid="stMetric"] { background: var(--hs-card); border: 1px solid var(--hs-line); border-radius: 14px;
                           padding: 14px 18px; box-shadow: 0 1px 2px rgba(28,37,51,.04); }
[data-testid="stMetricLabel"] p { font-size: 11px; font-weight: 650; color: var(--hs-muted); text-transform: uppercase;
                                  letter-spacing: .05em; }
[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: var(--hs-navy); }
@media (max-width: 1280px) {
  [data-testid="stMetric"] { padding: 10px 12px; }
  [data-testid="stMetricValue"] { font-size: 21px; }
}
[data-testid="stForm"], [class*="st-key-card"] { background: var(--hs-card); border: 1px solid var(--hs-line) !important;
                     border-radius: 16px; box-shadow: 0 1px 3px rgba(28,37,51,.05); }
[class*="st-key-card"] { padding: 18px 20px; }
[class*="st-key-card"] [data-testid="stMetric"] { border: none; box-shadow: none; padding: 0; background: transparent; }
.st-key-card_risk [data-testid="stMetricValue"] { font-size: 46px; line-height: 1.05; }
[data-testid="stExpander"] details { background: var(--hs-card); border-radius: 12px; }
[data-testid="stTabs"] button p { font-weight: 600; }
.hs-card-title { font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                 color: var(--hs-gold-ink); margin-bottom: 8px; }

/* buttons */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(180deg, #1a3d63, #0f2a49); border: 1px solid #0b1d33; color: #fff;
  box-shadow: 0 2px 6px rgba(11,29,51,.22); }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  border-color: var(--hs-gold); color: #fff; background: linear-gradient(180deg, #1f4772, #12304f); }

/* risk result */
.hs-verdict { display: inline-flex; align-items: center; gap: 6px; font-weight: 700; font-size: 14px;
              padding: 5px 14px 5px 10px; border-radius: 999px; }
.hs-verdict .hs-ico { font-size: 18px; }
.hs-verdict.hi { background: var(--hs-red-soft); color: var(--hs-red); }
.hs-verdict.lo { background: var(--hs-blue-soft); color: var(--hs-blue); }
.hs-dial { display: flex; justify-content: center; margin: 4px 0 2px; }

/* reservation summary */
.hs-res-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }
.hs-res-title { font-family: var(--hs-serif); font-size: 22px; color: var(--hs-navy); font-weight: 600; line-height: 1.2; }
.hs-res-sub { font-size: 13px; color: var(--hs-muted); margin-top: 2px; }
.hs-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; font-weight: 700; letter-spacing: .06em;
            text-transform: uppercase; color: var(--hs-gold-ink); background: var(--hs-gold-soft); border-radius: 999px;
            padding: 4px 10px; }
.hs-badge .hs-ico { font-size: 15px; }
.hs-timeline { display: grid; grid-template-columns: auto minmax(40px, 1fr) auto minmax(40px, 1fr) auto; align-items: start;
               gap: 6px; margin: 0 0 18px; }
.hs-tl-node { text-align: center; min-width: 92px; }
.hs-tl-node .dot { width: 36px; height: 36px; border-radius: 50%; margin: 0 auto 6px; display: flex; align-items: center;
                   justify-content: center; background: var(--hs-gold-soft); color: var(--hs-gold-ink); }
.hs-tl-node.main .dot { background: var(--hs-navy); color: #e2c88f; }
.hs-tl-node .lbl { font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em; color: var(--hs-muted); font-weight: 700; }
.hs-tl-node .date { font-size: 15px; font-weight: 700; color: var(--hs-navy); margin-top: 1px; white-space: nowrap; }
.hs-tl-node .day { font-size: 12px; color: var(--hs-muted); }
.hs-tl-line { position: relative; height: 2px; margin-top: 17px;
              background: repeating-linear-gradient(90deg, #d6cbb4 0 6px, transparent 6px 11px); }
.hs-tl-line span { position: absolute; left: 50%; top: -11px; transform: translateX(-50%); font-size: 12px; font-weight: 650;
                   color: var(--hs-gold-ink); background: #fff; padding: 2px 8px; white-space: nowrap; }
.hs-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.hs-fact { background: #faf8f4; border: 1px solid #f0ebe1; border-radius: 12px; padding: 10px 12px; display: flex;
           gap: 10px; align-items: flex-start; }
.hs-fact .hs-ico { color: var(--hs-gold-ink); font-size: 20px; margin-top: 1px; }
.hs-fact > div { font-size: 10.5px; color: var(--hs-muted); font-weight: 700; text-transform: uppercase; letter-spacing: .07em; min-width: 0; }
.hs-fact b { display: block; font-size: 14.5px; color: var(--hs-navy); text-transform: none; letter-spacing: 0; margin-top: 2px;
             font-weight: 650; overflow-wrap: anywhere; }

/* driver lists */
.hs-drivers { list-style: none; padding: 0 !important; margin: 0 0 14px !important; }
.hs-drivers li { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; margin: 0 !important;
                 border-bottom: 1px solid #f0ebe1; font-size: 14px; }
.hs-drivers li b { font-variant-numeric: tabular-nums; }
.hs-up b { color: #b8322b; } .hs-down b { color: #1f5fbf; }

/* home page */
.hs-action { display: flex; flex-direction: column; gap: 8px; }
.hs-action .hs-ico.big { width: 44px; height: 44px; border-radius: 12px; background: var(--hs-navy); color: #e2c88f;
                         display: inline-flex; align-items: center; justify-content: center; font-size: 24px; }
.hs-action h4 { font-family: var(--hs-serif); font-size: 19px; color: var(--hs-navy); margin: 6px 0 0; padding: 0; font-weight: 600; }
.hs-action p { font-size: 14px; color: var(--hs-muted); margin: 0; line-height: 1.55; }
[data-testid="stPageLink"] a { border-radius: 10px; }
[data-testid="stPageLink"] p { font-weight: 650; color: var(--hs-navy); }
.hs-bars { display: flex; flex-direction: column; gap: 9px; }
.hs-bar { display: grid; grid-template-columns: minmax(120px, 1.1fr) 2fr auto; align-items: center; gap: 12px; font-size: 13.5px; }
.hs-bar .track { height: 10px; border-radius: 999px; background: #f0ebe1; overflow: hidden; }
.hs-bar .fill { height: 100%; border-radius: 999px; background: #2a78d6; }
.hs-bar b { font-variant-numeric: tabular-nums; color: var(--hs-navy); font-size: 13px; }
.hs-steps { counter-reset: s; display: flex; flex-direction: column; gap: 14px; margin: 4px 0 6px; }
.hs-steps div { display: grid; grid-template-columns: 34px 1fr; gap: 12px; align-items: start; font-size: 14px; color: var(--hs-ink); }
.hs-steps div::before { counter-increment: s; content: counter(s); width: 30px; height: 30px; border-radius: 50%;
                        background: var(--hs-gold-soft); color: var(--hs-gold-ink); font-weight: 700; display: flex;
                        align-items: center; justify-content: center; }
.hs-steps b { display: block; color: var(--hs-navy); }
.hs-note { display: flex; gap: 10px; align-items: flex-start; background: #fbf7ee; border: 1px solid #efe3c8; border-radius: 12px;
           padding: 12px 14px; font-size: 13px; color: #5d4d2b; line-height: 1.5; }
.hs-note .hs-ico { color: var(--hs-gold-ink); }

/* pipeline flow (method page) */
.hs-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 6px; margin: 6px 0 4px; }
.hs-flow .node { flex: 1 1 96px; min-width: 0; background: #fff; border: 1px solid var(--hs-line); border-radius: 12px;
                 padding: 12px 12px; box-shadow: 0 1px 2px rgba(28,37,51,.05); border-top: 3px solid var(--hs-gold); }
.hs-flow .node b { display: block; color: var(--hs-navy); font-size: 14px; }
.hs-flow .node small { color: var(--hs-muted); font-size: 12px; }
.hs-flow .arrow { align-self: center; color: #b9ab8c; font-size: 18px; }

/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid rgba(197,160,89,.22); }
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] p { color: #d9b86c !important; text-transform: uppercase;
                                     letter-spacing: .12em; font-size: 11px !important; font-weight: 700 !important; }
[data-testid="stSidebarNavLink"] { border-radius: 10px; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(197,160,89,.16) !important; box-shadow: inset 3px 0 0 #c5a059; }
[data-testid="stSidebarNavLink"][aria-current="page"] p { color: #fff !important; font-weight: 600; }
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"] { color: #e2c88f !important; }
.hs-side { background: rgba(255,255,255,.05); border: 1px solid rgba(197,160,89,.28); border-radius: 12px;
           padding: 14px 14px 10px; font-size: 13px; }
.hs-side .t { text-transform: uppercase; letter-spacing: .1em; font-size: 10.5px; font-weight: 700; color: #d9b86c; }
.hs-side .m { font-size: 17px; font-weight: 700; color: #fff; margin: 4px 0 8px; }
.hs-side .m span { font-size: 11.5px; font-weight: 600; color: #f3e3bd; background: rgba(197,160,89,.2);
                   border-radius: 999px; padding: 2px 8px; margin-left: 6px; }
.hs-side dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; margin: 0 0 8px !important; padding: 0 !important; }
.hs-side dt, .hs-side dd { margin: 0 !important; padding: 0 !important; font-size: 13px; }
.hs-side dt { color: rgba(233,237,243,.68); font-weight: 400; } .hs-side dd { text-align: right; color: #fff; font-weight: 600; }
.hs-side code { font-size: 11px; color: #f3e3bd; background: transparent; padding: 0; }
.hs-side-note { font-size: 12px; color: rgba(233,237,243,.66); line-height: 1.5; margin-top: 12px; }

/* footer */
.hs-footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--hs-line); display: flex; justify-content: space-between;
             flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--hs-muted); }
.hs-footer b { color: var(--hs-navy); font-family: var(--hs-serif); font-size: 13px; }
</style>
"""


def icon(name: str, cls: str = "") -> str:
    """A Material Symbols icon (the font Streamlit already ships) for use inside st.html."""
    return f'<span class="hs-ico {cls}">{escape(name)}</span>'


def svg_img(svg: str, width: int, height: int, alt: str = "") -> str:
    """Inline SVG as an <img> data URI: st.html's sanitiser removes <svg> elements but keeps images."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img src="data:image/svg+xml;base64,{data}" width="{width}" height="{height}" alt="{escape(alt)}" '
            'style="display:block">')


@functools.lru_cache(maxsize=2)
def skyline_svg(large: bool = False) -> str:
    """The header artwork, a night-time hotel skyline with lit windows (drawn here, no image files), as an <img>."""
    rng = random.Random(11)
    width, height = (380, 180) if large else (300, 142)
    parts = ['<svg viewBox="0 0 360 170" xmlns="http://www.w3.org/2000/svg">',
             '<circle cx="320" cy="34" r="24" fill="#ecd496" opacity=".10"/>',
             '<circle cx="320" cy="34" r="11" fill="#ecd496" opacity=".95"/>']
    for x, y in [(30, 22), (92, 12), (176, 18), (232, 30), (276, 10), (350, 64)]:
        parts.append(f'<circle cx="{x}" cy="{y}" r="1.1" fill="#fff" opacity=".55"/>')
    towers = [(4, 34, .40), (42, 48, .64), (94, 30, .48), (128, 60, .80), (192, 40, .58), (236, 30, .36),
              (270, 46, .52), (320, 36, .30)]
    shades = ["#1b3658", "#21426a", "#183152", "#264b75"]
    for i, (x, w, frac) in enumerate(towers):
        top = 170 - frac * 150
        parts.append(f'<rect x="{x}" y="{top:.1f}" width="{w}" height="{170 - top:.1f}" rx="2" fill="{shades[i % 4]}"/>')
        for wy in range(int(top) + 10, 162, 9):
            for wx in range(x + 5, x + w - 6, 7):
                if rng.random() < .34:
                    parts.append(f'<rect x="{wx}" y="{wy}" width="3.6" height="5" fill="#f3d58c" '
                                 f'opacity="{rng.uniform(.55, .95):.2f}"/>')
                else:
                    parts.append(f'<rect x="{wx}" y="{wy}" width="3.6" height="5" fill="#fff" opacity=".06"/>')
    parts.append('<rect x="138" y="34" width="40" height="12" rx="2" fill="#c5a059"/>'
                 '<text x="158" y="43.2" text-anchor="middle" font-family="Georgia, serif" font-size="8.5" '
                 'font-weight="700" letter-spacing="1.6" fill="#0b1d33">HOTEL</text>'
                 '<rect x="0" y="167" width="360" height="3" fill="#c5a059"/></svg>')
    return svg_img("".join(parts), width, height)


def apply_style():
    """Global CSS and the sidebar logo; called once per run from streamlit_app.py."""
    st.html(CSS)
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo_icon.svg"), size="large")


def page_header(eyebrow: str, title: str, subtitle: str, chips: list | None = None, icon_name: str = "hotel",
                large: bool = False):
    chip_html = "".join(f'<span class="hs-chip">{escape(c)}</span>' for c in chips or [])
    st.html(f'<div class="hs-hero{" lg" if large else ""}"><div class="hs-hero-text">'
            f'<div class="hs-eyebrow">{icon(icon_name)}{escape(eyebrow)}</div><h1>{escape(title)}</h1>'
            f'<p>{subtitle}</p>{f"<div class=hs-chips>{chip_html}</div>" if chips else ""}</div>'
            f'<div class="hs-hero-art">{skyline_svg(large)}</div></div>')


def section(title: str, note: str = ""):
    st.html(f'<div class="hs-section"><h3>{escape(title)}</h3><span>{escape(note)}</span></div>')


def step(icon_name: str, title: str, note: str = ""):
    st.html(f'<div class="hs-step">{icon(icon_name)}{escape(title)}'
            f'{f"<small>· {escape(note)}</small>" if note else ""}</div>')


def nav_link(page: str, label: str, icon_name: str = ":material/arrow_forward:"):
    """st.page_link that degrades to plain text when a page runs on its own (as in the tests)."""
    try:
        st.page_link(page, label=label, icon=icon_name)
    except StreamlitAPIException:
        st.caption(label)


def footer():
    st.html('<div class="hs-footer"><span><b>HotelSight</b> · Reservation cancellation risk · Final Year Project, '
            'UiTM</span><span>Data: Antonio, de Almeida &amp; Nunes (2019), <i>Hotel booking demand datasets</i>. '
            'Research prototype, not validated for live use.</span></div>')


@st.cache_resource(show_spinner="Loading the saved model (once)…")
def get_model() -> DeployedModel:
    """Loaded once per server process. Nothing is trained here."""
    m = DeployedModel(threads=os.cpu_count() or 2)   # threads only change speed (batch scoring), not results
    _ = m.explainer           # build the TreeSHAP explainer once as well
    return m


@st.cache_data
def get_countries() -> dict:
    return country_names()


@st.cache_data
def get_ranges() -> dict:
    return input_ranges()


@st.cache_data
def load(name: str):
    return getattr(results, name)()


def pct(x: float, digits: int = 1) -> str:
    """Percent text; extra decimals near 0% and 100% so 99.98% never shows as 100.0%."""
    if digits == 1 and (x > 0.995 or x < 0.005):
        digits = 2
    return f"{100 * x:.{digits}f}%"


def euros(x: float) -> str:
    return f"€{x:,.0f}" if abs(x) >= 1000 else f"€{x:,.2f}"


def friendly(field: str) -> str:
    return FRIENDLY_NAME.get(field, field)


def base_chart(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(font=FONT, background="transparent", padding={"left": 6, "right": 10, "top": 4, "bottom": 4})
                 .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#ece7dc", domainColor="#d6cfbf",
                                 labelFontSize=12, titleFontSize=12, titleFontWeight=600, labelLimit=420, tickColor="#d6cfbf")
                 .configure_view(strokeWidth=0)
                 .configure_title(color=NAVY, fontSize=14, fontWeight=600, anchor="start")
                 .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=12, symbolType="circle"))


def diverging_legend(title: str = ""):
    """Legend for diverging_bars, drawn in HTML so long axis labels never push it out of the chart."""
    dot = '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{};margin-right:6px"></span>'
    st.html(f'<div style="display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;font-size:13px;color:{MUTED}">'
            f'{f"<span class=hs-card-title style=margin:0;flex:1>{escape(title)}</span>" if title else ""}'
            f'<span>{dot.format(RED)}Raises cancellation risk</span><span>{dot.format(BLUE)}Lowers cancellation risk</span></div>')


def diverging_bars(df: pd.DataFrame, label: str, value: str, value_title: str, tooltip: list, height=None):
    """Horizontal bars from 0: red = pushes towards cancellation, blue = away from it."""
    df = df.copy()
    df["Direction"] = df[value].map(lambda v: "Raises cancellation risk" if v > 0 else "Lowers cancellation risk")
    df["text"] = df[value].map(lambda v: f"{v:+.2f}")
    order = list(df[label])
    lo, hi = min(0.0, float(df[value].min())), max(0.0, float(df[value].max()))
    pad = 0.18 * max(hi - lo, 1e-6)                       # room for the value labels on both sides
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        y=alt.Y(f"{label}:N", sort=order, title=None),
        x=alt.X(f"{value}:Q", title=value_title,
                scale=alt.Scale(domain=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)], nice=False)),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises cancellation risk", "Lowers cancellation risk"],
                                                       range=[RED, BLUE]), legend=None),   # HTML legend: see diverging_legend
        tooltip=tooltip)
    text = alt.Chart(df).mark_text(dx=alt.ExprRef("datum['" + value + "'] > 0 ? 6 : -6"),
                                   align=alt.ExprRef("datum['" + value + "'] > 0 ? 'left' : 'right'"),
                                   color=INK, fontSize=12).encode(
        y=alt.Y(f"{label}:N", sort=order), x=f"{value}:Q", text="text:N")
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")
    return base_chart((bars + text + rule).properties(height=alt.Step(height or 34)))


def sidebar_model_card():
    sel = load("selected")
    p = sel["manual_parameters"]
    rows = [("Trees", p["n_estimators"]), ("Max depth", p["max_depth"]), ("Learning rate", p["learning_rate"]),
            ("Column sample", p["colsample_bytree"]), ("Inputs", 24), ("Threshold", "0.50")]
    with st.sidebar:
        st.html(f"""
        <div class="hs-side">
          <div class="t">Deployed model</div>
          <div class="m">{escape(sel['model'])}<span>config {sel['configuration']}</span></div>
          <dl>{''.join(f'<dt>{k}</dt><dd>{v}</dd>' for k, v in rows)}</dl>
          <div>Arm <code>retained_historical24</code></div>
          <div>SHA-256 <code>{sel['model_sha256'][:12]}…</code> · checked at load</div>
        </div>
        <div class="hs-side-note">Research prototype evaluated on historical records from two Portuguese hotels
        (2015–2017). Not validated for live use or for Malaysian hotels.</div>""")
