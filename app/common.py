"""Shared, cached loaders, page styling and chart helpers for every dashboard page."""
from __future__ import annotations

import base64
import functools
import os
import random
import re
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

from hotelsight import currency, results            # noqa: E402
from hotelsight import waterfall as wf              # noqa: E402
from hotelsight.contract import FRIENDLY_NAME, country_names, input_ranges   # noqa: E402
from hotelsight.pipeline import DeployedModel       # noqa: E402

ASSETS = ROOT / "assets"

# Data colours, checked with the dataviz palette validator on a white surface. Each one has a single meaning:
STAY = "#2A78D6"      # "lowers cancellation risk" / likely to stay / primary series
CANCEL = "#E34948"    # "raises cancellation risk" / likely to cancel
SECOND = "#B7791F"    # second series, where the series has nothing to do with risk
FADED = "#CBD5E1"     # de-emphasised series
TOTAL = "#8492A6"     # a score drawn from zero (waterfall start and end), not a push up or down
INK = "#0F1B2D"
MUTED = "#55637A"
SEQ = ["#CDE2FB", "#9EC5F4", "#6DA7EC", "#3987E5", "#256ABF", "#184F95", "#0D366B"]   # one hue, light to dark
OUTCOME = alt.Scale(domain=["Likely to cancel", "Likely to stay"], range=[CANCEL, STAY])
# Charts use installed fonts only: Vega measures label widths at render time, before a web font such as
# the page's Inter has loaded, which clips long axis labels and overlaps legend entries.
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"

CSS = """
<style>
:root { --ink:#0F1B2D; --ink-2:#3B4A60; --muted:#55637A; --line:#E2E8F0; --field:#8492A6;
        --bg:#F5F7FA; --surface:#FFFFFF; --subtle:#F8FAFC; --side:#131F36;
        --accent:#1D4ED8; --accent-dark:#1E40AF; --accent-soft:#E8EFFC; --coral:#C2353A; --coral-soft:#FDECEC;
        --heading:"Plus Jakarta Sans", Inter, "Segoe UI", sans-serif;
        --shadow:0 1px 2px rgba(15,27,45,.06); }
[data-testid="stMainBlockContainer"] { padding: 1.6rem 2rem 1.5rem; max-width: 1280px; margin: 0 auto; }
[data-testid="stHeader"] { background: transparent; }
.hs-ico { font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal; font-size: 20px;
          line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
          direction: ltr; font-feature-settings: "liga"; -webkit-font-smoothing: antialiased; vertical-align: middle; }

/* page and section headings */
.hs-page { margin: 0 0 4px; }
.hs-kicker { font-size: 12.5px; font-weight: 600; color: var(--accent); margin-bottom: 2px; }
.hs-page h1 { font-family: var(--heading); font-size: 26px; line-height: 1.25; font-weight: 700; letter-spacing: -.01em;
              color: var(--ink); margin: 0; padding: 0; }
.hs-page p { margin: 4px 0 0; font-size: 14.5px; line-height: 1.5; color: var(--muted); max-width: 800px; }
.hs-stamp { display: flex; align-items: center; gap: 6px; margin: 6px 0 0; font-size: 12.5px; color: var(--muted); }
.hs-stamp .hs-ico { font-size: 15px; }
.hs-section { display: flex; align-items: baseline; flex-wrap: wrap; gap: 2px 10px; margin: 18px 0 0; }
.hs-section h2 { font-family: var(--heading); font-size: 17px; line-height: 1.4; font-weight: 700; letter-spacing: -.01em;
                 color: var(--ink); margin: 0; padding: 0; }
.hs-section span { font-size: 13px; color: var(--muted); }
.hs-card-title { font-family: var(--heading); font-size: 14.5px; font-weight: 700; letter-spacing: -.005em; color: var(--ink);
                 margin-bottom: 10px; }
.hs-card-title small { font-family: Inter, "Segoe UI", sans-serif; font-weight: 400; font-size: 12.5px; color: var(--muted);
                       letter-spacing: 0; margin-left: 6px; }
.hs-step { display: flex; align-items: center; gap: 8px; font-family: var(--heading); font-weight: 700; font-size: 14px;
           color: var(--ink); margin: 2px 0 -6px; }
.hs-step .hs-ico { color: var(--accent); font-size: 19px; }

/* cards */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
                           padding: 13px 15px; box-shadow: var(--shadow); position: relative; min-height: 98px;
                           box-sizing: border-box; }
[data-testid="stMetric"] > div { padding: 0; }
[data-testid="stMetricLabel"] p { font-size: 13px; font-weight: 500; color: var(--muted); }
/* KPI rows (see kpis): every card the same height, with an icon tile that says what the number is */
[class*="st-key-kpi"] [data-testid="stMetric"]::after { font-family: "Material Symbols Rounded"; font-size: 20px; line-height: 1;
  font-feature-settings: "liga"; -webkit-font-smoothing: antialiased; position: absolute; top: 12px; right: 12px;
  width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
  color: var(--accent); background: var(--accent-soft); font-size: 18px; }
[class*="st-key-kpi"] [data-testid="stMetricLabel"] { padding-right: 40px; }
[data-testid="stMetricValue"] { font-family: var(--heading); font-size: 24px; line-height: 1.25; font-weight: 700;
                                letter-spacing: -.02em; color: var(--ink); font-variant-numeric: tabular-nums; }
[data-testid="stMetricDelta"] { background: transparent !important; padding: 0 !important; color: var(--muted) !important;
                                font-size: 12.5px; }
[data-testid="stMetricDelta"] svg { display: none; }
@media (max-width: 1180px) { [data-testid="stMetricValue"] { font-size: 21px; } }
[data-testid="stForm"], [class*="st-key-card"] { background: var(--surface); border: 1px solid var(--line) !important;
                                                 border-radius: 10px; box-shadow: var(--shadow); }
[data-testid="stForm"] { padding: 18px 20px 20px; }
[class*="st-key-card"] { padding: 14px 16px; }
[class*="st-key-card"] [data-testid="stMetric"] { border: none; box-shadow: none; padding: 0; background: transparent; }
.st-key-card_risk [data-testid="stMetricValue"] { font-size: 40px; line-height: 1.1; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; opacity: 1 !important; }
[data-testid="stCaptionContainer"] p { font-size: 12.5px; line-height: 1.5; }
.hs-section { margin-top: 22px !important; }

/* overview header: a compact title bar with actions, as in BI tools (no marketing banner) */
.st-key-ov_actions [data-testid="stPageLink"] a { justify-content: center; padding: 6px 14px; border-radius: 8px;
  border: 1px solid var(--line); background: var(--surface); box-shadow: var(--shadow); }
.st-key-ov_actions [data-testid="stPageLink"] a:hover { border-color: var(--accent); text-decoration: none; }
.st-key-ov_primary [data-testid="stPageLink"] a { background: var(--accent); border-color: var(--accent); }
.st-key-ov_primary [data-testid="stPageLink"] p, .st-key-ov_primary [data-testid="stIconMaterial"] { color: #fff !important; }
.st-key-ov_primary [data-testid="stPageLink"] a:hover { background: var(--accent-dark); }
[data-testid="stExpander"] details { background: var(--surface); border-color: var(--line); border-radius: 10px; }
[data-testid="stTabs"] button p { font-weight: 600; }
[data-testid="stDataFrame"] { border-radius: 8px; }

/* the theme gives fields a white fill and a border that clears 3:1; everything else keeps the lighter line */
[data-testid="stFileUploaderDropzone"] { background: var(--subtle); border: 1px dashed var(--field); }
hr { border-color: var(--line) !important; }
[data-testid="stTabs"] [role="tablist"] { border-bottom-color: var(--line); }
[data-testid="stDataFrame"] { border-color: var(--line); }

/* the submit row stays in view while the long form scrolls */
[data-testid="stLayoutWrapper"]:has(> .st-key-form_actions) { position: sticky; bottom: 0; z-index: 5; background: var(--surface);
  width: calc(100% + 40px); max-width: none; margin: 2px -20px -20px; padding: 12px 20px; box-sizing: border-box;
  border-top: 1px solid var(--line); border-radius: 0 0 12px 12px; }

/* buttons and links */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] { font-weight: 600;
  transition: background-color .15s ease; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: var(--accent-dark); border-color: var(--accent-dark); color: #fff; }
[data-testid="stPageLink"] p { font-weight: 600; color: var(--accent); }
[data-testid="stPageLink"] a { padding-left: 0; }
[data-testid="stPageLink"] a:hover { background: transparent; text-decoration: underline; text-decoration-color: var(--accent); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }

/* home band */
.st-key-hero { background: var(--side) __HERO_ART__ no-repeat right bottom / auto 100%; border-radius: 14px; padding: 24px 28px; }
.hs-hero h1 { font-family: var(--heading); font-size: 26px; line-height: 1.25; font-weight: 700; letter-spacing: -.01em;
              color: #fff; margin: 0; padding: 0; }
.hs-hero p { margin: 6px 0 0; font-size: 14.5px; line-height: 1.5; color: #C2CEE0; max-width: 600px; }
.st-key-hero [data-testid="stPageLink"] a { justify-content: center; padding: 7px 14px; border-radius: 8px; border: 1px solid #5F7296; }
.st-key-hero [data-testid="stPageLink"] p { color: #fff; }
.st-key-hero [data-testid="stPageLink"] [data-testid="stIconMaterial"] { color: #fff; }
.st-key-hero [data-testid="stPageLink"] a:hover { background: rgba(255,255,255,.08); text-decoration: none; }
.st-key-hero_primary [data-testid="stPageLink"] a { background: #fff; border-color: #fff; }
.st-key-hero_primary [data-testid="stPageLink"] p, .st-key-hero_primary [data-testid="stPageLink"] [data-testid="stIconMaterial"] { color: var(--accent-dark); }
.st-key-hero_primary [data-testid="stPageLink"] a:hover { background: var(--accent-soft); }
@media (max-width: 1100px) { .st-key-hero { background-image: none; } }

/* reservation result */
[data-testid="stLayoutWrapper"]:has(> .st-key-result_panel) { position: sticky; top: 1rem; align-self: flex-start; width: 100%; }
@media (max-width: 1180px) {
  [data-testid="stLayoutWrapper"]:has(> .st-key-result_panel) { position: static; }
  [data-testid="stHorizontalBlock"]:has(.st-key-result_panel) { flex-direction: column; }
  [data-testid="stHorizontalBlock"]:has(.st-key-result_panel) > [data-testid="stColumn"] { width: 100%; flex: 1 1 auto; }
}
.hs-empty { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 6px; padding: 44px 24px; }
.hs-empty .hs-ico { font-size: 28px; color: var(--accent); background: var(--accent-soft); width: 56px; height: 56px; border-radius: 50%;
                    display: flex; align-items: center; justify-content: center; margin-bottom: 6px; }
.hs-empty b { font-family: var(--heading); font-size: 15.5px; color: var(--ink); }
.hs-empty span { font-size: 14px; line-height: 1.5; color: var(--muted); max-width: 330px; }
.hs-res-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap; margin-bottom: 2px; }
.hs-res-title { font-family: var(--heading); font-size: 15.5px; color: var(--ink); font-weight: 700; line-height: 1.3; }
.hs-res-sub { font-size: 13px; color: var(--muted); }
.hs-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 600; color: var(--accent);
            background: var(--accent-soft); border-radius: 6px; padding: 3px 8px; white-space: nowrap; }
.hs-badge .hs-ico { font-size: 15px; }
.hs-verdict { display: inline-flex; align-items: center; gap: 5px; font-weight: 600; font-size: 13.5px; white-space: nowrap;
              padding: 4px 10px 4px 7px; border-radius: 6px; margin-top: 6px; }
.hs-verdict .hs-ico { font-size: 18px; }
.hs-verdict.hi { background: var(--coral-soft); color: #B42D33; }
.hs-verdict.lo { background: var(--accent-soft); color: var(--accent); }
.hs-dial img { margin-left: auto; }
.hs-timeline { display: grid; grid-template-columns: auto minmax(24px, 1fr) auto minmax(24px, 1fr) auto; align-items: center;
               gap: 8px; margin: 2px 0 14px; }
.hs-tl-node .lbl { font-size: 12px; color: var(--muted); }
.hs-tl-node .date { font-size: 14px; font-weight: 600; color: var(--ink); white-space: nowrap; }
.hs-tl-node .day { font-size: 12px; color: var(--muted); }
.hs-tl-line { height: 1px; background: var(--line); }
.st-key-card_stay { container-type: inline-size; }
@container (max-width: 400px) {
  .hs-timeline { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .hs-tl-line { display: none; }
  .hs-tl-node .date { white-space: normal; }
  .st-key-card_stay .hs-facts { grid-template-columns: minmax(0, 1fr); }
}
.hs-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 20px; }
.hs-fact { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0; border-top: 1px solid var(--line); font-size: 13.5px; }
.hs-fact span { color: var(--muted); white-space: nowrap; }
.hs-fact b { color: var(--ink); font-weight: 600; text-align: right; overflow-wrap: anywhere; }

/* driver lists and bars */
.hs-list-title { display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; color: var(--ink-2); margin: 0 0 2px; }
.hs-list-title .hs-ico { font-size: 17px; }
.hs-list-title.up .hs-ico { color: #C2353A; } .hs-list-title.down .hs-ico { color: var(--accent); }
.hs-drivers { list-style: none; padding: 0 !important; margin: 0 0 10px !important; }
.hs-drivers:last-child { margin-bottom: 0 !important; }
.hs-drivers li { display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; margin: 0 !important; font-size: 13.5px;
                 color: var(--ink-2); border-bottom: 1px solid var(--line); }
.hs-drivers li:last-child { border-bottom: none; }
.hs-drivers li b { font-variant-numeric: tabular-nums; color: var(--ink); font-weight: 600; }
.hs-bars { display: flex; flex-direction: column; gap: 10px; }
.hs-bar { display: grid; grid-template-columns: minmax(142px, 1fr) minmax(60px, 1.6fr) 46px; align-items: center; gap: 12px;
          font-size: 13.5px; color: var(--ink-2); }
.hs-bar .track { height: 8px; border-radius: 4px; background: #EDF1F7; overflow: hidden; }
.hs-bar .fill { height: 100%; border-radius: 4px; background: #2A78D6; }
.hs-bar b { font-variant-numeric: tabular-nums; color: var(--ink); font-size: 13.5px; font-weight: 600; text-align: right; }
.hs-steps { counter-reset: s; display: flex; flex-direction: column; gap: 10px; margin: 0 0 12px; }
.hs-steps div { display: grid; grid-template-columns: 26px 1fr; gap: 10px; align-items: start; font-size: 13.5px; line-height: 1.45; color: var(--muted); }
.hs-steps div::before { counter-increment: s; content: counter(s); width: 24px; height: 24px; border-radius: 50%; font-size: 12.5px;
                        background: var(--accent-soft); color: var(--accent); font-weight: 700; display: flex; align-items: center; justify-content: center; }
.hs-steps b { display: block; color: var(--ink); font-weight: 600; }
.hs-note { display: flex; gap: 8px; align-items: flex-start; background: var(--subtle); border: 1px solid var(--line); border-radius: 8px;
           padding: 10px 12px; font-size: 13px; color: var(--ink-2); line-height: 1.5; }
.hs-note .hs-ico { color: var(--muted); font-size: 18px; }

/* the plain-language answer shown above a chart */
.hs-take { display: flex; gap: 8px; align-items: flex-start; font-size: 14px; line-height: 1.5; color: var(--ink);
           background: var(--accent-soft); border-radius: 8px; padding: 8px 12px; margin: -2px 0 10px; }
.hs-take .hs-ico { font-size: 18px; color: var(--accent); margin-top: 1px; }
.hs-take.risk { background: var(--coral-soft); } .hs-take.risk .hs-ico { color: var(--coral); }
.hs-take.good { background: #E7F6EC; } .hs-take.good .hs-ico { color: #15803D; }
.hs-take b { font-weight: 700; }

/* overview: today's summary strip and the explore tiles */
.hs-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 6px 0 2px; }
@media (max-width: 1000px) { .hs-summary { grid-template-columns: minmax(0, 1fr); } }
.hs-sum { display: flex; gap: 12px; align-items: flex-start; background: var(--surface); border: 1px solid var(--line);
          border-left: 4px solid var(--accent); border-radius: 0 10px 10px 0; padding: 12px 14px; box-shadow: var(--shadow); }
.hs-sum.risk { border-left-color: var(--coral); } .hs-sum.good { border-left-color: #15803D; }
.hs-sum .hs-ico { font-size: 22px; color: var(--accent); background: var(--accent-soft); border-radius: 8px; padding: 6px; }
.hs-sum.risk .hs-ico { color: var(--coral); background: var(--coral-soft); }
.hs-sum.good .hs-ico { color: #15803D; background: #E7F6EC; }
.hs-sum small { display: block; font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase;
                letter-spacing: .04em; margin-bottom: 2px; }
.hs-sum span { font-size: 14px; line-height: 1.45; color: var(--ink); }
.hs-tile small { display: block; font-size: 12.5px; color: var(--muted); font-weight: 600; }
.hs-tile b { display: block; font-family: var(--heading); font-size: 22px; color: var(--ink); line-height: 1.3;
             font-variant-numeric: tabular-nums; }
.hs-tile span { display: block; font-size: 13px; color: var(--ink-2); line-height: 1.45; margin: 2px 0 4px; }

/* findings, alerts and recommendations */
.hs-find p { margin: 4px 0 6px; font-size: 13.5px; color: var(--ink-2); line-height: 1.55; }
.hs-find .hs-card-title .hs-ico { color: var(--accent); font-size: 19px; margin-right: 2px; }
.hs-find.risk .hs-card-title .hs-ico { color: var(--coral); }
.hs-find.good .hs-card-title .hs-ico { color: #15803D; }
.hs-alerts { display: flex; flex-direction: column; gap: 8px; }
.hs-alert { display: flex; gap: 10px; align-items: flex-start; font-size: 13.5px; color: var(--ink-2); line-height: 1.5;
            padding: 9px 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--subtle); }
.hs-alert .hs-ico { font-size: 19px; color: var(--accent); }
.hs-alert.risk .hs-ico { color: var(--coral); }
.hs-alert.good .hs-ico { color: #15803D; }
.hs-alert b { color: var(--ink); }

/* pipeline flow (method page) */
.hs-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 6px; margin: 6px 0 4px; }
.hs-flow .node { flex: 1 1 96px; min-width: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
                 padding: 10px 12px; box-shadow: var(--shadow); }
.hs-flow .node b { display: block; color: var(--ink); font-size: 13.5px; font-weight: 600; }
.hs-flow .node small { color: var(--muted); font-size: 12px; }
.hs-flow .arrow { align-self: center; color: var(--field); font-size: 16px; }
.hs-kv { display: grid; grid-template-columns: auto 1fr; font-size: 13.5px; margin: 0 !important; padding: 0 !important; }
.hs-kv dt, .hs-kv dd { margin: 0 !important; padding: 6px 0 !important; border-top: 1px solid var(--line); }
.hs-kv dt { color: var(--muted); font-weight: 400; }
.hs-kv dd { color: var(--ink); font-weight: 600; text-align: right; overflow-wrap: anywhere; padding-left: 24px !important; }
.hs-kv code { font-size: 12.5px; color: var(--ink-2); background: transparent; padding: 0; }

/* sidebar */
[data-testid="stApp"]:not(:has([data-testid="stExpandSidebarButton"])) [data-testid="stSidebar"] {
  width: 256px !important; min-width: 256px !important; max-width: 256px !important; }
[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] { width: 100% !important; transform: none !important;
  left: 0 !important; }
[data-testid="stApp"]:has([data-testid="stExpandSidebarButton"]) [data-testid="stMainBlockContainer"] { padding-top: 3.75rem; }
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] { margin-top: auto; padding-bottom: 1rem; }
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] p { color: #93A4BD !important; font-size: 12px !important;
                                                                          font-weight: 600 !important; letter-spacing: .02em; }
[data-testid="stSidebarNavLink"] { border-radius: 8px; transition: background-color .15s ease; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,.06); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(96,165,250,.16) !important; box-shadow: inset 2px 0 0 #60A5FA; }
[data-testid="stSidebarNavLink"][aria-current="page"] p { color: #fff !important; font-weight: 600; }
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"] { color: #93C5FD !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color: #93A4BD !important; }
.hs-side { border-top: 1px solid #27354F; padding-top: 12px; }
.hs-side b { display: block; color: #fff; font-size: 13px; font-weight: 600; }

/* footer */
.hs-footer { margin-top: 32px; padding-top: 12px; border-top: 1px solid var(--line); display: flex; justify-content: space-between;
             flex-wrap: wrap; gap: 4px 16px; font-size: 12px; color: var(--muted); }
</style>
"""


def icon(name: str, cls: str = "") -> str:
    """A Material Symbols icon (the font Streamlit already ships) for use inside st.html."""
    return f'<span class="hs-ico {cls}" aria-hidden="true">{escape(name)}</span>'


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def svg_img(svg: str, width: int, height: int, alt: str = "") -> str:
    """Inline SVG as an <img> data URI: st.html's sanitiser removes <svg> elements but keeps images."""
    return f'<img src="{svg_uri(svg)}" width="{width}" height="{height}" alt="{escape(alt)}" style="display:block">'


def skyline_svg() -> str:
    """The home band's artwork: a quiet hotel skyline with a few lit windows (drawn here, no image files)."""
    rng = random.Random(11)
    parts = ['<svg viewBox="0 0 420 170" width="420" height="170" xmlns="http://www.w3.org/2000/svg">']
    towers = [(70, 42, .34), (116, 52, .58), (172, 34, .44), (210, 66, .80), (280, 44, .54), (328, 34, .36), (366, 54, .50)]
    shades = ["#1A2944", "#1F3050", "#18263F", "#223558"]
    for i, (x, w, frac) in enumerate(towers):
        top = 170 - frac * 150
        parts.append(f'<rect x="{x}" y="{top:.1f}" width="{w}" height="{170 - top:.1f}" rx="2" fill="{shades[i % 4]}"/>')
        for wy in range(int(top) + 10, 164, 10):
            for wx in range(x + 6, x + w - 6, 8):
                lit = rng.random() < .22
                fill = "#F6D58A" if lit else "#FFFFFF"
                opacity = rng.uniform(.5, .9) if lit else .05
                parts.append(f'<rect x="{wx}" y="{wy}" width="3.6" height="5" fill="{fill}" opacity="{opacity:.2f}"/>')
    parts.append('</svg>')
    return "".join(parts)


@functools.lru_cache(maxsize=1)
def css() -> str:
    return CSS.replace("__HERO_ART__", f'url("{svg_uri(skyline_svg())}")')


def apply_style():
    """Global CSS and the sidebar logo; called once per run from streamlit_app.py."""
    st.html(css())
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo_icon.svg"), size="large")


def page_header(area: str, title: str, subtitle: str):
    st.html(f'<div class="hs-page"><div class="hs-kicker">{escape(area)}</div><h1>{escape(title)}</h1><p>{subtitle}</p></div>')


def data_stamp():
    """One quiet line saying how fresh each data source is, shown under the title of the analysis pages."""
    from hotelsight import history
    d = booking_history()
    h = history.load()
    saved = (f"{len(h):,} saved predictions, last saved {h['created_at'].max():%d %b %Y, %H:%M}" if len(h)
             else "No saved predictions yet")
    st.html(f'<div class="hs-stamp">{icon("schedule")}<span>{saved} · Booking history '
            f'{d["arrival"].min():%b %Y} – {d["arrival"].max():%b %Y}</span></div>')


def section(title: str, note: str = ""):
    st.html(f'<div class="hs-section"><h2>{escape(title)}</h2><span>{escape(note)}</span></div>')


KPI_TONES = {"blue": ("var(--accent)", "var(--accent-soft)"), "coral": ("#B42D33", "var(--coral-soft)"),
             "slate": ("var(--ink-2)", "#EEF2F7")}


def kpis(key: str, icons: list[str], tones: dict | None = None):
    """Columns for a row of st.metric cards. Each card gets an icon tile; tones maps a card index to
    "coral" (cancellation risk) or "slate" (neutral) instead of the default blue."""
    rules = ""
    for i, name in enumerate(icons):
        fg, bg = KPI_TONES[(tones or {}).get(i, "blue")]
        rules += (f'.st-key-kpi_{key} [data-testid="stColumn"]:nth-child({i + 1}) [data-testid="stMetric"]::after'
                  f'{{content:"{name}";color:{fg};background:{bg}}}')
    st.html(f"<style>{rules}</style>")
    return st.container(key=f"kpi_{key}").columns(len(icons))


def takeaway(text: str, kind: str = "info"):
    """The one-line answer a chart gives, in plain words, shown above the chart. text may contain <b>."""
    ic = {"info": "lightbulb", "risk": "warning", "good": "check_circle"}[kind]
    st.html(f'<div class="hs-take {kind}">{icon(ic)}<span>{text}</span></div>')


def money_short(x_eur) -> str:
    """A short money figure for sentences, e.g. €1.97M or €45.3K, in the display currency."""
    v = float(to_display(x_eur))
    sym = euros(0).replace("0", "").replace(".", "").replace(",", "").strip()
    if abs(v) >= 1e6:
        return f"{sym}{v / 1e6:.2f}M"
    return f"{sym}{v / 1e3:.1f}K" if abs(v) >= 1e4 else euros(x_eur)


def chart_card(key: str, question: str, answer: str | None = None, kind: str = "info"):
    """A card whose title is the question the chart answers, with the answer underneath. Use as a context manager."""
    box = st.container(key=key, height="stretch")
    with box:
        st.html(f'<div class="hs-card-title">{escape(question)}</div>')
        if answer:
            takeaway(answer, kind)
    return box


def dash(values: pd.Series) -> pd.Series:
    """Missing text shown as a dash instead of the table's grey 'None'."""
    return values.where(values.notna() & (values.astype(str).str.strip() != ""), "—")


def verdict_colours(df: pd.DataFrame, columns=("Prediction", "Correct?")):
    """Colour the verdict text in a table: coral for likely to cancel or wrong, blue for likely to stay.
    Large tables are returned unstyled, because styling every cell would slow the page down."""
    cols = [c for c in columns if c in df.columns]
    if not cols or len(df) > 2000:
        return df
    tone = {"Likely to cancel": "color:#B42D33;font-weight:600", "Wrong": "color:#B42D33;font-weight:600",
            "Likely to stay": "color:#1D4ED8;font-weight:600"}
    return df.style.map(lambda v: tone.get(v, ""), subset=cols)


def step(icon_name: str, title: str):
    st.html(f'<div class="hs-step">{icon(icon_name)}{escape(title)}</div>')


def nav_link(page: str, label: str, icon_name: str | None = None):
    """st.page_link that degrades to plain text when a page runs on its own (as in the tests)."""
    try:
        st.page_link(page, label=label, icon=icon_name)
    except StreamlitAPIException:
        st.caption(label)


def footer():
    st.html('<div class="hs-footer"><span>HotelSight · Hotel Booking Cancellation Prediction</span>'
            '<span>Dataset from Antonio, de Almeida &amp; Nunes (2019)</span></div>')


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


# ------------------------------------------------------------------ display currency
# The model works in euros, so data, history, exports and the room-rate input stay in EUR. Only amounts shown on
# screen are converted, using the dated rate table in hotelsight/currency.py.
def _fx() -> dict:
    return st.session_state.get("fx") or currency.load_rates()


def currency_code() -> str:
    code = st.session_state.get("currency", currency.BASE)
    return code if code in currency.CURRENCIES else currency.BASE


def euros(x: float) -> str:
    """An amount held in euros, shown in the chosen display currency (the name is kept from the euro-only version)."""
    return currency.fmt(x, currency_code(), _fx())


def money(x: float, digits: int | None = None) -> str:
    """Like euros(), with a fixed number of decimals when digits is given."""
    return currency.fmt(x, currency_code(), _fx(), digits)


@st.cache_data(show_spinner="Loading the booking history…")
def booking_history() -> pd.DataFrame:
    """The cleaned historical bookings for the performance views (read once per session)."""
    from hotelsight import analytics
    return analytics.load_bookings()


def saved_predictions():
    """(raw history with stay values, tidy insight frame) for the saved predictions, or (None, None) when empty."""
    from hotelsight import history, insights
    h = history.load()
    if h.empty:
        return None, None
    h = history.with_stay(h)
    data = insights.prepare(h, get_countries())
    data["reference"], data["batch_id"], data["source"] = h["reference"], h["batch_id"], h["source"]
    data["batch_name"], data["created_at"] = h["batch_name"], h["created_at"]
    return h, data


ALL_UPLOADS, ALL_YEARS, ALL_WEEKS = "All checks and uploads", "All years", "All weeks"
DATA_HISTORY, DATA_CURRENT = "Booking history dataset", "Current predicted bookings"   # the two performance sources


def prediction_filters(data: pd.DataFrame, key: str):
    """The same filter bar on the risk pages: check or upload, arrival year and arrival week.

    Returns (rows for the chosen upload and year, rows for the chosen week as well, chosen week or ALL_WEEKS).
    Another page can open this one on a given upload by setting st.session_state[f"{key}_batch"] first."""
    batches = (data.groupby("batch_id").agg(name=("batch_name", "first"), when=("created_at", "min"),
                                            n=("reference", "size")).sort_values("when", ascending=False))
    label = {b: f"{r['name']} · {r['when']:%d %b %Y %H:%M} · {r['n']:,} reservation{'s' * (r['n'] != 1)}"
             for b, r in batches.iterrows()}
    options = [ALL_UPLOADS] + list(batches.index)
    if st.session_state.get(f"{key}_batch") not in options:
        st.session_state.pop(f"{key}_batch", None)
    with st.container(key=f"card_{key}_filters"):
        c1, c2, c3 = st.columns([1.7, .8, 1.2])
        pick = c1.selectbox("Check or upload", options, key=f"{key}_batch",
                            format_func=lambda b: b if b == ALL_UPLOADS else label[b])
        d = data if pick == ALL_UPLOADS else data[data["batch_id"] == pick]
        years = sorted(int(y) for y in d["arrival"].dt.year.dropna().unique())
        year = c2.selectbox("Arrival year", [ALL_YEARS] + years, key=f"{key}_year_{pick}")
        if year != ALL_YEARS:
            d = d[d["arrival"].dt.year == year]
        weeks = [pd.Timestamp(w) for w in sorted(d["week"].dropna().unique())]
        week = c3.selectbox("Arrival week", [ALL_WEEKS] + weeks, key=f"{key}_week_{pick}_{year}",
                            format_func=lambda w: w if isinstance(w, str) else f"Week of {w:%d %b %Y}",
                            help="Weeks run Monday to Sunday by arrival date.")
    sel = d if week == ALL_WEEKS else d[d["week"] == week]
    return d, sel, week


def to_display(values_eur):
    """Euro amounts (a number or a Series) converted to the display currency, for charts."""
    return currency.convert(values_eur, currency_code(), _fx())


_EURO_TEXT = re.compile(r"€([\d,]+(?:\.\d+)?)")


def convert_euro_text(text):
    """Convert euro amounts written inside a saved text such as 'Average daily rate: €95.00' for display."""
    code = currency_code()
    if code == currency.BASE or not isinstance(text, str) or "€" not in text:
        return text
    return _EURO_TEXT.sub(lambda m: currency.fmt(float(m.group(1).replace(",", "")), code, _fx(), 2), text)


def money_view(df: pd.DataFrame, columns: list[str]):
    """A display copy of df with the named EUR columns converted and renamed, plus their column_config.

    A column called 'Stay value (EUR)' becomes 'Stay value (MYR)'. The original df is untouched, so exports stay in EUR."""
    code, fx = currency_code(), _fx()
    view, config = df.copy(), {}
    for col in columns:
        new = col.replace("(EUR)", f"({code})")
        view[col] = currency.convert(pd.to_numeric(view[col], errors="coerce"), code, fx)
        view = view.rename(columns={col: new})
        config[new] = st.column_config.NumberColumn(format=currency.column_format(code))
    return view, config


def currency_control():
    """Sidebar: choose the display currency and refresh the exchange rates."""
    if "fx" not in st.session_state:
        st.session_state["fx"] = currency.load_rates()
    with st.sidebar:
        st.selectbox("Show amounts in", list(currency.CURRENCIES), key="currency",
                     format_func=lambda c: f"{c} · {currency.CURRENCIES[c][0]}",
                     help="The model works in euros. The room rate you enter and any downloaded file stay in EUR. "
                          "Only the amounts shown on screen are converted.")
        code, fx = currency_code(), st.session_state["fx"]
        if code != currency.BASE:
            when = pd.Timestamp(fx["date"]).strftime("%d %b %Y")
            st.caption(f"1 EUR = {currency.rate(code, fx):,.4f} {code} · rates of {when}")
            if st.button("Update rates", key="fx_update", icon=":material/sync:", width="stretch",
                         help="Get the latest European Central Bank reference rates. Needs an internet connection."):
                try:
                    st.session_state["fx"] = currency.refresh()
                    st.toast("Exchange rates updated.", icon=":material/check_circle:")
                    st.rerun()
                except Exception:                                    # offline or the service is down: keep the old rates
                    st.toast("Could not reach the rate service. The saved rates are still in use.",
                             icon=":material/warning:")


def friendly(field: str) -> str:
    return FRIENDLY_NAME.get(field, field)


def base_chart(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(font=FONT, background="transparent", padding={"left": 6, "right": 16, "top": 4, "bottom": 4})
                 .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#EDF1F7", domainColor="#CBD5E1",
                                 labelFontSize=12, titleFontSize=12, titleFontWeight=500, titlePadding=8, labelLimit=260, tickColor="#CBD5E1")
                 .configure_axisDiscrete(ticks=False, domain=False, labelPadding=8)
                 .configure_view(strokeWidth=0)
                 .configure_title(color=INK, fontSize=14, fontWeight=600, anchor="start")
                 .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=12, symbolType="circle"))


def show_chart(chart):
    """Draw a finished chart at full width, without Streamlit's chart theme (base_chart sets the look)."""
    st.altair_chart(chart, width="stretch", theme=None)


def diverging_legend(title: str = "", totals: bool = False):
    """Legend for diverging_bars and waterfall_chart, drawn in HTML so long axis labels never push it out of the chart."""
    dot = '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{};margin-right:6px"></span>'
    score = f'<span>{dot.format(TOTAL)}Score</span>' if totals else ""
    st.html(f'<div style="display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;font-size:13px;color:{MUTED}">'
            f'{f"<span class=hs-card-title style=margin:0;flex:1>{escape(title)}</span>" if title else ""}'
            f'<span>{dot.format(CANCEL)}Raises cancellation risk</span><span>{dot.format(STAY)}Lowers cancellation risk</span>'
            f'{score}</div>')


def waterfall_chart(rows: list[dict], height=None):
    """SHAP waterfall for one booking: from the model's average score to this booking's score.

    Red steps raise the cancellation risk, blue steps lower it, grey bars are the two scores measured from zero.
    Each step starts where the previous one ended, so the bars are read left to right like a running total."""
    df = pd.DataFrame(rows)
    df["Type"] = [("Score" if r["kind"] != wf.STEP else "Raises cancellation risk" if r["value"] > 0
                   else "Lowers cancellation risk") for r in rows]
    signed = lambda v: f"{v:+.2f}".replace("-", "−")
    df["text"] = [(f"{r['value']:+.3f}".replace("-", "−") + f" · {wf.probability(r['score']):.1%}") if r["kind"] != wf.STEP
                  else signed(r["value"]) for r in rows]
    df["tip"] = df[["start", "end"]].max(axis=1)
    df["Contribution (log-odds)"] = [signed(r["value"]) if r["kind"] == wf.STEP else "–" for r in rows]
    df["Score after this step"] = [f"{r['score']:+.3f}".replace("-", "−") for r in rows]
    df["Probability at this point"] = [f"{wf.probability(r['score']):.1%}" for r in rows]
    order = list(df["label"])
    lo, hi = min(0.0, float(df[["start", "end"]].min().min())), max(0.0, float(df["tip"].max()))
    pad = 0.24 * max(hi - lo, 1e-6)                          # room for the value labels
    x = alt.X("start:Q", title="Score (log-odds)", scale=alt.Scale(domain=[lo - (0.04 * (hi - lo)), hi + pad], nice=False))
    bars = alt.Chart(df).mark_bar(cornerRadius=3, height={"band": 0.66}).encode(
        y=alt.Y("label:N", sort=order, title=None), x=x, x2="end:Q",
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Raises cancellation risk", "Lowers cancellation risk", "Score"],
                                                  range=[CANCEL, STAY, TOTAL]), legend=None),
        tooltip=[alt.Tooltip("label:N", title="Input"), "Contribution (log-odds):N", "Score after this step:N",
                 "Probability at this point:N"])
    text = alt.Chart(df).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(
        y=alt.Y("label:N", sort=order), x="tip:Q", text="text:N")
    base = float(df["end"].iloc[0])
    start_line = alt.Chart(pd.DataFrame({"x": [base]})).mark_rule(color=TOTAL, strokeDash=[4, 3]).encode(x="x:Q")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8492A6").encode(x="x:Q")
    return base_chart((zero + start_line + bars + text).properties(height=alt.Step(height or 30)))


def diverging_bars(df: pd.DataFrame, label: str, value: str, value_title: str, tooltip: list, height=None):
    """Horizontal bars from 0: red = pushes towards cancellation, blue = away from it."""
    df = df.copy()
    df["Direction"] = df[value].map(lambda v: "Raises cancellation risk" if v > 0 else "Lowers cancellation risk")
    df["text"] = df[value].map(lambda v: f"{v:+.2f}".replace("-", "−"))
    order = list(df[label])
    lo, hi = min(0.0, float(df[value].min())), max(0.0, float(df[value].max()))
    pad = 0.18 * max(hi - lo, 1e-6)                       # room for the value labels on both sides
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.66}).encode(
        y=alt.Y(f"{label}:N", sort=order, title=None),
        x=alt.X(f"{value}:Q", title=value_title,
                scale=alt.Scale(domain=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)], nice=False)),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises cancellation risk", "Lowers cancellation risk"],
                                                       range=[CANCEL, STAY]), legend=None),   # HTML legend: see diverging_legend
        tooltip=tooltip)
    text = alt.Chart(df).mark_text(dx=alt.ExprRef("datum['" + value + "'] > 0 ? 6 : -6"),
                                   align=alt.ExprRef("datum['" + value + "'] > 0 ? 'left' : 'right'"),
                                   color=INK, fontSize=12).encode(
        y=alt.Y(f"{label}:N", sort=order), x=f"{value}:Q", text="text:N")
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8492A6").encode(x="x:Q")
    return base_chart((bars + text + rule).properties(height=alt.Step(height or 30)))


def sidebar_model_card():
    """The deployed model's name at the bottom of the sidebar."""
    sel = load("selected")
    with st.sidebar:
        st.html(f'<div class="hs-side"><b>{escape(sel["model"])} model</b></div>')
