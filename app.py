"""
Runway Sentinel AI — Streamlit Command Center
==============================================
Aviation-grade FOD detection and autonomous hazard response.
Run with:  streamlit run app.py
Demo mode: RUNWAYGUARD_MOCK=true streamlit run app.py
"""
import os
import tempfile
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# ── Page Configuration (MUST be the very first Streamlit call) ─────────────────
st.set_page_config(
    page_title="Runway Sentinel AI",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load config (imports after set_page_config to allow st.secrets resolution) ─
from src import config  # noqa: E402

# ── Real pipeline (lazy — only loaded when USE_MOCK=False) ─────────────────────
if not config.USE_MOCK:
    from src.detection import detect_fod  # noqa: E402
    from src.risk import compute_risk_score  # noqa: E402
    from src.agent import run_agent  # noqa: E402
    from src.reporting import generate_report  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & MOCK DATA
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_SCENARIOS = [
    {
        "fod_present": True,
        "object_class": "Metal Debris",
        "risk_raw": "HIGH",
        "risk_category": "CRITICAL",
        "numeric_score": 8.74,
        "location_estimate": "Touchdown Zone Center",
        "confidence": 0.94,
        "action": "halt_traffic",
        "action_reason": "High-risk metallic FOD detected on active touchdown zone.",
        "action_detail": "All traffic halted on RWY 27L. ATC notified. SFO-OPS-2025-001 opened.",
        "report_text": None,
    },
    {
        "fod_present": True,
        "object_class": "Cone",
        "risk_raw": "MEDIUM",
        "risk_category": "MEDIUM",
        "numeric_score": 4.21,
        "location_estimate": "Runway Edge Right",
        "confidence": 0.81,
        "action": "alert_ground_crew",
        "action_reason": "Obstruction cone detected near runway edge.",
        "action_detail": "Ground crew vehicle dispatched. Priority B. ETA 4 min.",
        "report_text": None,
    },
    {
        "fod_present": True,
        "object_class": "Bird",
        "risk_raw": "MEDIUM",
        "risk_category": "MEDIUM",
        "numeric_score": 3.15,
        "location_estimate": "Threshold Left",
        "confidence": 0.76,
        "action": "dispatch_sweep_team",
        "action_reason": "Wildlife intrusion near runway threshold.",
        "action_detail": "Wildlife management team en route. Pyrotechnic hazing authorized.",
        "report_text": None,
    },
    {
        "fod_present": False,
        "object_class": "None",
        "risk_raw": "LOW",
        "risk_category": "CLEAR",
        "numeric_score": 0.0,
        "location_estimate": "N/A",
        "confidence": 0.98,
        "action": "log_only",
        "action_reason": "No FOD detected. Runway surface verified clear.",
        "action_detail": "Routine scan logged. No action required. Audit entry #RSA-0042.",
        "report_text": None,
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "scenario_idx" not in st.session_state:
    st.session_state.scenario_idx: int = 0
if "last_result" not in st.session_state:
    st.session_state.last_result: dict | None = None
if "system_status" not in st.session_state:
    st.session_state.system_status: str = "IDLE"
if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded: bool = False

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — AVIATION COMMAND CENTER DARK THEME
# ═══════════════════════════════════════════════════════════════════════════════

DARK_CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f1e !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
[data-testid="stHeader"]        { background: transparent !important; }
[data-testid="stSidebar"]       { background: #0f172a !important; }
.main .block-container          { padding: 0.5rem 1rem !important; max-width: 100% !important; }
footer, #MainMenu               { visibility: hidden !important; }

/* ── Typography overrides ─────────────────────────────────────────────────── */
.stMarkdown, .stText, p, h1, h2, h3, h4, label, span {
    color: #e2e8f0 !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]       { background: #111827; border-radius: 6px; padding: 2px; }
.stTabs [data-baseweb="tab"]            { border-radius: 4px; color: #64748b !important; font-size: 0.75rem; font-weight: 600; }
.stTabs [aria-selected="true"]          { background: #1e293b !important; color: #e2e8f0 !important; }
.stTabs [data-baseweb="tab-panel"]      { padding: 6px 0 0 0 !important; }

/* ── Metrics ──────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
[data-testid="metric-container"] label          { color: #64748b !important; font-size: 0.7rem !important; letter-spacing: 0.12em !important; }
[data-testid="stMetricValue"]                   { color: #e2e8f0 !important; font-family: 'Courier New', monospace !important; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    transition: all 0.15s ease !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ef4444, #b91c1c) !important;
    color: white !important;
    box-shadow: 0 0 16px #ef444444 !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 28px #ef4444aa !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid #334155 !important;
}

/* ── Inputs & Uploaders ───────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: #111827 !important;
    border: 1px dashed #1e2d45 !important;
    border-radius: 8px !important;
}
[data-testid="stCameraInput"] { border-radius: 8px !important; }

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"]             { background: #111827 !important; border-radius: 8px; }
.dvn-scroller                           { background: #111827 !important; }

/* ── Alerts ───────────────────────────────────────────────────────────────── */
[data-baseweb="notification"]           { background: #1e293b !important; border-radius: 8px !important; }

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr                                      { border-color: #1e2d45 !important; margin: 8px 0 !important; }

/* ─────────────────────────────────────────────────────────────────────────── */
/* COMPONENT STYLES                                                            */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Panel card */
.rs-panel {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 8px;
}
.rs-panel-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #475569;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 6px;
}
.rs-kv            { display: flex; align-items: center; padding: 5px 0; border-bottom: 1px solid #0f172a; }
.rs-kv:last-child { border-bottom: none; }
.rs-kv-label      { font-size: 0.6rem; color: #475569; letter-spacing: 0.1em; text-transform: uppercase; min-width: 120px; }
.rs-kv-value      { font-size: 0.82rem; color: #e2e8f0; font-family: 'Courier New', monospace; }

/* Confidence bar */
.conf-bar-bg      { background: #0f172a; border-radius: 3px; height: 4px; margin-top: 8px; overflow: hidden; }

/* Glowing action panels */
@keyframes glow-red {
    0%,100% { box-shadow: 0 0 8px #ef444455, 0 0 24px #ef444422; }
    50%      { box-shadow: 0 0 24px #ef4444bb, 0 0 52px #ef444455; }
}
@keyframes glow-amber {
    0%,100% { box-shadow: 0 0 8px #f59e0b44, 0 0 20px #f59e0b22; }
    50%      { box-shadow: 0 0 22px #f59e0bbb, 0 0 44px #f59e0b44; }
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.15; }
}

.action-panel-red   { background:#0a0e1a; border:1px solid #ef4444; border-radius:8px; padding:14px; animation: glow-red 2.5s ease-in-out infinite; }
.action-panel-amber { background:#0a0e1a; border:1px solid #f59e0b; border-radius:8px; padding:14px; animation: glow-amber 2.5s ease-in-out infinite; }
.action-panel-green { background:#0a0e1a; border:1px solid #10b981; border-radius:8px; padding:14px; }
.action-panel-idle  { background:#0a0e1a; border:1px solid #1e2d45; border-radius:8px; padding:14px; }

.action-title { font-size:0.65rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:10px; }
.terminal-body {
    background: #050a14;
    border-radius: 6px;
    padding: 12px;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #cbd5e1;
    line-height: 1.65;
}
.term-cmd   { font-weight: 700; font-size: 0.9rem; }
.term-label { color: #475569; font-size: 0.6rem; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 8px; }

/* Report panel */
.report-section-title { font-size:0.65rem; font-weight:700; letter-spacing:0.12em; color:#22d3ee; text-transform:uppercase; margin:8px 0 4px 0; }
.report-footer        { font-size:0.6rem; color:#475569; border-top:1px solid #1e2d45; margin-top:10px; padding-top:7px; text-align:center; font-family:'Courier New',monospace; }
.mono                 { font-family:'Courier New',monospace !important; }

/* Scenario badge grid */
.scenario-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:5px; margin:6px 0 10px 0; }
.scenario-tile { border-radius:5px; padding:6px 4px; text-align:center; }

/* System info box */
.sysinfo { margin-top:8px; padding:10px; background:#050a14; border:1px solid #1e2d45; border-radius:6px; }
.sysinfo-row { font-size:0.6rem; color:#475569; font-family:'Courier New',monospace; line-height:1.9; }

/* Section labels */
.section-label { font-size:0.6rem; letter-spacing:0.16em; color:#475569; text-transform:uppercase; font-weight:700; margin-bottom:5px; display:block; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def risk_color(category: str) -> str:
    return {
        "CLEAR":    "#10b981",
        "LOW":      "#22d3ee",
        "MEDIUM":   "#f59e0b",
        "HIGH":     "#f97316",
        "CRITICAL": "#ef4444",
    }.get(category, "#94a3b8")


def action_color(action: str) -> str:
    return {
        "halt_traffic":       "#ef4444",
        "alert_ground_crew":  "#f59e0b",
        "dispatch_sweep_team":"#f97316",
        "log_only":           "#10b981",
    }.get(action, "#94a3b8")


def action_panel_class(action: str) -> str:
    if action == "halt_traffic":
        return "action-panel-red"
    if action in ("alert_ground_crew", "dispatch_sweep_team"):
        return "action-panel-amber"
    return "action-panel-green"


def run_mock_pipeline(scenario: dict) -> dict:
    time.sleep(1.4)
    return scenario


def run_live_pipeline(image_path: str) -> dict:
    detection = detect_fod(image_path)
    risk      = compute_risk_score(detection)
    audit     = run_agent(detection, risk)
    report    = generate_report(detection, risk, audit)

    primary = (audit["actions_executed"] or [{"tool": "log_only", "args": {}, "result": {"detail": "No action."}}])[0]
    return {
        "fod_present":      detection["fod_present"],
        "object_class":     detection["object_class"],
        "risk_raw":         detection["risk_raw"],
        "risk_category":    risk["risk_category"],
        "numeric_score":    risk["numeric_score"],
        "location_estimate":detection["location_estimate"],
        "confidence":       detection["confidence"],
        "action":           primary["tool"],
        "action_reason":    primary.get("args", {}).get("reason", "Autonomous assessment threshold reached."),
        "action_detail":    primary["result"].get("detail", "Action executed."),
        "report_text":      report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_detection_panel(result: dict) -> None:
    rc       = risk_color(result["risk_category"])
    conf_pct = int(float(result["confidence"]) * 100)

    fod_badge = (
        f'<span style="background:{rc}22;color:{rc};border:1px solid {rc};'
        f'font-size:.63rem;font-weight:700;padding:3px 10px;border-radius:4px;letter-spacing:.1em;">'
        f'{"⚠ FOD DETECTED" if result["fod_present"] else "✔ RUNWAY CLEAR"}</span>'
    )
    conf_bar = (
        f'<div class="conf-bar-bg"><div style="height:4px;width:{conf_pct}%;'
        f'background:{rc};border-radius:3px;transition:width .7s;"></div></div>'
    )

    st.markdown(f"""
<div class="rs-panel">
  <div class="rs-panel-title">⬡ Vision Detection Output</div>
  <div class="rs-kv" style="padding:5px 0;">{fod_badge}</div>
  <div class="rs-kv">
    <span class="rs-kv-label">Object Class</span>
    <span class="rs-kv-value">{str(result["object_class"]).upper()}</span>
  </div>
  <div class="rs-kv">
    <span class="rs-kv-label">Risk Category</span>
    <span class="rs-kv-value" style="color:{rc};font-weight:700;font-size:1rem;">
      {result["risk_category"]}
      <span style="color:{rc}88;font-size:.7rem;"> ({result["numeric_score"]}/10)</span>
    </span>
  </div>
  <div class="rs-kv">
    <span class="rs-kv-label">Location</span>
    <span class="rs-kv-value">{str(result["location_estimate"]).upper()}</span>
  </div>
  <div class="rs-kv">
    <span class="rs-kv-label">Confidence</span>
    <span class="rs-kv-value">{conf_pct}%</span>
  </div>
  {conf_bar}
  <div class="rs-kv" style="margin-top:6px;">
    <span class="rs-kv-label">Scan Timestamp</span>
    <span class="rs-kv-value" style="font-size:.7rem;">{now_ts()}</span>
  </div>
</div>""", unsafe_allow_html=True)


def render_action_panel(result: dict | None) -> None:
    if result is None:
        st.markdown("""
<div class="action-panel-idle">
  <div class="action-title" style="color:#1e2d45;">⚡ Agentic Decision Engine</div>
  <div class="terminal-body">
    <span style="color:#1e2d45;">$ AWAITING DETECTION RESULT...</span>
  </div>
</div>""", unsafe_allow_html=True)
        return

    action    = result["action"]
    ac        = action_color(action)
    panel_cls = action_panel_class(action)
    icon = {"halt_traffic":"🚨","alert_ground_crew":"⚠️","dispatch_sweep_team":"🚜","log_only":"📋"}.get(action, "⚡")

    st.markdown(f"""
<div class="{panel_cls}">
  <div class="action-title" style="color:{ac};">⚡ Agentic Decision Engine</div>
  <div class="terminal-body">
    <div>
      <span style="color:#475569;">$&nbsp;</span>
      <span class="term-cmd" style="color:{ac};">{icon} [ACTION EXECUTED] {action}()</span>
    </div>
    <hr style="border:none;border-top:1px solid {ac}44;margin:8px 0;">
    <div class="term-label">Reason:</div>
    <div>{result["action_reason"]}</div>
    <div class="term-label" style="margin-top:8px;">Outcome:</div>
    <div style="color:#94a3b8;">{result["action_detail"]}</div>
    <div class="term-label" style="margin-top:8px;">Timestamp:</div>
    <div style="color:#64748b;">{now_ts()}</div>
  </div>
</div>""", unsafe_allow_html=True)


def render_report_panel(result: dict | None) -> None:
    if result is None:
        st.markdown("""
<div class="rs-panel" style="text-align:center;padding:28px 14px;">
  <div style="color:#1e2d45;font-size:.77rem;font-family:'Courier New',monospace;">
    NO INCIDENT DATA<br>
    <span style="font-size:.7rem;">Run a scan to generate a report</span>
  </div>
</div>""", unsafe_allow_html=True)
        return

    rc          = risk_color(result["risk_category"])
    report_text = result.get("report_text")

    if report_text:
        # Live mode: show LLM-generated structured report
        body = f'<pre style="color:#cbd5e1;font-size:.72rem;white-space:pre-wrap;font-family:\'Courier New\',monospace;line-height:1.5;">{report_text}</pre>'
    else:
        # Mock mode: inline synthesized summary
        status_icon = "🔴" if result["fod_present"] else "🟢"
        status_txt  = "FOD CONFIRMED — RUNWAY COMPROMISED" if result["fod_present"] else "RUNWAY CLEAR — NO ACTION REQUIRED"
        body = f"""
<div class="report-section-title">▸ Status</div>
<div class="mono" style="font-size:.77rem;color:#e2e8f0;">{status_icon} {status_txt}</div>

<div class="report-section-title">▸ Detection Summary</div>
<div style="font-size:.77rem;color:#94a3b8;">
  Object <span class="mono" style="color:#e2e8f0;">{result["object_class"]}</span>
  at <span class="mono" style="color:#e2e8f0;">{result["location_estimate"]}</span> —
  <span class="mono" style="color:{rc};">{int(float(result["confidence"])*100)}%</span> confidence.
  Vision risk: <span style="color:{rc};font-weight:600;">{result["risk_raw"]}</span>.
</div>

<div class="report-section-title">▸ Risk Assessment</div>
<div style="font-size:.77rem;color:#94a3b8;">
  Score: <span class="mono" style="color:{rc};font-weight:700;">{result["numeric_score"]}/10.0</span> —
  Category: <span style="color:{rc};font-weight:700;">{result["risk_category"]}</span>.
  Formula: <span class="mono" style="font-size:.7rem;color:#475569;">obj_weight × loc_centrality × confidence</span>
  <span style="color:#334155;">(Zero LLM — deterministic.)</span>
</div>

<div class="report-section-title">▸ Autonomous Action</div>
<div class="mono" style="font-size:.77rem;color:#cbd5e1;">{result["action"]}() — {result["action_detail"]}</div>"""

    st.markdown(f"""
<div class="rs-panel">
  <div class="rs-panel-title">📋 Incident Report</div>
  {body}
  <div class="report-footer">Runway Sentinel AI · {now_ts()} · UNCLASSIFIED // OFFICIAL USE ONLY</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — HEADER
# ═══════════════════════════════════════════════════════════════════════════════

status_colors = {"IDLE":"#10b981","ANALYZING":"#f59e0b","ALERT":"#ef4444","CLEAR":"#10b981"}
status_icons  = {"IDLE":"●","ANALYZING":"◉","ALERT":"▲","CLEAR":"✔"}
scolor        = status_colors.get(st.session_state.system_status, "#94a3b8")
sicon         = status_icons.get(st.session_state.system_status, "●")
mode_badge    = "MOCK / DEMO" if config.USE_MOCK else "LIVE DEPLOYED"
mode_color    = "#f59e0b" if config.USE_MOCK else "#10b981"

st.markdown(f"""
<div style="background:#111827;border-bottom:2px solid #1e2d45;padding:10px 20px;
     display:flex;align-items:center;gap:14px;margin-bottom:12px;border-radius:0 0 6px 6px;">

  <!-- Logo -->
  <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
    <span style="background:#ef4444;color:white;border-radius:5px;padding:4px 9px;
                 font-size:.9rem;font-weight:900;letter-spacing:.05em;">RS</span>
    <div>
      <div style="font-size:1.05rem;font-weight:800;color:#e2e8f0;line-height:1.1;">
        Runway Sentinel<span style="color:#ef4444;">AI</span>
      </div>
      <div style="font-size:.58rem;color:#64748b;letter-spacing:.18em;font-weight:400;">
        FOD DETECTION &amp; RESPONSE SYSTEM
      </div>
    </div>
  </div>

  <div style="width:1px;height:32px;background:#1e2d45;margin:0 4px;"></div>

  <!-- Capability badges -->
  <span style="font-size:.6rem;font-family:'Courier New',monospace;border:1px solid {mode_color};
               color:{mode_color};border-radius:4px;padding:2px 8px;letter-spacing:.08em;white-space:nowrap;">
    {mode_badge}
  </span>
  <span style="font-size:.6rem;font-family:'Courier New',monospace;border:1px solid #22d3ee;
               color:#22d3ee;border-radius:4px;padding:2px 8px;letter-spacing:.08em;white-space:nowrap;">
    GROQ VISION · LLAMA-4
  </span>
  <span style="font-size:.6rem;font-family:'Courier New',monospace;border:1px solid #a78bfa;
               color:#a78bfa;border-radius:4px;padding:2px 8px;letter-spacing:.08em;white-space:nowrap;">
    AGENTIC LAYER ACTIVE
  </span>
  <span style="font-size:.6rem;font-family:'Courier New',monospace;border:1px solid #0ea5e9;
               color:#0ea5e9;border-radius:4px;padding:2px 8px;letter-spacing:.08em;white-space:nowrap;">
    RISK ENGINE v2
  </span>

  <!-- Status + Timestamp (right-aligned) -->
  <div style="margin-left:auto;display:flex;align-items:center;gap:10px;flex-shrink:0;">
    <div style="display:inline-flex;align-items:center;gap:6px;background:#0a0f1e;
                border:1px solid #1e2d45;border-radius:20px;padding:4px 12px;">
      <span style="color:{scolor};font-size:1.1rem;animation:{'pulse-dot 1.2s ease-in-out infinite' if st.session_state.system_status == 'ANALYZING' else 'none'};">
        {sicon}
      </span>
      <span style="color:{scolor};font-weight:700;letter-spacing:.12em;font-size:.82rem;
                   font-family:'Courier New',monospace;">
        {st.session_state.system_status}
      </span>
    </div>
    <span style="font-family:'Courier New',monospace;font-size:.7rem;color:#475569;white-space:nowrap;">
      {now_ts()}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — THREE COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

col_input, col_analysis, col_report = st.columns([1, 1.35, 1.35], gap="small")

# ──────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Image Input
# ──────────────────────────────────────────────────────────────────────────────
with col_input:
    st.markdown('<span class="section-label">✈ Runway Image Input</span>', unsafe_allow_html=True)

    tab_upload, tab_cam = st.tabs(["📁  Upload Image", "📷  Live Camera"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Runway image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

    with tab_cam:
        camera_img = st.camera_input("Capture from webcam", label_visibility="collapsed")

    # ── Demo Scenario Selector ─────────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="section-label">⬡ Demo Scenarios</span>', unsafe_allow_html=True)

    scenario_defs = [
        ("#ef4444", "CRIT", "Metal"),
        ("#f59e0b", "MED",  "Cone"),
        ("#f97316", "MED",  "Bird"),
        ("#10b981", "CLR",  "None"),
    ]
    tiles_html = '<div class="scenario-grid">'
    for c, badge, label in scenario_defs:
        tiles_html += (
            f'<div class="scenario-tile" style="background:{c}12;border:1px solid {c}44;">'
            f'<div style="color:{c};font-size:.62rem;font-weight:700;font-family:Courier New,monospace;">{badge}</div>'
            f'<div style="color:#475569;font-size:.55rem;">{label}</div>'
            f'</div>'
        )
    tiles_html += "</div>"
    st.markdown(tiles_html, unsafe_allow_html=True)

    if st.button("▶  LOAD NEXT SCENARIO", use_container_width=True, type="secondary"):
        st.session_state.scenario_idx = (st.session_state.scenario_idx + 1) % len(MOCK_SCENARIOS)
        st.session_state.sample_loaded = True
        st.rerun()

    current_scenario_name = ["CRITICAL — Metal Debris", "MEDIUM — Cone", "MEDIUM — Bird", "CLEAR"][
        st.session_state.scenario_idx % 4
    ]
    if st.session_state.sample_loaded:
        rc_badge = risk_color(MOCK_SCENARIOS[st.session_state.scenario_idx % 4]["risk_category"])
        st.markdown(
            f'<div style="text-align:center;font-size:.6rem;font-family:Courier New,monospace;'
            f'color:{rc_badge};margin-top:3px;letter-spacing:.08em;">◈ {current_scenario_name}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Main Scan Button ───────────────────────────────────────────────────────
    scan_clicked = st.button("⟳  INITIATE SCAN", use_container_width=True, type="primary")

    # ── System Info ────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="sysinfo">
  <div class="sysinfo-row">
    MODE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {'MOCK (synthetic)' if config.USE_MOCK else 'LIVE (Groq API)'}<br>
    PIPELINE&nbsp; CV ⟶ RISK ⟶ AGENT ⟶ REPORT<br>
    VISION&nbsp;&nbsp;&nbsp; {config.VISION_MODEL.split('/')[-1]}<br>
    AGENT&nbsp;&nbsp;&nbsp;&nbsp; {config.AGENT_MODEL}<br>
    RISK&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Deterministic (Zero LLM)<br>
    SCANS&nbsp;&nbsp;&nbsp;&nbsp; {len(st.session_state.history):04d} logged
  </div>
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MIDDLE COLUMN — AI Pipeline Output
# ──────────────────────────────────────────────────────────────────────────────
with col_analysis:
    st.markdown('<span class="section-label">🧠 AI Pipeline Output</span>', unsafe_allow_html=True)

    # ── Process scan click ─────────────────────────────────────────────────────
# ── Process scan click ─────────────────────────────────────────────────────
    if scan_clicked:
        has_upload = uploaded_file is not None
        has_camera = camera_img is not None
        has_sample = st.session_state.sample_loaded

        if not has_upload and not has_camera and not has_sample:
            st.warning("⚠ No input provided. Upload an image, capture from camera, or load a demo scenario first.")
        else:
            st.session_state.system_status = "ANALYZING"
            # Clear previous error on new scan attempt
            st.session_state.last_error = None
            
            with st.spinner("🔍 Running FOD Detection Pipeline…"):
                try:
                    if config.USE_MOCK or (has_sample and not (has_upload or has_camera)):
                        # Demo / MOCK mode: use synthetic scenario
                        idx = st.session_state.scenario_idx % len(MOCK_SCENARIOS)
                        result = run_mock_pipeline(MOCK_SCENARIOS[idx])
                    else:
                        # Live mode: write upload to temp file for detection
                        image_source = uploaded_file or camera_img
                        suffix = "." + (getattr(image_source, "name", "tmp.jpg").split(".")[-1])
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(image_source.getvalue())
                            tmp_path = tmp.name
                        try:
                            result = run_live_pipeline(tmp_path)
                        finally:
                            os.unlink(tmp_path)

                    ts = now_ts()
                    st.session_state.last_result   = result
                    st.session_state.system_status = "ALERT" if result["fod_present"] else "CLEAR"
                    st.session_state.history.insert(0, {
                        "Timestamp (UTC)": ts,
                        "Object":          result["object_class"],
                        "Risk Score":      f"{result['risk_category']} ({result['numeric_score']})",
                        "Action":          f"{result['action']}()",
                        "Location":        result["location_estimate"],
                    })
                except Exception as exc:
                    # Save error to state so it survives st.rerun()
                    st.session_state.last_error = str(exc)
                    st.session_state.system_status = "IDLE"

            st.rerun()

    # ── Display Error Message if Scan Failed ───────────────────────────────────
    if getattr(st.session_state, "last_error", None):
        st.error(f"❌ Scan Failed: {st.session_state.last_error}")

    # ── Detection results ──────────────────────────────────────────────────────
    if st.session_state.last_result:
        render_detection_panel(st.session_state.last_result)
    else:
        st.markdown("""
<div class="rs-panel" style="text-align:center;padding:40px 14px;">
  <div style="font-size:2.5rem;opacity:.12;">📡</div>
  <div style="color:#1e2d45;font-size:.77rem;margin-top:10px;font-family:'Courier New',monospace;letter-spacing:.1em;">
    AWAITING INPUT — SYSTEM STANDBY
  </div>
</div>""", unsafe_allow_html=True)

    # ── Agentic layer panel ────────────────────────────────────────────────────
    st.markdown(
        '<span class="section-label" style="margin-top:6px;display:block;">⚡ Agentic Decision Layer</span>',
        unsafe_allow_html=True,
    )
    render_action_panel(st.session_state.last_result)

# ──────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Incident Report + History
# ──────────────────────────────────────────────────────────────────────────────
with col_report:
    st.markdown('<span class="section-label">📡 Synthesis & Audit Log</span>', unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────────
    r = st.session_state.last_result
    m1, m2 = st.columns(2)
    with m1:
        st.metric(
            label="⬡ RISK SCORE",
            value=f"{r['numeric_score']:.2f}" if r else "—",
            delta=f"{r['risk_category']}" if r else None,
        )
    with m2:
        st.metric(
            label="⬡ CONFIDENCE",
            value=f"{int(float(r['confidence'])*100)}%" if r else "—",
            delta="Groq Vision" if r else None,
        )

    # ── Incident report ────────────────────────────────────────────────────────
    render_report_panel(st.session_state.last_result)

    # ── History log ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="section-label">📜 Scan History Log</span>', unsafe_allow_html=True)

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history[:10])
        st.dataframe(df, use_container_width=True, hide_index=True, height=200)

        if st.button("🗑  Clear History", use_container_width=True, type="secondary"):
            st.session_state.history       = []
            st.session_state.last_result   = None
            st.session_state.system_status = "IDLE"
            st.rerun()
    else:
        st.markdown("""
<div class="rs-panel" style="text-align:center;padding:16px;color:#1e2d45;
     font-size:.77rem;font-family:'Courier New',monospace;letter-spacing:.1em;">
  NO HISTORY — RUN FIRST SCAN
</div>""", unsafe_allow_html=True)
