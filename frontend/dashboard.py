"""
frontend/dashboard.py
=====================
AI Workforce Intelligence Platform — Streamlit Dashboard
Calls the live FastAPI backend (http://localhost:8000) exclusively.
No direct reads from data/processed/*.csv files.
"""

import math
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Constants ────────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

# Design-system colour tokens
C_BG        = "#F7F7F5"
C_SURFACE   = "#FFFFFF"
C_INK       = "#1C1F26"
C_MUTED     = "#5B6270"
C_DIVIDER   = "#DDD9D0"
C_HIGH      = "#B3492D"
C_MEDIUM    = "#B98A2E"
C_LOW       = "#3F6E52"
C_ACCENT    = "#2B4C6F"


# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Workforce Intelligence Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Base reset ──────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {C_INK};
    background-color: {C_BG};
}}

/* ── Headings use serif ──────────────────────────────────────────── */
h1, h2, h3, .display-title {{
    font-family: 'Source Serif 4', Lora, Georgia, serif;
    font-weight: 600;
    color: {C_INK};
    letter-spacing: -0.01em;
}}

/* ── Remove Streamlit default padding / decoration ───────────────── */
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
header[data-testid="stHeader"] {{ background: {C_BG}; box-shadow: none; border-bottom: 1px solid {C_DIVIDER}; }}
[data-testid="stSidebar"] {{ background: {C_SURFACE}; border-right: 1px solid {C_DIVIDER}; }}
[data-testid="stSidebar"] .css-1d391kg {{ padding-top: 1.5rem; }}
div[data-testid="metric-container"] {{
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    border-radius: 2px;
    padding: 1rem 1.2rem 0.8rem;
}}
div[data-testid="metric-container"] label {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: {C_MUTED};
    text-transform: none;
}}
div[data-testid="metric-container"] [data-testid="metric-value"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: {C_INK};
}}

/* ── Hairline section divider ────────────────────────────────────── */
.section-divider {{
    border: none;
    border-top: 1px solid {C_DIVIDER};
    margin: 2rem 0 1.5rem;
}}

/* ── Section heading ─────────────────────────────────────────────── */
.section-heading {{
    font-family: 'Source Serif 4', Lora, Georgia, serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: {C_INK};
    margin: 0 0 0.75rem;
}}

/* ── Risk-severity left-edge device ─────────────────────────────── */
.risk-row {{
    padding: 0.55rem 0.8rem;
    margin: 0.2rem 0;
    border-left: 4px solid {C_DIVIDER};
    background: {C_SURFACE};
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.88rem;
}}
.risk-HIGH  {{ border-left-color: {C_HIGH};   }}
.risk-MEDIUM {{ border-left-color: {C_MEDIUM}; }}
.risk-LOW   {{ border-left-color: {C_LOW};    }}
.risk-NA    {{ border-left-color: {C_DIVIDER}; }}
.risk-label-HIGH   {{ color: {C_HIGH};   font-weight: 500; }}
.risk-label-MEDIUM {{ color: {C_MEDIUM}; font-weight: 500; }}
.risk-label-LOW    {{ color: {C_LOW};    font-weight: 500; }}

/* ── Data-provenance banner ──────────────────────────────────────── */
.provenance-banner {{
    background: {C_SURFACE};
    border-left: 4px solid {C_ACCENT};
    padding: 0.65rem 0.9rem;
    margin: 0 0 0.75rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    color: {C_MUTED};
    line-height: 1.5;
}}

/* ── Engagement caption ──────────────────────────────────────────── */
.eng-caption {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.74rem;
    color: {C_MUTED};
    margin-top: 0.15rem;
    line-height: 1.4;
}}

/* ── Drill-down field rows ───────────────────────────────────────── */
.field-label {{
    font-size: 0.72rem;
    font-weight: 500;
    color: {C_MUTED};
    margin-bottom: 0.05rem;
    text-transform: none;
}}
.field-value {{
    font-size: 0.92rem;
    color: {C_INK};
    margin-bottom: 0.6rem;
}}
.null-value {{
    font-size: 0.88rem;
    color: {C_MUTED};
    font-style: italic;
}}

/* ── Error / not-found messages ──────────────────────────────────── */
.not-found-msg {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.88rem;
    color: {C_MUTED};
    padding: 0.6rem 0;
}}
.backend-error {{
    background: {C_SURFACE};
    border-left: 4px solid {C_HIGH};
    padding: 0.65rem 0.9rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.88rem;
    color: {C_INK};
}}

/* ── Page number selector alignment ─────────────────────────────── */
.page-nav {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    color: {C_MUTED};
}}

/* ── Top Bar Navigation Toggle ───────────────────────────────────── */
.nav-toggle-container {{
    margin-top: 0.85rem;
    margin-bottom: 0.5rem;
    padding: 0 0.5rem;
}}
div[data-testid="stRadio"] {{
    margin: 0;
    padding: 0;
    overflow: visible !important;
}}
div[data-testid="stRadio"] > div[role="radiogroup"] {{
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 2.2rem !important;
    flex-wrap: wrap !important;
    overflow: visible !important;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{
    display: inline-flex !important;
    align-items: center !important;
    vertical-align: middle !important;
    cursor: pointer !important;
    padding: 0.4rem 0.85rem 0.4rem 0.4rem !important;
    border-radius: 4px !important;
    transition: all 0.15s ease-in-out !important;
    border-bottom: 2px solid transparent !important;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div {{
    display: flex !important;
    align-items: center !important;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"] p {{
    margin: 0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: {C_MUTED} !important;
    line-height: 1.2 !important;
    font-weight: 500 !important;
}}
/* Active selection indicator: bold text + accent color #2B4C6F + bottom highlight */
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
    border-bottom: 2px solid {C_ACCENT} !important;
    background-color: rgba(43, 76, 111, 0.05) !important;
}}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {{
    color: {C_ACCENT} !important;
    font-weight: 600 !important;
}}

/* ── Employee Self-Service (My Profile) Styling ──────────────────── */
.profile-panel {{
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    border-left: 4px solid {C_ACCENT};
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    font-family: 'IBM Plex Sans', sans-serif;
}}
.profile-recs-item {{
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    border-left: 4px solid {C_ACCENT};
    padding: 0.75rem 1rem;
    margin: 0.45rem 0;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.88rem;
    color: {C_INK};
    line-height: 1.5;
}}

/* ── Platform Assistant (RAG) Styling ───────────────────────────── */
.rag-panel {{
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    border-left: 4px solid {C_ACCENT};
    padding: 1.25rem 1.4rem;
    margin: 1rem 0;
    font-family: 'IBM Plex Sans', sans-serif;
}}
.rag-refusal {{
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    border-left: 4px solid {C_MEDIUM};
    padding: 1.1rem 1.3rem;
    margin: 1rem 0;
    font-family: 'IBM Plex Sans', sans-serif;
}}
.rag-source-box {{
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    padding: 0.85rem 1rem;
    margin: 0.5rem 0;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
}}

/* ── Enterprise AI Agents Styling ───────────────────────────────── */
.agent-panel {{
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    border-left: 4px solid {C_ACCENT};
    padding: 1.25rem 1.4rem;
    margin: 1rem 0;
    font-family: 'IBM Plex Sans', sans-serif;
}}
.agent-refusal {{
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    border-left: 4px solid {C_MEDIUM};
    padding: 1.1rem 1.3rem;
    margin: 1rem 0;
    font-family: 'IBM Plex Sans', sans-serif;
}}
.agent-routing-banner {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: {C_SURFACE};
    border: 1px solid {C_DIVIDER};
    padding: 0.75rem 1rem;
    margin-bottom: 0.85rem;
    border-radius: 2px;
}}
.agent-badge {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 2px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
_BACKEND_ERROR = False
_EMBEDDED_CLIENT = None

def _get_client():
    """Returns in-process FastAPI TestClient adapter when running without external Uvicorn."""
    global _EMBEDDED_CLIENT
    if _EMBEDDED_CLIENT is None:
        try:
            from fastapi.testclient import TestClient
            from app.main import app
            _EMBEDDED_CLIENT = TestClient(app)
        except Exception:
            _EMBEDDED_CLIENT = False
    return _EMBEDDED_CLIENT if _EMBEDDED_CLIENT is not False else None


def _get(endpoint: str, params: dict = None):
    """GET from FastAPI (via live HTTP or in-process ASGI adapter)."""
    global _BACKEND_ERROR
    # 1. Attempt live HTTP request if external server is reachable
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=2)
        r.raise_for_status()
        _BACKEND_ERROR = False
        return r.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"__status": 404, "detail": e.response.json().get("detail", "Not found.")}
        _BACKEND_ERROR = True
        return None
    except Exception:
        pass

    # 2. In-process FastAPI ASGI adapter for Streamlit Community Cloud
    client = _get_client()
    if client is not None:
        try:
            r = client.get(endpoint, params=params)
            if r.status_code == 200:
                _BACKEND_ERROR = False
                return r.json()
            elif r.status_code == 404:
                return {"__status": 404, "detail": r.json().get("detail", "Not found.")}
            _BACKEND_ERROR = True
            return None
        except Exception:
            _BACKEND_ERROR = True
            return None

    _BACKEND_ERROR = True
    return None


def _post(endpoint: str, json_data: dict = None):
    """POST to FastAPI (via live HTTP or in-process ASGI adapter)."""
    global _BACKEND_ERROR
    # 1. Attempt live HTTP request if external server is reachable
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=30)
        r.raise_for_status()
        _BACKEND_ERROR = False
        return r.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass
    except requests.exceptions.HTTPError as e:
        detail = "HTTP error"
        if e.response is not None:
            try:
                detail = e.response.json().get("detail", detail)
            except Exception:
                detail = str(e)
        return {"__error": detail}
    except Exception:
        pass

    # 2. In-process FastAPI ASGI adapter for Streamlit Community Cloud
    client = _get_client()
    if client is not None:
        try:
            r = client.post(endpoint, json=json_data)
            if r.status_code == 200:
                _BACKEND_ERROR = False
                return r.json()
            detail = "HTTP error"
            try:
                detail = r.json().get("detail", detail)
            except Exception:
                pass
            return {"__error": detail}
        except Exception as e:
            return {"__error": str(e)}

    return {"__error": "In-process backend unavailable"}


def _severity_color(sev: str) -> str:
    return {
        "HIGH": C_HIGH,
        "MEDIUM": C_MEDIUM,
        "LOW": C_LOW,
    }.get(str(sev).upper(), C_DIVIDER)


def _risk_css_class(sev: str) -> str:
    s = str(sev).upper()
    if s in ("HIGH", "MEDIUM", "LOW"):
        return f"risk-{s}"
    return "risk-NA"


# ── Backend-not-reachable guard ───────────────────────────────────────────────
summary = _get("/dashboard/summary")
if _BACKEND_ERROR or summary is None:
    st.markdown(
        '<div class="backend-error">'
        '⚠ Backend not reachable — start the API with '
        '<code>uvicorn app.main:app</code> and refresh.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Mode Toggle at Top of App ─────────────────────────────────────────────────
nav_container = st.container()
with nav_container:
    st.markdown('<div class="nav-toggle-container">', unsafe_allow_html=True)
    portal_mode = st.radio(
        "Select Portal View",
        options=["HR / Manager View", "My Profile", "Platform Assistant (RAG)", "Enterprise AI Agents"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="portal_mode_toggle",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<hr style="border:none;border-top:1px solid {C_DIVIDER};margin:0.25rem 0 1.5rem 0;">', unsafe_allow_html=True)


# ==============================================================================
# VIEW 1: HR / MANAGER VIEW (Default Dashboard)
# ==============================================================================
def render_hr_manager_view():
    # ── 1. Title ──────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="display-title" style="font-size:2rem;margin-bottom:0.15rem;">'
        'AI Workforce Intelligence Platform</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.82rem;'
        f'color:{C_MUTED};margin-top:0;margin-bottom:1.5rem;">'
        f'Live data from API · All engagement figures based on survey respondents only'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── 2. KPI row ────────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Total Employees", f"{summary['total_employees']:,}")

    with kpi2:
        st.metric("High Risk Count", f"{summary['high_risk_count']:,}")

    with kpi3:
        st.metric("High Risk %", f"{summary['high_risk_percentage']:.1f}%")

    with kpi4:
        avg_eng = summary.get("average_engagement")
        st.metric("Average Engagement", f"{avg_eng:.2f} / 5.0" if avg_eng else "—")
        st.markdown(
            f'<div class="eng-caption">{summary.get("engagement_coverage_note", "")}</div>',
            unsafe_allow_html=True,
        )

    # ── Fetch department data (used in both sidebar and chart) ────────────────
    dept_data = _get("/dashboard/attrition-by-department") or []
    dept_names = ["All"] + [d["department"] for d in dept_data]

    # ── 3. Sidebar: Department filter ─────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<p style="font-family:\'Source Serif 4\',serif;font-size:1rem;'
            f'font-weight:600;color:{C_INK};margin-bottom:0.5rem;">Filters</p>',
            unsafe_allow_html=True,
        )
        selected_dept = st.selectbox("Department", options=dept_names, index=0)
        st.markdown(
            f'<hr style="border:none;border-top:1px solid {C_DIVIDER};margin:1rem 0;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.75rem;'
            f'color:{C_MUTED};line-height:1.5;">'
            f'Risk threshold: ≥ 0.40 → High<br>'
            f'Skill gap: ≥ 100 missing → High, ≥ 50 → Medium'
            f'</p>',
            unsafe_allow_html=True,
        )

    # ── Zone spacer + divider ────────────────────────────────────────────────
    st.markdown('<div style="margin:2.5rem 0"></div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 4. Attrition Risk by Department ──────────────────────────────────────
    st.markdown('<p class="section-heading">Attrition Risk by Department</p>', unsafe_allow_html=True)

    if dept_data:
        filtered_dept = (
            dept_data if selected_dept == "All"
            else [d for d in dept_data if d["department"] == selected_dept]
        )

        bar_colors = []
        for d in filtered_dept:
            pct = d["high_risk_percentage"]
            if pct >= 50:
                bar_colors.append(C_HIGH)
            elif pct >= 35:
                bar_colors.append(C_MEDIUM)
            else:
                bar_colors.append(C_LOW)

        dept_labels = [d["department"] for d in filtered_dept]
        high_pcts   = [d["high_risk_percentage"] for d in filtered_dept]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="High Risk %",
            x=high_pcts,
            y=dept_labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p:.1f}%  ({int(d['high_risk_count'])} employees)"
                  for p, d in zip(high_pcts, filtered_dept)],
            textposition="inside",
            textfont=dict(family="IBM Plex Sans", size=12, color="#FFFFFF"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "High risk: %{x:.1f}% (%{customdata[0]} employees)<br>"
                "Total: %{customdata[1]}<extra></extra>"
            ),
            customdata=[[d["high_risk_count"], d["total_employees"]] for d in filtered_dept],
        ))

        fig.update_layout(
            paper_bgcolor=C_BG,
            plot_bgcolor=C_SURFACE,
            font=dict(family="IBM Plex Sans", color=C_INK),
            margin=dict(l=0, r=20, t=10, b=10),
            height=max(120, 80 * len(filtered_dept)),
            showlegend=False,
            xaxis=dict(
                title="High Risk %",
                range=[0, 100],
                ticksuffix="%",
                gridcolor=C_DIVIDER,
                linecolor=C_DIVIDER,
            ),
            yaxis=dict(
                autorange="reversed",
                gridcolor=C_DIVIDER,
                linecolor=C_DIVIDER,
            ),
            bargap=0.35,
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.markdown('<p class="not-found-msg">No department data available.</p>', unsafe_allow_html=True)

    # ── Zone spacer + divider ────────────────────────────────────────────────
    st.markdown('<div style="margin:2.5rem 0"></div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5. Critical Organisation Skill Gaps ──────────────────────────────────
    st.markdown('<p class="section-heading">Critical Organisation Skill Gaps</p>', unsafe_allow_html=True)

    SKILL_GAPS_PROVENANCE_NOTICE = (
        "Data provenance -- This section is currently running on simulated skill data (see docs/model_card.md) "
        "while we await a real per-employee skills source (HRIS export, LMS records, or survey). "
        "Treat these figures as provisional, not a validated basis for individual HR decisions. "
        "This section will be refreshed once real skill data is available."
    )

    st.markdown(
        f'<div class="provenance-banner">'
        f'<strong style="color:{C_ACCENT};font-size:0.78rem;">Data provenance</strong> &mdash; '
        f'This section is currently running on simulated skill data (see docs/model_card.md) while we await '
        f'a real per-employee skills source (HRIS export, LMS records, or survey). Treat these figures as provisional, '
        f'not a validated basis for individual HR decisions. This section will be refreshed once real skill data is available.'
        f'</div>',
        unsafe_allow_html=True,
    )

    skill_gaps = _get("/dashboard/skill-gaps") or []

    if skill_gaps:
        sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        skill_gaps_sorted = sorted(skill_gaps, key=lambda x: sev_order.get(x.get("severity", "LOW"), 9))

        sg_by_count = sorted(skill_gaps, key=lambda x: x.get("total_missing_count", 0))
        chart_labels  = [sg["skill_name"] for sg in sg_by_count]
        chart_counts  = [sg["total_missing_count"] for sg in sg_by_count]
        chart_colors  = [_severity_color(sg.get("severity", "LOW")) for sg in sg_by_count]
        chart_sevs    = [sg.get("severity", "LOW") for sg in sg_by_count]

        fig_sg = go.Figure()
        fig_sg.add_trace(go.Bar(
            name="Employees missing skill",
            x=chart_counts,
            y=chart_labels,
            orientation="h",
            marker_color=chart_colors,
            text=[f"{c}" for c in chart_counts],
            textposition="outside",
            textfont=dict(family="IBM Plex Sans", size=10, color=C_MUTED),
            hovertemplate="<b>%{y}</b><br>%{x} employees missing · %{customdata}<extra></extra>",
            customdata=chart_sevs,
        ))
        fig_sg.update_layout(
            paper_bgcolor=C_BG,
            plot_bgcolor=C_SURFACE,
            font=dict(family="IBM Plex Sans", color=C_INK),
            margin=dict(l=0, r=60, t=8, b=10),
            height=max(400, 22 * len(skill_gaps)),
            showlegend=False,
            xaxis=dict(
                title="Employees missing skill",
                gridcolor=C_DIVIDER,
                linecolor=C_DIVIDER,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                linecolor=C_DIVIDER,
                tickfont=dict(size=11),
            ),
            bargap=0.28,
        )
        st.plotly_chart(fig_sg, width='stretch')

        def _render_skill_band(items: list):
            for sg in items:
                sev = sg.get("severity", "LOW")
                css = _risk_css_class(sev)
                concentrated_flag = " · <em style='color:#999'>role-concentrated</em>" if sg.get("is_role_concentrated") else ""
                raw_roles = sg.get("top_affected_roles", "")

                role_parts = [r.strip() for r in raw_roles.split(",") if r.strip()]
                top3 = role_parts[:3]
                extra = role_parts[3:]
                top3_str = ", ".join(top3)

                st.markdown(
                    f'<div class="risk-row {css}">'
                    f'<span style="font-weight:500;color:{C_INK};">{sg["skill_name"]}</span>'
                    f'&nbsp;&nbsp;<span style="color:{_severity_color(sev)};font-weight:500;">{sev}</span>'
                    f'&nbsp;·&nbsp;<span style="color:{C_MUTED};">{sg["total_missing_count"]} employees missing</span>'
                    f'{concentrated_flag}'
                    f'<br><span style="font-size:0.78rem;color:{C_MUTED};">{top3_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if extra:
                    with st.popover(f"+{len(extra)} more roles", width='content'):
                        st.markdown(
                            f'<div style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.82rem;'
                            f'color:{C_MUTED};line-height:1.8;">'
                            + "".join(f"<div>{r}</div>" for r in extra)
                            + "</div>",
                            unsafe_allow_html=True,
                        )

        high_items   = [sg for sg in skill_gaps_sorted if sg.get("severity") == "HIGH"]
        medium_items = [sg for sg in skill_gaps_sorted if sg.get("severity") == "MEDIUM"]
        low_items    = [sg for sg in skill_gaps_sorted if sg.get("severity") == "LOW"]

        with st.expander(
            f"High severity ({len(high_items)} skills)",
            expanded=True,
            key="sg_expander_high",
        ):
            _render_skill_band(high_items)

        with st.expander(
            f"Medium severity ({len(medium_items)} skills)",
            expanded=False,
            key="sg_expander_medium",
        ):
            _render_skill_band(medium_items)

        with st.expander(
            f"Low severity ({len(low_items)} skills)",
            expanded=False,
            key="sg_expander_low",
        ):
            _render_skill_band(low_items)

    else:
        st.markdown('<p class="not-found-msg">No skill gap data available.</p>', unsafe_allow_html=True)

    # ── Zone spacer + divider ────────────────────────────────────────────────
    st.markdown('<div style="margin:2.5rem 0"></div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 6. AI Upskilling Recommendations ─────────────────────────────────────
    st.markdown('<p class="section-heading">AI Upskilling Recommendations</p>', unsafe_allow_html=True)

    rec_params = {}
    if selected_dept != "All":
        rec_params["department"] = selected_dept

    recs = _get("/dashboard/recommendations", params=rec_params) or []

    if recs:
        df_recs = pd.DataFrame(recs)
        df_recs = df_recs.rename(columns={
            "EmployeeNumber":      "Emp #",
            "JobRole":             "Job Role",
            "Department":          "Department",
            "RiskLevel":           "Risk Level",
            "SkillGapSeverity":    "Skill Gap",
            "Top3Recommendations": "Top 3 Recommendations",
        })
        df_recs = df_recs[[c for c in ["Emp #", "Job Role", "Department", "Risk Level", "Skill Gap", "Top 3 Recommendations"] if c in df_recs.columns]]

        def _sev_icon(val: str) -> str:
            return {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}.get(str(val).upper(), val)

        if "Risk Level" in df_recs.columns:
            df_recs["Risk Level"] = df_recs["Risk Level"].apply(_sev_icon)
        if "Skill Gap" in df_recs.columns:
            df_recs["Skill Gap"] = df_recs["Skill Gap"].apply(
                lambda v: _sev_icon(v) if v in ("HIGH", "MEDIUM", "LOW") else v
            )

        st.markdown(
            f'<span style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.78rem;color:{C_MUTED};">'
            f'{len(df_recs)} employees · sortable by any column · scrollable</span>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df_recs,
            width='stretch',
            height=600,
            column_config={
                "Emp #": st.column_config.NumberColumn("Emp #", width="small"),
                "Job Role": st.column_config.TextColumn("Job Role", width="medium"),
                "Department": st.column_config.TextColumn("Department", width="medium"),
                "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
                "Skill Gap": st.column_config.TextColumn("Skill Gap", width="small"),
                "Top 3 Recommendations": st.column_config.TextColumn(
                    "Top 3 Recommendations",
                    width="large",
                    help="Semicolon-separated list of recommended training courses",
                ),
            },
            hide_index=True,
        )
    else:
        st.markdown(
            '<p class="not-found-msg">No recommendations found for the current filter.</p>',
            unsafe_allow_html=True,
        )

    # ── Zone spacer + divider ────────────────────────────────────────────────
    st.markdown('<div style="margin:2.5rem 0"></div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 7. Employee Drill-down ────────────────────────────────────────────────
    st.markdown('<p class="section-heading">Employee Drill-down</p>', unsafe_allow_html=True)

    drill_col, _ = st.columns([2, 5])
    with drill_col:
        emp_id_input = st.text_input(
            "Employee number",
            value="",
            placeholder="e.g. 1",
            help="Enter an EmployeeNumber to retrieve the full 360° intelligence record.",
            key="hr_drilldown_input",
        )

    if emp_id_input.strip():
        try:
            emp_id = int(emp_id_input.strip())
        except ValueError:
            st.markdown(
                '<p class="not-found-msg">Please enter a valid integer Employee number.</p>',
                unsafe_allow_html=True,
            )
            st.stop()

        result = _get(f"/employees/{emp_id}")

        if result is None and _BACKEND_ERROR:
            st.markdown(
                '<div class="backend-error">⚠ Backend not reachable — check the API server.</div>',
                unsafe_allow_html=True,
            )
        elif result is not None and result.get("__status") == 404:
            st.markdown(
                '<p class="not-found-msg">Employee not found. Check the ID and try again.</p>',
                unsafe_allow_html=True,
            )
        elif result:
            risk_sev  = result.get("RiskLevel", "LOW")
            skill_sev_raw = result.get("SkillGapSeverity", "LOW")
            skill_sev = skill_sev_raw if skill_sev_raw in ("HIGH", "MEDIUM", "LOW") else "NA"
            css_risk  = _risk_css_class(risk_sev)

            eng_score = result.get("EngagementScore")
            sat_score = result.get("SatisfactionScore")
            wlb_score = result.get("WorkLifeBalanceScore")

            onet_title = result.get("ONET_Title", "—")
            onet_conf  = result.get("ONET_Confidence", "")
            onet_full  = f"{onet_title} (confidence: {onet_conf})" if onet_conf else onet_title

            recs_raw   = result.get("Top3Recommendations", "—")
            if recs_raw and recs_raw not in ("N/A - Manager (use Department-level analysis)", "—"):
                recs_display = "<br>".join(
                    f"&nbsp;· {r.strip()}" for r in recs_raw.split(";") if r.strip()
                )
            else:
                recs_display = f'<span class="null-value">{recs_raw}</span>'

            def _field(label: str, value, null_msg: str = None):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    val_html = f'<span class="null-value">{null_msg or "—"}</span>'
                else:
                    val_html = f'<span class="field-value">{value}</span>'
                return (
                    f'<p class="field-label">{label}</p>'
                    f'{val_html}'
                )

            risk_label_class = f"risk-label-{risk_sev}" if risk_sev in ("HIGH", "MEDIUM", "LOW") else ""
            risk_display = (
                f'<span class="{risk_label_class}">{risk_sev}</span>'
                f' &nbsp;<span style="font-size:0.8rem;color:{C_MUTED};">'
                f'(score: {result.get("RiskScore", "—")})</span>'
            )

            skill_gap_display = skill_sev_raw

            st.markdown(
                f'<div class="risk-row {css_risk}" style="margin-top:0.5rem;">'
                f'<strong style="font-size:1rem;color:{C_INK};">'
                f'Employee #{result["EmployeeNumber"]}</strong>'
                f' &nbsp; {result.get("JobRole","—")} &nbsp;·&nbsp; {result.get("Department","—")}'
                f'<br>'
                f'<span class="field-label" style="font-size:0.72rem;">Attrition risk</span> '
                f'{risk_display}'
                f'</div>',
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(
                    _field("O*NET Role Mapping", onet_full)
                    + _field(
                        "Skill Gap",
                        f'{skill_gap_display} ({int(result["SkillGapCount"])} skills missing)'
                        if isinstance(result.get("SkillGapCount"), (int, float))
                           and not math.isnan(result.get("SkillGapCount", float("nan")))
                        else skill_gap_display
                    ),
                    unsafe_allow_html=True,
                )

            with col_b:
                if eng_score is None:
                    eng_html = '<p class="field-label">Engagement Score</p><span class="null-value">No survey data available</span>'
                    sat_html = '<p class="field-label">Satisfaction Score</p><span class="null-value">No survey data available</span>'
                    wlb_html = '<p class="field-label">Work-Life Balance Score</p><span class="null-value">No survey data available</span>'
                else:
                    eng_html = _field("Engagement Score", f"{eng_score:.2f} / 5.0")
                    sat_html = _field("Satisfaction Score", f"{sat_score:.2f} / 5.0" if sat_score else None, "No survey data available")
                    wlb_html = _field("Work-Life Balance Score", f"{wlb_score:.2f} / 5.0" if wlb_score else None, "No survey data available")

                st.markdown(eng_html + sat_html + wlb_html, unsafe_allow_html=True)

            st.markdown(
                f'<p class="field-label" style="margin-top:0.6rem;">Top 3 Upskilling Recommendations</p>'
                f'<div style="font-size:0.88rem;color:{C_INK};line-height:1.7;">{recs_display}</div>',
                unsafe_allow_html=True,
            )


# ==============================================================================
# VIEW 2: EMPLOYEE SELF-SERVICE VIEW ("My Profile")
# ==============================================================================
def render_my_profile_view():
    st.markdown(
        '<h1 class="display-title" style="font-size:2rem;margin-bottom:0.15rem;">'
        'My Career & Skills Profile</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.82rem;'
        f'color:{C_MUTED};margin-top:0;margin-bottom:1.5rem;">'
        f'Employee Self-Service · Personal Growth & Development Snapshot'
        f'</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            f'<p style="font-family:\'Source Serif 4\',serif;font-size:1rem;'
            f'font-weight:600;color:{C_INK};margin-bottom:0.5rem;">Employee Portal</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.78rem;'
            f'color:{C_MUTED};line-height:1.6;">'
            f'Welcome to your personal career development snapshot.<br><br>'
            f'Review your role benchmarks, engagement feedback, and personalized learning pathways.<br><br>'
            f'Switch to <b>HR / Manager View</b> above for organization-level workforce intelligence.'
            f'</p>',
            unsafe_allow_html=True,
        )

    # 1. Simple input: "Enter your Employee ID"
    input_col, _ = st.columns([2, 3])
    with input_col:
        profile_emp_id_str = st.text_input(
            "Enter your Employee ID",
            value="",
            placeholder="e.g. 1001",
            key="my_profile_emp_id_input",
            help="Enter your numeric Employee ID to view your personal career development profile.",
        )
        st.markdown(
            f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.75rem;'
            f'color:{C_MUTED};margin-top:-0.4rem;margin-bottom:1.2rem;">'
            f'Note: In a production environment, this is authenticated automatically via Single Sign-On (SSO). '
            f'This input is a prototype placeholder for authentication.'
            f'</p>',
            unsafe_allow_html=True,
        )

    if not profile_emp_id_str.strip():
        st.markdown(
            f'<div class="profile-panel">'
            f'<p style="font-size:0.95rem;font-weight:600;color:{C_ACCENT};margin:0 0 0.4rem;">'
            f'Welcome to your career snapshot</p>'
            f'<p style="font-size:0.85rem;color:{C_MUTED};margin:0;">'
            f'Please enter your Employee ID above to view your personalized role summary, '
            f'engagement feedback, and customized learning next steps.'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        profile_emp_id = int(profile_emp_id_str.strip())
    except ValueError:
        st.markdown(
            '<p class="not-found-msg">Please enter a valid numeric Employee ID.</p>',
            unsafe_allow_html=True,
        )
        return

    profile_data = _get(f"/employees/{profile_emp_id}")

    if profile_data is None and _BACKEND_ERROR:
        st.markdown(
            '<div class="backend-error">⚠ Backend not reachable — check the API server.</div>',
            unsafe_allow_html=True,
        )
    elif profile_data is not None and profile_data.get("__status") == 404:
        st.markdown(
            '<p class="not-found-msg">We couldn\'t find a profile for that Employee ID. Check the number and try again.</p>',
            unsafe_allow_html=True,
        )
    elif profile_data:
        emp_num = profile_data.get("EmployeeNumber", profile_emp_id)
        job_role = profile_data.get("JobRole", "—")
        dept = profile_data.get("Department", "—")
        tenure = profile_data.get("YearsAtCompany")
        tenure_str = f"{tenure} years with company" if tenure is not None else "Tenure not specified"

        eng_score = profile_data.get("EngagementScore")
        sat_score = profile_data.get("SatisfactionScore")
        wlb_score = profile_data.get("WorkLifeBalanceScore")

        onet_title = profile_data.get("ONET_Title", "—")
        onet_conf = profile_data.get("ONET_Confidence", "")
        onet_str = f"{onet_title} (benchmark alignment: {onet_conf})" if onet_conf else onet_title

        skill_gap_sev = profile_data.get("SkillGapSeverity", "")
        skill_gap_count = profile_data.get("SkillGapCount")
        is_mgr = (job_role == "Manager" or skill_gap_sev == "N/A - Manager")

        avg_comp_eng = summary.get("average_engagement", 2.95) if summary else 2.95

        # ── 2. "Your Snapshot" Section ────────────────────────────────────────
        st.markdown('<p class="section-heading">Your Snapshot</p>', unsafe_allow_html=True)

        snap_col_l, snap_col_r = st.columns(2)
        with snap_col_l:
            st.markdown(
                f'<div class="profile-panel">'
                f'<p class="field-label">Current Role & Department</p>'
                f'<p style="font-size:1.15rem;font-weight:600;color:{C_INK};margin:0 0 0.25rem;">{job_role}</p>'
                f'<p style="font-size:0.88rem;color:{C_MUTED};margin:0 0 0.8rem;">{dept} · {tenure_str}</p>'
                f'<p class="field-label">Industry Role Alignment</p>'
                f'<p style="font-size:0.88rem;color:{C_INK};margin:0;">{onet_str}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with snap_col_r:
            if eng_score is not None:
                eng_comp_str = f"Your engagement: {eng_score:.2f} / 5.0 (company average: {avg_comp_eng:.2f} / 5.0)"
                sat_comp_str = f"Job satisfaction: {sat_score:.2f} / 5.0" if sat_score is not None else ""
                wlb_comp_str = f"Work-life balance: {wlb_score:.2f} / 5.0" if wlb_score is not None else ""

                extra_feedback = ""
                if sat_comp_str:
                    extra_feedback += f'<div style="font-size:0.84rem;color:{C_MUTED};margin-top:0.3rem;">{sat_comp_str}</div>'
                if wlb_comp_str:
                    extra_feedback += f'<div style="font-size:0.84rem;color:{C_MUTED};margin-top:0.2rem;">{wlb_comp_str}</div>'

                st.markdown(
                    f'<div class="profile-panel">'
                    f'<p class="field-label">Engagement & Survey Feedback</p>'
                    f'<p style="font-size:0.95rem;font-weight:500;color:{C_INK};margin:0 0 0.25rem;">{eng_comp_str}</p>'
                    f'{extra_feedback}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="profile-panel">'
                    f'<p class="field-label">Engagement & Survey Feedback</p>'
                    f'<p style="font-size:0.92rem;color:{C_MUTED};font-style:italic;margin:0 0 0.35rem;">'
                    f'No survey data on file for you yet'
                    f'</p>'
                    f'<p style="font-size:0.78rem;color:{C_MUTED};margin:0;">'
                    f'Your feedback will appear here once the next survey cycle completes.'
                    f'</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="margin:1.5rem 0"></div>', unsafe_allow_html=True)

        # ── 3. "Your Skill Development" Section ───────────────────────────────
        st.markdown('<p class="section-heading">Your Skill Development</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.78rem;'
            f'color:{C_MUTED};margin-top:-0.45rem;margin-bottom:0.85rem;">'
            f'Target competencies benchmarked against industry standards for your role'
            f'</p>',
            unsafe_allow_html=True,
        )

        if is_mgr:
            growth_text = "Role profile: Management & Leadership Track"
            growth_sub = "As a manager, your growth is supported through executive coaching, departmental leadership forums, and strategic talent development rather than individual skill benchmark gap counts."
        elif skill_gap_count is not None and not math.isnan(skill_gap_count):
            growth_text = f"Growth areas identified: {int(skill_gap_count)}"
            growth_sub = "These target focus areas have been highlighted based on industry benchmarks for your role to support your continuous career progression."
        else:
            growth_text = "Growth areas identified: 0"
            growth_sub = "Your current skill profile aligns closely with role benchmarks. Explore advanced development opportunities below."

        st.markdown(
            f'<div class="profile-panel">'
            f'<p style="font-size:1rem;font-weight:600;color:{C_INK};margin:0 0 0.3rem;">{growth_text}</p>'
            f'<p style="font-size:0.84rem;color:{C_MUTED};margin:0;">{growth_sub}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.85rem;'
            f'font-weight:600;color:{C_ACCENT};margin:1rem 0 0.5rem;">'
            f'Recommended next steps:'
            f'</p>',
            unsafe_allow_html=True,
        )

        recs_raw = profile_data.get("Top3Recommendations", "")
        if is_mgr or recs_raw == "N/A - Manager (use Department-level analysis)":
            st.markdown(
                f'<div class="profile-recs-item">'
                f'<strong>1. Executive Leadership Roundtable:</strong> Participate in quarterly cross-functional strategic leadership sessions.'
                f'</div>'
                f'<div class="profile-recs-item">'
                f'<strong>2. Strategic Mentorship:</strong> Engage in executive talent mentoring and senior leadership sponsorship.'
                f'</div>'
                f'<div class="profile-recs-item">'
                f'<strong>3. People Development Excellence:</strong> Participate in tailored high-performance coaching and team resilience workshops.'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif recs_raw and recs_raw != "—":
            course_list = [c.strip() for c in recs_raw.split(";") if c.strip()]
            for idx, course in enumerate(course_list, 1):
                st.markdown(
                    f'<div class="profile-recs-item">'
                    f'<strong style="color:{C_ACCENT};">Step {idx}:</strong> {course}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<p style="font-size:0.84rem;color:{C_MUTED};font-style:italic;">'
                f'No specific recommendations currently assigned. Check back after your next capability review.'
                f'</p>',
                unsafe_allow_html=True,
            )

        # Seamless data notice attached directly under recommendations (no floating box/border)
        st.markdown(
            f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.76rem;'
            f'color:{C_MUTED};font-style:italic;margin-top:0.85rem;margin-bottom:0;line-height:1.4;">'
            f'Note: The skill and course suggestions below are placeholder estimates while we set up a real skills record for your role. They\'ll be updated once that\'s in place.'
            f'</p>',
            unsafe_allow_html=True,
        )


# ==============================================================================
# VIEW 3: PLATFORM ASSISTANT (RAG Knowledge Base)
# ==============================================================================
def render_platform_assistant_view():
    st.markdown(
        '<h1 class="display-title" style="font-size:2rem;margin-bottom:0.15rem;">'
        'Platform Assistant & Knowledge Base</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.84rem;'
        f'color:{C_MUTED};margin-top:0;margin-bottom:1.2rem;">'
        f'Grounded question answering powered by official O*NET occupational descriptions and model documentation'
        f'</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="provenance-banner">'
        f'<strong>Data provenance</strong> &mdash; This assistant searches verified project documentation '
        f'(<code>model_card.md</code>, <code>data_relationships.md</code>) and 1,016 official O*NET occupational descriptions. '
        f'External policies or general facts not in the corpus are strictly refused without hallucination.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="margin:1.2rem 0"></div>', unsafe_allow_html=True)

    # Example question buttons
    st.markdown(
        f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.85rem;'
        f'font-weight:600;color:{C_INK};margin-bottom:0.45rem;">'
        f'Example questions:'
        f'</p>',
        unsafe_allow_html=True,
    )

    trigger_query = None
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        if st.button("🔬 What does a Research Scientist do?", key="rag_ex1", use_container_width=True):
            trigger_query = "What does a Research Scientist do?"
            st.session_state["rag_user_input"] = trigger_query
        if st.button("⚠️ Why is the Manager role's O*NET mapping unreliable?", key="rag_ex2", use_container_width=True):
            trigger_query = "Why is the Manager role's O*NET mapping unreliable?"
            st.session_state["rag_user_input"] = trigger_query
        if st.button("🎯 What is the production model's decision threshold and why was it chosen?", key="rag_ex3", use_container_width=True):
            trigger_query = "What is the production model's decision threshold and why was it chosen?"
            st.session_state["rag_user_input"] = trigger_query

    with ex_col2:
        if st.button("📋 What is the company's parental leave policy? (Out of domain)", key="rag_ex4", use_container_width=True):
            trigger_query = "What is the company's parental leave policy?"
            st.session_state["rag_user_input"] = trigger_query
        if st.button("🌍 What is the capital of France? (General knowledge)", key="rag_ex5", use_container_width=True):
            trigger_query = "What is the capital of France?"
            st.session_state["rag_user_input"] = trigger_query

    st.markdown('<div style="margin:1rem 0"></div>', unsafe_allow_html=True)

    if "rag_user_input" not in st.session_state:
        st.session_state["rag_user_input"] = ""

    # Query Input Form
    with st.form(key="rag_search_form"):
        user_query = st.text_input(
            "Ask a question about roles, model specifications, or data relationships:",
            placeholder="e.g. What does a Laboratory Technician do?",
            key="rag_user_input"
        )
        submitted = st.form_submit_button("Ask Assistant")

    query_to_run = None
    if trigger_query:
        query_to_run = trigger_query.strip()
    elif submitted and user_query and user_query.strip():
        query_to_run = user_query.strip()

    if query_to_run:
        st.session_state["rag_active_query"] = query_to_run
        with st.spinner("Searching verified knowledge base..."):
            res = _post("/rag/ask", {"question": query_to_run})

        if not res or "__error" in res:
            err_detail = res.get("__error", "Backend connection failed") if res else "Unknown error"
            st.error(f"Failed to query knowledge base: {err_detail}")
            return

        st.session_state["rag_last_result"] = res

    last_res = st.session_state.get("rag_last_result")
    if last_res:
        answer = last_res.get("answer", "")
        sources = last_res.get("sources", [])

        is_refusal = (
            "don't have information" in answer.lower()
            or "not have information" in answer.lower()
            or "not contain sufficient" in answer.lower()
        )

        if is_refusal:
            st.markdown(
                f'<div class="rag-refusal">'
                f'<p style="font-size:0.75rem;font-weight:600;color:{C_MEDIUM};text-transform:uppercase;letter-spacing:0.04em;margin:0 0 0.35rem;">'
                f'Information Not in Knowledge Base'
                f'</p>'
                f'<p style="font-size:1.02rem;color:{C_INK};margin:0;font-family:\'IBM Plex Sans\',sans-serif;">'
                f'{answer}'
                f'</p>'
                f'<p style="font-size:0.78rem;color:{C_MUTED};margin-top:0.5rem;margin-bottom:0;line-height:1.4;">'
                f'The platform strictly refuses to answer questions that are not verified within its local documentation or O*NET occupational database.'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="rag-panel">'
                f'<p style="font-size:0.75rem;font-weight:600;color:{C_ACCENT};text-transform:uppercase;letter-spacing:0.04em;margin:0 0 0.35rem;">'
                f'Verified Platform Knowledge'
                f'</p>'
                f'<p style="font-size:1.08rem;color:{C_INK};margin:0;line-height:1.5;font-family:\'IBM Plex Sans\',sans-serif;">'
                f'{answer}'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Audited Sources
        with st.expander(f"Audited Sources & Provenance ({len(sources)} chunks retrieved)", expanded=not is_refusal):
            for idx, s in enumerate(sources, 1):
                score = s.get("score", 0.0)
                source_label = s.get("source", "Unknown")
                excerpt = s.get("excerpt", "").strip()

                st.markdown(
                    f'<div class="rag-source-box">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">'
                    f'<strong style="color:{C_ACCENT};font-size:0.86rem;">[{idx}] {source_label}</strong>'
                    f'<span style="font-size:0.75rem;color:{C_MUTED};background:{C_BG};padding:0.15rem 0.4rem;border-radius:2px;border:1px solid {C_DIVIDER};">'
                    f'Relevance: {score:.4f}'
                    f'</span>'
                    f'</div>'
                    f'<div style="color:{C_INK};font-size:0.83rem;line-height:1.45;font-style:italic;">'
                    f'"{excerpt}"'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ==============================================================================
# VIEW 4: ENTERPRISE AI AGENTS (LangGraph Multi-Agent Orchestrator)
# ==============================================================================
def render_agentic_orchestrator_view():
    st.markdown(
        '<h1 class="display-title" style="font-size:2rem;margin-bottom:0.15rem;">'
        'Enterprise HR AI Agents</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.84rem;'
        f'color:{C_MUTED};margin-top:0;margin-bottom:1.2rem;">'
        f'Unified multi-agent decision platform powered by LangGraph, deterministic intent routing, '
        f'and specialized enterprise intelligence.'
        f'</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="provenance-banner">'
        f'<strong>Agentic Architecture</strong> &mdash; Queries submitted here are sent directly to '
        f'the live backend API (<code>POST /agents/ask</code>) and dispatched across 5 specialized agents: '
        f'<strong>Policy</strong>, <strong>Workforce Intelligence</strong>, <strong>Upskilling</strong>, '
        f'<strong>Career</strong>, and <strong>HR Operations</strong>.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="margin:1.2rem 0"></div>', unsafe_allow_html=True)

    # Example question buttons across all capabilities
    st.markdown(
        f'<p style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.85rem;'
        f'font-weight:600;color:{C_INK};margin-bottom:0.45rem;">'
        f'Verified agent test scenarios:'
        f'</p>',
        unsafe_allow_html=True,
    )

    trigger_query = None
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button("📜 Policy: Career Stagnation", key="agent_ex1", use_container_width=True):
            trigger_query = "What does POL-CAREER-001 say about career stagnation?"
            st.session_state["agent_user_input"] = trigger_query
        if st.button("🚀 Career: Lab Tech Pathway", key="agent_ex4", use_container_width=True):
            trigger_query = "What is the career progression path for Laboratory Technician?"
            st.session_state["agent_user_input"] = trigger_query

    with ex_col2:
        if st.button("📊 Workforce: Risk for #1", key="agent_ex2", use_container_width=True):
            trigger_query = "What is the attrition risk for employee #1?"
            st.session_state["agent_user_input"] = trigger_query
        if st.button("👤 HR Ops: Profile for #1", key="agent_ex5", use_container_width=True):
            trigger_query = "Show employee record for employee #1"
            st.session_state["agent_user_input"] = trigger_query

    with ex_col3:
        if st.button("🎓 Upskilling: Courses for #1", key="agent_ex3", use_container_width=True):
            trigger_query = "Recommend courses for employee #1"
            st.session_state["agent_user_input"] = trigger_query
        if st.button("🌐 Out of Domain: Weather Query", key="agent_ex6", use_container_width=True):
            trigger_query = "What is the weather today?"
            st.session_state["agent_user_input"] = trigger_query

    st.markdown('<div style="margin:1rem 0"></div>', unsafe_allow_html=True)

    if "agent_user_input" not in st.session_state:
        st.session_state["agent_user_input"] = ""

    # Query Input Form
    with st.form(key="agent_search_form"):
        user_query = st.text_input(
            "Ask a question to the Global Agentic Orchestrator:",
            placeholder="e.g. What is the attrition risk for employee #1?",
            key="agent_user_input"
        )
        submitted = st.form_submit_button("Ask Agent Platform")

    query_to_run = None
    if trigger_query:
        query_to_run = trigger_query.strip()
    elif submitted and user_query and user_query.strip():
        query_to_run = user_query.strip()

    if query_to_run:
        st.session_state["agent_active_query"] = query_to_run
        with st.spinner("Dispatching through Global LangGraph Orchestrator..."):
            res = _post("/agents/ask", {"question": query_to_run})

        if not res or "__error" in res:
            err_detail = res.get("__error", "Backend connection failed") if res else "Unknown error"
            st.error(f"Failed to query Agentic Orchestrator: {err_detail}")
            return

        st.session_state["agent_last_result"] = res

    last_res = st.session_state.get("agent_last_result")
    if last_res:
        answer = last_res.get("answer", "")
        agent_routed = last_res.get("agent_routed", "Unknown")
        refusal_status = last_res.get("refusal_status", False)
        provenance = last_res.get("provenance", [])

        # Format Agent Name and Badge Color
        agent_display_names = {
            "policy_agent": ("Policy Agent", "#2B4C6F", "#EBF1F7"),
            "workforce_intelligence_agent": ("Workforce Intelligence Agent", "#B3492D", "#FBEFEF"),
            "upskilling_agent": ("Upskilling Agent", "#3F6E52", "#EAF3ED"),
            "career_agent": ("Career Progression Agent", "#4A5568", "#EDF2F7"),
            "hr_ops_agent": ("HR Operations Agent", "#B98A2E", "#FAF4EB"),
            "fallback_handler": ("Out-of-Domain Fallback", "#5B6270", "#F0F0EE"),
        }
        name, color, bg = agent_display_names.get(
            agent_routed,
            (agent_routed, "#5B6270", "#F0F0EE")
        )

        status_text = "Refusal / Out of Domain" if refusal_status else "Handled & Verified"
        status_color = C_MEDIUM if refusal_status else C_LOW
        status_bg = "#FAF4EB" if refusal_status else "#EAF3ED"

        st.markdown(
            f'<div class="agent-routing-banner">'
            f'<div>'
            f'<span style="font-size:0.75rem;color:{C_MUTED};margin-right:0.5rem;text-transform:uppercase;letter-spacing:0.04em;">Routed Agent:</span>'
            f'<span class="agent-badge" style="color:{color};background:{bg};border:1px solid {color}40;">{name}</span>'
            f'</div>'
            f'<div>'
            f'<span style="font-size:0.75rem;color:{C_MUTED};margin-right:0.5rem;text-transform:uppercase;letter-spacing:0.04em;">Status:</span>'
            f'<span class="agent-badge" style="color:{status_color};background:{status_bg};border:1px solid {status_color}40;">{status_text}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if refusal_status:
            st.markdown(
                f'<div class="agent-refusal">'
                f'<p style="font-size:0.75rem;font-weight:600;color:{C_MEDIUM};text-transform:uppercase;letter-spacing:0.04em;margin:0 0 0.35rem;">'
                f'Agent Refusal / Boundary Notice'
                f'</p>'
                f'<div style="font-size:1.02rem;color:{C_INK};margin:0;font-family:\'IBM Plex Sans\',sans-serif;">'
                f'{answer}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="agent-panel">'
                f'<div style="color:{C_INK};line-height:1.6;font-family:\'IBM Plex Sans\',sans-serif;">'
                f'{answer}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if provenance:
            with st.expander(f"Audited Provenance ({len(provenance)} records)", expanded=not refusal_status):
                for idx, p in enumerate(provenance, 1):
                    source = p.get("source", "Internal Record")
                    rec_type = p.get("record_type", "")
                    gov = p.get("governance", "")
                    st.markdown(
                        f'<div class="rag-source-box">'
                        f'<strong style="color:{C_ACCENT};">[{idx}] {source}</strong>'
                        f'{" &bull; " + rec_type if rec_type else ""}'
                        f'{" &bull; <em>" + gov + "</em>" if gov else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Render active view ────────────────────────────────────────────────────────
if portal_mode == "HR / Manager View":
    render_hr_manager_view()
elif portal_mode == "My Profile":
    render_my_profile_view()
elif portal_mode == "Platform Assistant (RAG)":
    render_platform_assistant_view()
else:
    render_agentic_orchestrator_view()


