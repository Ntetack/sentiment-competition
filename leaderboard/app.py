"""
app.py — Live Leaderboard for the Image Sentiment Analysis Challenge
Run with: streamlit run app.py
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ─── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎭 Sentiment Challenge — Leaderboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RESULTS_PATH = Path(__file__).parent / "results.json"
CLASS_NAMES = ["Very Negative 😠", "Negative 😞", "Neutral 😐", "Positive 😊", "Very Positive 😄"]
GITHUB_REPO = os.getenv("GITHUB_REPO", "https://github.com/your-username/sentiment-competition")
AUTO_REFRESH_SECONDS = 60


# ─── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #111122 50%, #0a0f1a 100%);
        min-height: 100vh;
    }

    /* Header */
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    .hero-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #64748b;
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.25rem;
    }

    /* Rank badges */
    .rank-1 { color: #fbbf24; font-weight: 700; font-size: 1.2rem; }
    .rank-2 { color: #94a3b8; font-weight: 700; font-size: 1.1rem; }
    .rank-3 { color: #b45309; font-weight: 700; font-size: 1.1rem; }

    /* Status badges */
    .badge-success {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
    }

    .badge-error {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
    }

    /* Table */
    .leaderboard-row {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
    }

    .leaderboard-row:hover {
        background: rgba(167, 139, 250, 0.05);
        border-color: rgba(167, 139, 250, 0.2);
    }

    /* Section headers */
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 3px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 2rem 0;
    }

    /* Streamlit overrides */
    .stDataFrame { background: transparent; }
    div[data-testid="metric-container"] { background: transparent; }
    .stSpinner { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_data():
    if not RESULTS_PATH.exists():
        return None, []
    with open(RESULTS_PATH) as f:
        raw = json.load(f)
    return raw, raw.get("submissions", [])


def format_datetime(iso_str: str) -> str:
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M UTC")
    except Exception:
        return iso_str


def rank_emoji(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


# ─── Layout ───────────────────────────────────────────────────────────────────

def render_hero(last_updated: str):
    st.markdown("""
    <div style="padding: 3rem 0 2rem 0;">
        <p class="hero-sub">🎭 Image Sentiment Analysis Challenge</p>
        <h1 class="hero-title">LEADERBOARD</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <p style="text-align:center; color:#475569; font-size:0.8rem; font-family:'Space Mono',monospace;">
            🔄 Last updated: {format_datetime(last_updated)}
            &nbsp;·&nbsp;
            Auto-refresh every {AUTO_REFRESH_SECONDS}s
        </p>
        """, unsafe_allow_html=True)


def render_stats(submissions: list):
    valid = [s for s in submissions if s.get("status") == "success"]
    teams = set(s["team_name"] for s in valid)
    best_f1 = max((s["f1_score"] for s in valid), default=0.0)
    
    cols = st.columns(4)
    stats = [
        ("Total Submissions", len(submissions), ""),
        ("Teams", len(teams), ""),
        ("Best F1", f"{best_f1:.4f}", ""),
        ("Classes", "5", "Sentiment Labels"),
    ]
    
    for col, (label, value, sub) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_podium(submissions: list):
    valid = sorted(
        [s for s in submissions if s.get("status") == "success"],
        key=lambda x: x["f1_score"],
        reverse=True
    )
    
    if not valid:
        st.info("No successful submissions yet. Be the first! 🚀")
        return
    
    st.markdown('<p class="section-title">🏆 Top 3</p>', unsafe_allow_html=True)
    
    top3 = valid[:3]
    # Arrange as 2nd, 1st, 3rd
    order = [1, 0, 2] if len(top3) >= 3 else list(range(len(top3)))
    
    cols = st.columns(len(order))
    heights = ["80px", "120px", "60px"]
    
    for col_idx, sub_idx in enumerate(order):
        if sub_idx >= len(top3):
            continue
        sub = top3[sub_idx]
        with cols[col_idx]:
            emoji = rank_emoji(sub_idx + 1)
            h = heights[col_idx] if col_idx < len(heights) else "60px"
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(167,139,250,0.2);
                border-radius: 16px;
                padding: 1.5rem;
                text-align: center;
                margin-top: {h};
            ">
                <div style="font-size: 2rem;">{emoji}</div>
                <div style="
                    font-family: 'Space Mono', monospace;
                    font-weight: 700;
                    color: #e2e8f0;
                    margin-top: 0.5rem;
                    font-size: 1rem;
                ">{sub['team_name']}</div>
                <div style="color: #64748b; font-size: 0.75rem; margin: 0.25rem 0;">{sub['model_name']}</div>
                <div style="
                    font-family: 'Space Mono', monospace;
                    font-size: 1.8rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #a78bfa, #38bdf8);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">{sub['f1_score']:.4f}</div>
                <div style="color: #475569; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;">F1 Score</div>
            </div>
            """, unsafe_allow_html=True)


def render_full_table(submissions: list):
    st.markdown('<p class="section-title" style="margin-top:2rem;">📊 All Submissions</p>', unsafe_allow_html=True)
    
    valid = [s for s in submissions if s.get("status") == "success"]
    
    if not valid:
        st.info("No evaluations completed yet.")
        return
    
    df = pd.DataFrame(valid)
    
    # Format columns
    display_df = pd.DataFrame({
        "Rank": df["rank"].apply(lambda r: rank_emoji(r)),
        "Team": df["team_name"],
        "Model": df["model_name"],
        "Architecture": df.get("architecture", "Unknown"),
        "F1 Score": df["f1_score"].apply(lambda x: f"{x:.4f}"),
        "Accuracy": df["accuracy"].apply(lambda x: f"{x:.4f}"),
        "Inference (ms)": df["inference_time_ms"].apply(lambda x: f"{x:.0f}"),
        "Submitted": df["submitted_at"].apply(format_datetime),
        "Issue": df["issue_number"].apply(lambda n: f"#{n}"),
    })
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn(width="small"),
            "F1 Score": st.column_config.TextColumn(width="small"),
        }
    )


def render_f1_chart(submissions: list):
    valid = [s for s in submissions if s.get("status") == "success"]
    if len(valid) < 2:
        return
    
    st.markdown('<p class="section-title" style="margin-top:2rem;">📈 Score Distribution</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # F1 scores bar chart
        sorted_subs = sorted(valid, key=lambda x: x["f1_score"], reverse=True)[:15]
        df = pd.DataFrame(sorted_subs)
        
        fig = px.bar(
            df,
            x="team_name",
            y="f1_score",
            color="f1_score",
            color_continuous_scale=["#1e293b", "#a78bfa", "#38bdf8"],
            title="Top F1 Scores by Team",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="DM Sans"),
            title_font=dict(color="#e2e8f0"),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickangle=-30),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Per-class F1 radar for top-3
        top3 = sorted(valid, key=lambda x: x["f1_score"], reverse=True)[:3]
        
        fig = go.Figure()
        colors = ["#a78bfa", "#38bdf8", "#34d399"]
        
        for i, sub in enumerate(top3):
            per_class = sub.get("per_class_f1", [0] * 5)
            short_names = ["V.Neg", "Neg", "Neutral", "Pos", "V.Pos"]
            
            fig.add_trace(go.Scatterpolar(
                r=per_class + [per_class[0]],
                theta=short_names + [short_names[0]],
                fill="toself",
                fillcolor=colors[i].replace("#", "rgba(") + ", 0.1)",
                line=dict(color=colors[i], width=2),
                name=f"#{i+1} {sub['team_name']}",
                opacity=0.8,
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor="rgba(255,255,255,0.08)",
                    linecolor="rgba(255,255,255,0.08)",
                    tickfont=dict(color="#64748b"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.08)",
                    linecolor="rgba(255,255,255,0.08)",
                    tickfont=dict(color="#94a3b8"),
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="DM Sans"),
            title=dict(text="Per-class F1 — Top 3", font=dict(color="#e2e8f0")),
            legend=dict(font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_timeline(submissions: list):
    valid = [s for s in submissions if s.get("status") == "success" and s.get("submitted_at")]
    if len(valid) < 3:
        return
    
    df = pd.DataFrame(valid)
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df = df.sort_values("submitted_at")
    
    st.markdown('<p class="section-title" style="margin-top:2rem;">🕐 Score Progression</p>', unsafe_allow_html=True)
    
    fig = px.scatter(
        df,
        x="submitted_at",
        y="f1_score",
        color="team_name",
        size="f1_score",
        size_max=20,
        hover_data=["model_name", "accuracy"],
        title="F1 Scores Over Time",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="DM Sans"),
        title_font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cta():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="
            text-align: center;
            padding: 2rem;
            background: rgba(167,139,250,0.05);
            border: 1px solid rgba(167,139,250,0.15);
            border-radius: 20px;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚀</div>
            <div style="
                font-family: 'Space Mono', monospace;
                font-size: 1.1rem;
                color: #e2e8f0;
                font-weight: 700;
                margin-bottom: 0.5rem;
            ">Ready to compete?</div>
            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 1.25rem;">
                Train your model and submit via GitHub Issues.
                Results appear here within ~10 minutes.
            </div>
            <a href="{GITHUB_REPO}/issues/new?template=submission.yml" target="_blank" style="
                background: linear-gradient(135deg, #a78bfa, #38bdf8);
                color: #0a0a0f;
                padding: 0.75rem 2rem;
                border-radius: 999px;
                text-decoration: none;
                font-weight: 700;
                font-family: 'Space Mono', monospace;
                font-size: 0.85rem;
                letter-spacing: 1px;
            ">Submit Your Model →</a>
        </div>
        """, unsafe_allow_html=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    raw, submissions = load_data()
    last_updated = raw.get("last_updated", "") if raw else ""
    
    render_hero(last_updated)
    
    st.markdown("---")
    render_stats(submissions)
    
    st.markdown("---")
    render_podium(submissions)
    
    render_full_table(submissions)
    render_f1_chart(submissions)
    render_timeline(submissions)
    render_cta()
    
    # Auto-refresh
    st.markdown(f"""
    <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {AUTO_REFRESH_SECONDS * 1000});
    </script>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
