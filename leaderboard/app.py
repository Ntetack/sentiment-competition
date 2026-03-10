"""
app.py — Live Leaderboard for the Image Emotion Recognition Challenge
Run: streamlit run app.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎭 Emotion Challenge — Leaderboard",
    page_icon="🏆",
    layout="wide",
)

RESULTS_PATH = Path(__file__).parent / "results.json"
GITHUB_REPO = os.getenv("GITHUB_REPO", "https://github.com/your-username/sentiment-competition")
EMOTIONS = ["Angry 😠", "Disgust 🤢", "Fear 😨", "Happy 😄", "Neutral 😐", "Sad 😢", "Surprise 😲"]
EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
AUTO_REFRESH = 60

# ─── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp {
    background: #0d1117;
}

.hero {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -1px;
}
.hero-title span {
    background: linear-gradient(90deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.stat-label {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
}

.podium-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.podium-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
}
.podium-team {
    color: #f0f6fc;
    font-weight: 600;
    font-size: 1rem;
    margin: 0.4rem 0 0.1rem 0;
}
.podium-model {
    color: #8b949e;
    font-size: 0.78rem;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.6rem;
    margin: 2rem 0 1.2rem 0;
}

.submit-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_data():
    if not RESULTS_PATH.exists():
        return {}, []
    with open(RESULTS_PATH) as f:
        raw = json.load(f)
    return raw, raw.get("submissions", [])


def fmt_date(iso: str) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%b %d %Y, %H:%M UTC")
    except Exception:
        return iso


def rank_emoji(r: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(r, f"#{r}")

# ─── Sections ─────────────────────────────────────────────────────────────────

def render_hero(last_updated):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-sub">🎭 Image Emotion Recognition Challenge</div>
        <div class="hero-title">LEADER<span>BOARD</span></div>
        <div style="color:#484f58; font-family:'IBM Plex Mono',monospace; font-size:0.72rem; margin-top:0.5rem;">
            Last updated: {fmt_date(last_updated)} · Auto-refresh every {AUTO_REFRESH}s
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats(submissions):
    valid = [s for s in submissions if s.get("status") == "success"]
    teams = len(set(s["team_name"] for s in valid))
    best = max((s["accuracy"] for s in valid), default=0.0)

    cols = st.columns(4)
    for col, (val, label) in zip(cols, [
        (len(submissions), "Submissions"),
        (teams, "Teams"),
        (f"{best*100:.2f}%", "Best Accuracy"),
        ("7", "Emotion Classes"),
    ]):
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)


def render_podium(submissions):
    valid = sorted(
        [s for s in submissions if s.get("status") == "success"],
        key=lambda x: x["accuracy"], reverse=True
    )
    if not valid:
        st.info("No submissions yet — be the first! 🚀")
        return

    st.markdown('<div class="section-label">🏆 Top Submissions</div>', unsafe_allow_html=True)

    top3 = valid[:3]
    order = [1, 0, 2] if len(top3) >= 3 else list(range(len(top3)))
    tops = [40, 0, 70]

    cols = st.columns(len(order))
    for ci, si in enumerate(order):
        if si >= len(top3):
            continue
        s = top3[si]
        mt = tops[ci] if ci < len(tops) else 40
        cols[ci].markdown(f"""
        <div class="podium-card" style="margin-top:{mt}px;">
            <div style="font-size:2rem;">{rank_emoji(si+1)}</div>
            <div class="podium-score">{s['accuracy']*100:.2f}%</div>
            <div class="podium-team">{s['team_name']}</div>
            <div class="podium-model">{s['model_name']}</div>
            <div style="color:#484f58; font-size:0.7rem; margin-top:0.4rem;">{fmt_date(s.get('submitted_at',''))}</div>
        </div>""", unsafe_allow_html=True)


def render_table(submissions):
    valid = [s for s in submissions if s.get("status") == "success"]
    if not valid:
        return

    st.markdown('<div class="section-label">📊 All Submissions</div>', unsafe_allow_html=True)

    df = pd.DataFrame([{
        "Rank": rank_emoji(s["rank"]),
        "Team": s["team_name"],
        "Model": s["model_name"],
        "Accuracy": f"{s['accuracy']*100:.2f}%",
        "Correct": f"{s['correct']} / {s['total']}",
        "Submitted": fmt_date(s.get("submitted_at", "")),
        "Issue": f"#{s['issue_number']}",
    } for s in sorted(valid, key=lambda x: x["rank"])])

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_charts(submissions):
    valid = [s for s in submissions if s.get("status") == "success"]
    if len(valid) < 2:
        return

    st.markdown('<div class="section-label">📈 Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Bar chart — accuracy per team (best submission)
    with col1:
        best_per_team = {}
        for s in valid:
            t = s["team_name"]
            if t not in best_per_team or s["accuracy"] > best_per_team[t]["accuracy"]:
                best_per_team[t] = s

        df = pd.DataFrame(list(best_per_team.values()))
        df = df.sort_values("accuracy", ascending=True)

        fig = px.bar(
            df, x="accuracy", y="team_name", orientation="h",
            color="accuracy",
            color_continuous_scale=["#1c2128", "#58a6ff"],
            title="Best Accuracy per Team",
            labels={"accuracy": "Accuracy", "team_name": ""},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", family="IBM Plex Sans"),
            title_font=dict(color="#f0f6fc"),
            coloraxis_showscale=False,
            xaxis=dict(tickformat=".0%", gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Radar — per-class accuracy for top 3
    with col2:
        top3 = sorted(valid, key=lambda x: x["accuracy"], reverse=True)[:3]
        colors = ["#58a6ff", "#bc8cff", "#3fb950"]
        short = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        fig = go.Figure()
        for i, s in enumerate(top3):
            pc = s.get("per_class", {})
            values = [pc.get(k, {}).get("acc", 0) for k in EMOTION_KEYS]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=short + [short[0]],
                fill="toself",
                line=dict(color=colors[i], width=2),
                fillcolor=colors[i].replace("#", "rgba(").rstrip(")") + ", 0.08)",
                name=f"#{i+1} {s['team_name']}",
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], gridcolor="#21262d", tickfont=dict(color="#484f58")),
                angularaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b949e", family="IBM Plex Sans"),
            title=dict(text="Per-class Accuracy — Top 3", font=dict(color="#f0f6fc")),
            legend=dict(font=dict(color="#8b949e")),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_cta():
    st.markdown(f"""
    <div class="submit-box">
        <div style="font-size:1.8rem; margin-bottom:0.5rem;">🚀</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1rem; color:#f0f6fc; font-weight:600; margin-bottom:0.5rem;">
            Ready to compete?
        </div>
        <div style="color:#8b949e; font-size:0.85rem; margin-bottom:1.25rem;">
            Generate your predictions CSV and submit via GitHub Issues.<br>
            Results appear here within ~5 minutes.
        </div>
        <a href="{GITHUB_REPO}/issues/new?template=submission.yml" target="_blank"
           style="background:linear-gradient(90deg,#58a6ff,#bc8cff); color:#0d1117;
                  padding:0.65rem 2rem; border-radius:8px; text-decoration:none;
                  font-weight:700; font-family:'IBM Plex Mono',monospace; font-size:0.82rem;">
            Submit Predictions →
        </a>
    </div>
    """, unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    raw, submissions = load_data()
    last_updated = raw.get("last_updated", "") if raw else ""

    render_hero(last_updated)
    st.markdown("---")
    render_stats(submissions)
    st.markdown("---")
    render_podium(submissions)
    render_table(submissions)
    render_charts(submissions)
    render_cta()

    # Auto-refresh
    st.markdown(f"""
    <script>
    setTimeout(() => window.location.reload(), {AUTO_REFRESH * 1000});
    </script>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
