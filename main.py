import streamlit as st
import pandas as pd
import requests
from streamlit_gsheets import GSheetsConnection
from datetime import datetime, timedelta, timezone

try:
    from google import genai
except ImportError:
    genai = None

st.set_page_config(
    page_title="Tournament Sweepstake Hub",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core progressive knockout staging tracking array
STAGE_ORDER = ["Group Stage", "Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Finals", "Champion"]

API_STAGE_MAP = {
    "LAST_32": "Round of 32",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-Finals",
    "SEMI_FINALS": "Semi-Finals",
    "FINAL": "Finals"
}

BRACKET_STAGES = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Finals"]

# Number of match slots expected in each knockout stage (fixed by tournament format)
STAGE_SLOT_COUNTS = {
    "Round of 32": 16,
    "Round of 16": 8,
    "Quarter-Finals": 4,
    "Semi-Finals": 2,
    "Finals": 1,
}

DEFAULT_FLAG = "https://flagcdn.com/w40/un.png"


def check_secrets():
    """Verify all required secrets are present."""
    required = {
        "football_api": ["api_token"],
        "passwords": ["admin_password"],
        "connections": ["gsheets"],
        "gemini_api": ["api_key"]
    }
    if not genai:
        return False, "The 'google-genai' package is not installed.\nRun: pip install google-genai"
    for section, keys in required.items():
        if section not in st.secrets:
            return False, f"Missing section: [{section}] in secrets.toml"
        for key in keys:
            if key not in st.secrets[section]:
                return False, f"Missing key: {key} in [{section}]"
    return True, ""


# -------------------------------------------------------------
# AI CORE UTILITIES (GEMINI 2.5)
# -------------------------------------------------------------
@st.cache_data(persist="disk", show_spinner=False)
def get_gemini_summary(match_id, h_player, a_player, score_str, goal_info):
    """Generates post-match summary exactly ONCE per unique match data signature."""
    api_key = st.secrets.get("gemini_api", {}).get("api_key")
    if not api_key or not genai:
        return "AI integration offline."

    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"Context: Football tournament match result.\n"
            f"Participants: {h_player} vs {a_player}.\n"
            f"Final Score: {score_str}.\n"
            f"Goal details: {goal_info}.\n"
            f"Instruction: Generate a one-sentence fun, dramatic, and witty post-match commentary. "
            f"Do not include any emojis. Do not format with markdown bolding or asterisks. Be punchy."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        if response and response.text:
            return response.text.strip().replace('"', '').replace('*', '')
        return "The match result left the AI speechless!"
    except Exception:
        return "What an incredible finish to this matchup!"


@st.cache_data(persist="disk", show_spinner=False)
def get_gemini_preview(match_id, h_player, a_player, h_prob, a_prob):
    """Generates upcoming match preview narrative exactly ONCE per match ID."""
    api_key = st.secrets.get("gemini_api", {}).get("api_key")
    if not api_key or not genai:
        return None

    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"Context: Upcoming tournament sweepstake football match.\n"
            f"Matchup: {h_player} vs {a_player}.\n"
            f"Calculated Win Probabilities: {h_player} has a {h_prob:.0%} chance, while {a_player} has a {a_prob:.0%} chance.\n"
            f"Instruction: Generate a short, single-sentence dramatic narrative intro line hype setup. "
            f"Use predictions, but do not include percentages in response."
            f"No emojis. No asterisks. The response should be funny."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        if response and response.text:
            return response.text.strip().replace('"', '').replace('*', '')
    except Exception:
        return None


# -------------------------------------------------------------
# DYNAMIC FOOTBALL API DATA INGESTION ENGINE
# -------------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_live_tournament_data(api_token):
    """
    Returns:
        stats: dict[team_name] -> team stat record
        bracket_matches: dict[stage] -> list of match dicts (deduped by match id,
            sorted by kickoff time then id), one entry per real fixture.
        all_matches: raw list of matches from the API (deduped by id, sorted)
    """
    stats = {}
    bracket_matches = {stage: [] for stage in BRACKET_STAGES}
    all_matches = []

    if not api_token or str(api_token).strip() == "":
        st.sidebar.error("⚠️ API Token is missing. Please check your secrets configuration.")
        return stats, bracket_matches, all_matches

    try:
        headers = {"X-Auth-Token": str(api_token).strip()}
        matches_url = "https://api.football-data.org/v4/competitions/WC/matches"
        matches_res = requests.get(matches_url, headers=headers, timeout=10)

        if matches_res.status_code != 200:
            st.sidebar.error(f"⚠️ Football API returned status {matches_res.status_code}.")
            return stats, bracket_matches, all_matches

        matches_data = matches_res.json()

        # Dedup by match id (the API is the source of truth for uniqueness, not team-name pairs)
        seen_ids = set()
        deduped_matches = []
        for match in matches_data.get("matches", []):
            m_id = match.get("id")
            if m_id in seen_ids:
                continue
            seen_ids.add(m_id)
            deduped_matches.append(match)

        matches = sorted(deduped_matches, key=lambda x: (x.get("utcDate", ""), x.get("id", 0)))

        for match in matches:
            stage = match["stage"]
            home_obj = match.get("homeTeam")
            away_obj = match.get("awayTeam")

            home_team = (home_obj.get("name") if home_obj else "TBD") or "TBD"
            away_team = (away_obj.get("name") if away_obj else "TBD") or "TBD"

            if home_team == "United States":
                home_team = "USA"
            if away_team == "United States":
                away_team = "USA"

            if home_obj:
                match["homeTeam"]["name"] = home_team
            if away_obj:
                match["awayTeam"]["name"] = away_team

            all_matches.append(match)
            status = match["status"]

            for t, team_meta in [(home_team, home_obj), (away_team, away_obj)]:
                if t not in ["TBD", "TBC"] and team_meta and t not in stats:
                    flag_url = team_meta.get("crest") or DEFAULT_FLAG
                    stats[t] = {
                        "Team": t, "Flag": flag_url, "Won": 0, "Lost": 0, "Points": 0,
                        "Goals Scored": 0, "Stage": "Group Stage", "Status": "Active",
                        "Match Scores": {}, "Live Stages": []
                    }

            current_stage_mapped = API_STAGE_MAP.get(stage)

            if current_stage_mapped:
                for t in [home_team, away_team]:
                    if t not in ["TBD", "TBC"]:
                        current_team_rank = (
                            STAGE_ORDER.index(stats[t]["Stage"]) if stats[t]["Stage"] in STAGE_ORDER else 0
                        )
                        match_stage_rank = STAGE_ORDER.index(current_stage_mapped)
                        if match_stage_rank > current_team_rank:
                            stats[t]["Stage"] = current_stage_mapped

                # One entry per real fixture id — this is what eliminates duplicate
                # bracket cards, including swapped home/away or rescheduled dupes.
                bracket_matches[current_stage_mapped].append(match)

            if status in ["FINISHED", "IN_PLAY", "PAUSED"]:
                score_data = match.get("score", {})
                full_time = score_data.get("fullTime", {})
                home_goals = full_time.get("home") or 0
                away_goals = full_time.get("away") or 0

                if home_team not in ["TBD", "TBC"]:
                    stats[home_team]["Goals Scored"] += home_goals
                if away_team not in ["TBD", "TBC"]:
                    stats[away_team]["Goals Scored"] += away_goals

                if current_stage_mapped:
                    if home_team not in ["TBD", "TBC"]:
                        stats[home_team]["Match Scores"][current_stage_mapped] = home_goals
                    if away_team not in ["TBD", "TBC"]:
                        stats[away_team]["Match Scores"][current_stage_mapped] = away_goals

                    if status in ["IN_PLAY", "PAUSED"]:
                        if home_team not in ["TBD", "TBC"]:
                            stats[home_team]["Live Stages"].append(current_stage_mapped)
                        if away_team not in ["TBD", "TBC"]:
                            stats[away_team]["Live Stages"].append(current_stage_mapped)

            if status == "FINISHED" and current_stage_mapped and home_team not in ["TBD", "TBC"] and away_team not in ["TBD", "TBC"]:
                winner = match.get("score", {}).get("winner")

                next_stage = current_stage_mapped
                if current_stage_mapped in STAGE_ORDER:
                    curr_idx = STAGE_ORDER.index(current_stage_mapped)
                    if curr_idx + 1 < len(STAGE_ORDER):
                        next_stage = STAGE_ORDER[curr_idx + 1]

                if winner == "HOME_TEAM":
                    stats[away_team]["Status"] = "Knocked Out"
                    stats[away_team]["Stage"] = current_stage_mapped
                    stats[home_team]["Stage"] = next_stage
                elif winner == "AWAY_TEAM":
                    stats[home_team]["Status"] = "Knocked Out"
                    stats[home_team]["Stage"] = current_stage_mapped
                    stats[away_team]["Stage"] = next_stage

                if stage == "FINAL" and winner:
                    champ = home_team if winner == "HOME_TEAM" else away_team
                    stats[champ]["Stage"] = "Champion"
                    stats[champ]["Status"] = "Winner"

    except Exception as e:
        st.sidebar.error(f"API Connection Issue: {e}")

    return stats, bracket_matches, all_matches


conn = st.connection("gsheets", type=GSheetsConnection)


def database_load_pipeline():
    try:
        df_p = conn.read(worksheet="Participants_Wimbledon", ttl=0)
        if df_p is None:
            df_p = pd.DataFrame(columns=["Participant Name", "Team Assigned"])
    except Exception as e:
        st.sidebar.warning(f"Connection to Sheets failed: {e}")
        df_p = pd.DataFrame(columns=["Participant Name", "Team Assigned"])

    if df_p.empty or "Participant Name" not in df_p.columns or "Team Assigned" not in df_p.columns:
        df_p = pd.DataFrame(columns=["Participant Name", "Team Assigned"])
    else:
        df_p = df_p.dropna(subset=["Participant Name"])
        df_p["Participant Name"] = df_p["Participant Name"].astype(str).str.strip()
        df_p["Team Assigned"] = df_p["Team Assigned"].astype(str).str.strip()

    api_token = st.secrets.get("football_api", {}).get("api_token", "")
    live_stats, bracket_matches, matches_list = fetch_live_tournament_data(api_token)
    df_t = pd.DataFrame(list(live_stats.values()))

    if df_t.empty:
        df_t = pd.DataFrame(columns=["Team", "Flag", "Won", "Lost", "Points", "Goals Scored", "Stage", "Status"])

    return df_p, df_t, bracket_matches, matches_list


def team_flag(df_teams, team_name):
    if df_teams.empty:
        return DEFAULT_FLAG
    row = df_teams[df_teams["Team"] == team_name]
    if row.empty:
        return DEFAULT_FLAG
    return row.iloc[0]["Flag"]


def render_team_markup(df_teams, team_to_player, team_name, current_stage_title, opponent_name=None):
    if team_name in ["TBD", "TBC"] or not team_name:
        return (
            "<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap;'>"
            "<span style='color:gray; font-style:italic;'>🏳️ TBD</span></div>"
        )

    match_row = df_teams[df_teams["Team"] == team_name]
    player_name = team_to_player.get(str(team_name).upper())
    display_name = player_name if player_name else team_name

    if match_row.empty:
        return (
            f"<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap;'>"
            f"<span>🏳️ {display_name}</span></div>"
        )

    row = match_row.iloc[0]
    flag_html = f"<img src='{row['Flag']}' width='20' style='vertical-align: middle; margin-right: 5px; flex-shrink: 0;'>"

    match_scores = row.get("Match Scores", {})
    score = match_scores.get(current_stage_title)

    is_live = current_stage_title in row.get("Live Stages", [])
    live_badge = (
        "<span style='color: #2ecc71; font-size: 10px; font-weight: bold; margin-left: 5px; "
        "animation: blinker 1s linear infinite;'>● LIVE</span>" if is_live else ""
    )

    trophy_suffix = (
        " <span style='font-size:14px; margin-left:3px;'>🏆</span>"
        if (current_stage_title == "Finals" and row["Stage"] == "Champion") else ""
    )

    score_html = ""
    if score is not None:
        score_html = f"<span style='font-size: 18px; font-weight: bold; margin-left: 8px; flex-shrink: 0;'>{score}</span>"
    elif opponent_name and opponent_name not in ["TBD", "TBC"]:
        opp_row = df_teams[df_teams["Team"] == opponent_name]
        if not opp_row.empty:
            team_goals = row["Goals Scored"]
            opp_goals = opp_row.iloc[0]["Goals Scored"]
            win_prob = (team_goals + 1) / (team_goals + opp_goals + 2)
            score_html = (
                f"<span style='font-size: 11px; color: #888; margin-left: 8px; font-style: italic;' "
                f"title='Win probability based on total goals scored'>{win_prob:.0%}</span>"
            )

    if row["Status"] == "Knocked Out" and row["Stage"] == current_stage_title:
        return (
            f"<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap; width: 100%;'>"
            f"<div style='display: flex; align-items: center; min-width: 0; overflow: hidden; text-overflow: ellipsis;'>"
            f"{flag_html}<span style='color:gray; text-decoration:line-through;'>{display_name}</span>"
            f"<span style='font-size:11px; color:red; margin-left:3px;'>❌</span></div>{score_html}</div>"
        )
    else:
        return (
            f"<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap; width: 100%;'>"
            f"<div style='display: flex; align-items: center; min-width: 0; overflow: hidden; text-overflow: ellipsis;'>"
            f"{flag_html}<span style='font-weight: bold;'>{display_name}</span>{live_badge}{trophy_suffix}</div>{score_html}</div>"
        )


def main():
    secrets_ok, error_msg = check_secrets()
    if not secrets_ok:
        st.error("🚨 Missing Configuration")
        st.info(f"Details: {error_msg}")
        st.stop()

    df_participants, df_teams, bracket_matches, raw_matches = database_load_pipeline()

    st.markdown("""
        <style>
        @media (max-width: 800px) {
            div[data-testid="stHorizontalBlock"] {
                overflow-x: auto;
                flex-wrap: nowrap !important;
                gap: 1rem !important;
                padding-bottom: 15px;
            }
            div[data-testid="column"] {
                min-width: 250px !important;
                flex: 0 0 auto !important;
            }
        }
        @keyframes blinker {
            50% { opacity: 0; }
        }
        </style>
    """, unsafe_allow_html=True)

    team_to_player = {}
    if not df_participants.empty:
        team_to_player = {
            str(k).strip().upper(): str(v).strip()
            for k, v in zip(df_participants["Team Assigned"], df_participants["Participant Name"])
        }

    st.sidebar.title("🏆 Sweepstake Hub")
    with st.sidebar:
        st.markdown("### Automated Live Rules")
        with st.expander("ℹ️ Tracking Logic", expanded=False):
            st.markdown("""
            Scores and progress are updated via official API:
            1. **Leaderboard:** Shows your assigned player and their team's current status.
            2. **Bracket:** Tracks progress through the knockout phases.
            """)
        if df_teams.empty:
            st.sidebar.info("📅 Tournament is currently in Group Stages. Knockout data will appear once the Round of 32 begins.")
        st.write("---")
        app_view = st.radio("Switch Dashboard View", ["📊 Public Fan Dashboard", "🔐 Admin Control Panel"])

    if app_view == "📊 Public Fan Dashboard":
        st.title("📊 The Wimbledon World Cup")
        st.caption("Updated automatically from official game knockout data feeds.")

        tab_lead, tab_bracket, tab_feed = st.tabs([
            "🏆 Live Leaderboard",
            "🌳 Knockout Bracket Tracker",
            "📱 Feed"
        ])

        with tab_lead:
            st.subheader("🏆 Global Sweepstake Leaderboard")
            df_merged = df_participants.merge(df_teams, left_on="Team Assigned", right_on="Team", how="left")
            df_merged["Goals Scored"] = df_merged["Goals Scored"].fillna(0).astype(int)
            df_merged["Flag"] = df_merged["Flag"].fillna(DEFAULT_FLAG)
            df_merged["Stage"] = df_merged["Stage"].fillna("Group Stage")

            df_display = df_merged[[
                "Flag", "Participant Name", "Stage", "Goals Scored"
            ]].rename(columns={
                "Participant Name": "Player",
                "Stage": "Current Progress Stage",
                "Goals Scored": "Total Goals Scored"
            })

            df_display = df_display.sort_values(
                by=["Current Progress Stage", "Total Goals Scored", "Player"],
                key=lambda col: col.apply(
                    lambda val: STAGE_ORDER.index(val) if val in STAGE_ORDER else -1
                ) if col.name == "Current Progress Stage" else col,
                ascending=[False, False, True]
            ).reset_index(drop=True)

            st.dataframe(
                df_display,
                use_container_width=True,
                height=450,
                column_config={"Flag": st.column_config.ImageColumn("🏳️", width="small")}
            )

        with tab_bracket:
            st.subheader("Tournament Knockout Progression")
            st.caption("⬅️ Swipe horizontally to navigate the bracket stages ➡️")

            grid_cols = st.columns(len(BRACKET_STAGES))

            stage_geometry = {
                "Round of 32":    {"start_pads": 0,    "mid_pads": 0},
                "Round of 16":    {"start_pads": 1,    "mid_pads": 2.35},
                "Quarter-Finals": {"start_pads": 3.6,  "mid_pads": 7.4},
                "Semi-Finals":    {"start_pads": 8.5,  "mid_pads": 17.5},
                "Finals":         {"start_pads": 18.5, "mid_pads": 0},
            }

            # Build ordered (team_a, team_b) pairs per stage directly from the
            # deduped, id-keyed match list returned by the fetch function.
            # Each fixture id appears exactly once, so no card can render twice.
            ordered_bracket = {}
            for stage in BRACKET_STAGES:
                matches_for_stage = sorted(
                    bracket_matches.get(stage, []),
                    key=lambda m: (m.get("utcDate", ""), m.get("id", 0))
                )
                pairs = [
                    (m["homeTeam"]["name"] if m.get("homeTeam") else "TBD",
                     m["awayTeam"]["name"] if m.get("awayTeam") else "TBD")
                    for m in matches_for_stage
                ]
                slots = STAGE_SLOT_COUNTS[stage]
                while len(pairs) < slots:
                    pairs.append(("TBD", "TBD"))
                ordered_bracket[stage] = pairs[:slots]

            for col_idx, stage_title in enumerate(BRACKET_STAGES):
                with grid_cols[col_idx]:
                    st.markdown(f"⚡ **{stage_title}**")
                    geom = stage_geometry[stage_title]
                    pairs = ordered_bracket[stage_title]

                    if geom["start_pads"] > 0:
                        height_px = geom["start_pads"] * 54
                        st.html(f"<div style='height: {height_px}px;'></div>")

                    for slot_idx, (team_a, team_b) in enumerate(pairs):
                        with st.container(border=True):
                            st.html(render_team_markup(df_teams, team_to_player, team_a, stage_title, opponent_name=team_b))
                            st.html("<div style='margin: 3px 0; border-top: 1px dashed #ddd;'></div>")
                            st.html(render_team_markup(df_teams, team_to_player, team_b, stage_title, opponent_name=team_a))

                        if slot_idx < (len(pairs) - 1) and geom["mid_pads"] > 0:
                            mid_height_px = geom["mid_pads"] * 54
                            st.html(f"<div style='height: {mid_height_px}px;'></div>")

        with tab_feed:
            # --- BST Night Logic Processing Block (4PM to 6AM Window) ---
            now_utc = datetime.now(timezone.utc)
            now_bst = now_utc + timedelta(hours=1)

            # Day cycle refreshes exactly at 6:00 AM BST
            if now_bst.hour >= 6:
                target_date = now_bst.date()
            else:
                target_date = now_bst.date() - timedelta(days=1)

            def get_night_matches(target_d):
                start_window = datetime.combine(target_d, datetime.min.time()).replace(hour=16, tzinfo=timezone.utc)
                end_window = start_window + timedelta(hours=14)

                matches = []
                for m in raw_matches:
                    m_utc = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                    m_bst = m_utc + timedelta(hours=1)

                    if start_window <= m_bst <= end_window:
                        matches.append((m, m_bst))
                return matches

            tonight_matches = get_night_matches(target_date)
            last_night_matches = get_night_matches(target_date - timedelta(days=1))

            feed_matches = last_night_matches
            feed_title = "🌙 Last Night's Action"

            if not feed_matches:
                feed_title = "🕒 Recent Finished Match Results"
                recent = [m for m in raw_matches if m.get("status") == "FINISHED"]
                feed_matches = [(m, None) for m in recent[-5:]]

            st.subheader(feed_title)
            if not feed_matches:
                st.write("No matches found in this window.")
            else:
                for m, m_time in feed_matches:
                    h_team = m["homeTeam"]["name"]
                    a_team = m["awayTeam"]["name"]
                    h_player = team_to_player.get(h_team.upper(), h_team)
                    a_player = team_to_player.get(a_team.upper(), a_team)

                    h_flag = team_flag(df_teams, h_team)
                    a_flag = team_flag(df_teams, a_team)

                    if m["status"] == "FINISHED":
                        score_str = f"{m['score']['fullTime']['home']} - {m['score']['fullTime']['away']}"
                        goals = m.get("goals", [])
                        goal_info = ", ".join([str(g['minute']) + "'" for g in goals]) if goals else "not provided"

                        st.markdown(
                            f"<img src='{h_flag}' width='20' style='vertical-align: middle; margin-right: 6px;'>"
                            f"**{h_player}** &nbsp; `{score_str}` &nbsp; **{a_player}**"
                            f"<img src='{a_flag}' width='20' style='vertical-align: middle; margin-left: 6px;'>",
                            unsafe_allow_html=True
                        )

                        with st.spinner("✨ AI is analyzing last night's action..."):
                            summary = get_gemini_summary(m.get("id"), h_player, a_player, score_str, goal_info)
                        st.write(summary)

            st.divider()

            st.subheader("🕒 Tonight's Schedule")
            if not tonight_matches:
                st.write("No fixtures carded for tonight.")
            else:
                for m, m_time in tonight_matches:
                    h_team = m["homeTeam"]["name"]
                    a_team = m["awayTeam"]["name"]
                    h_player = team_to_player.get(h_team.upper(), h_team)
                    a_player = team_to_player.get(a_team.upper(), a_team)

                    h_flag = team_flag(df_teams, h_team)
                    a_flag = team_flag(df_teams, a_team)

                    if m["status"] == "FINISHED":
                        score_str = f"{m['score']['fullTime']['home']} - {m['score']['fullTime']['away']}"

                        st.markdown(
                            f"✅ <img src='{h_flag}' width='20' style='vertical-align: middle; margin-right: 6px;'>"
                            f"**{h_player}** &nbsp; `{score_str}` &nbsp; **{a_player}**"
                            f"<img src='{a_flag}' width='20' style='vertical-align: middle; margin-left: 6px;'>",
                            unsafe_allow_html=True
                        )

                        goals = m.get("goals", [])
                        goal_info = ", ".join([str(g['minute']) + "'" for g in goals]) if goals else "not provided"

                        with st.spinner("✨ AI is wrapping up the match summary..."):
                            summary = get_gemini_summary(m.get("id"), h_player, a_player, score_str, goal_info)
                        if summary:
                            st.write(summary)

                    elif m["status"] in ["IN_PLAY", "PAUSED"]:
                        score_str = f"{m['score']['fullTime'].get('home', 0)} - {m['score']['fullTime'].get('away', 0)}"

                        st.markdown(
                            f"🔴 **LIVE NOW** | <img src='{h_flag}' width='20' style='vertical-align: middle; margin-right: 6px;'>"
                            f"**{h_player}** &nbsp; `{score_str}` &nbsp; **{a_player}**"
                            f"<img src='{a_flag}' width='20' style='vertical-align: middle; margin-left: 6px;'>",
                            unsafe_allow_html=True
                        )

                    else:
                        h_goals = df_teams[df_teams["Team"] == h_team]["Goals Scored"].sum() if not df_teams.empty else 0
                        a_goals = df_teams[df_teams["Team"] == a_team]["Goals Scored"].sum() if not df_teams.empty else 0
                        total_w = h_goals + a_goals + 2

                        h_prob = (h_goals + 1) / total_w
                        a_prob = (a_goals + 1) / total_w

                        st.markdown(
                            f"<img src='{h_flag}' width='20' style='vertical-align: middle; margin-right: 6px;'>"
                            f"**{h_player}** `{h_prob:.0%}` &nbsp; vs &nbsp; `{a_prob:.0%}` **{a_player}**"
                            f"<img src='{a_flag}' width='20' style='vertical-align: middle; margin-left: 6px;'>",
                            unsafe_allow_html=True
                        )
                        if m_time:
                            st.caption(f"Kickoff: {m_time.strftime('%H:%M')} (BST)")

                        with st.spinner("🤖 AI is generating the match preview..."):
                            preview = get_gemini_preview(m.get("id"), h_player, a_player, h_prob, a_prob)
                        if preview:
                            st.write(preview)

    # -------------------------------------------------------------
    # PANEL 2: PASSWORD PROTECTED ADMIN PANEL
    # -------------------------------------------------------------
    elif app_view == "🔐 Admin Control Panel":
        st.title("🔐 Admin Controller Dashboard")
        if "admin_authenticated" not in st.session_state:
            st.session_state["admin_authenticated"] = False

        if not st.session_state["admin_authenticated"]:
            st.markdown("### Authorization Required")
            with st.form("auth_form"):
                pass_input = st.text_input("Master Verification Key", type="password")
                submit_auth = st.form_submit_button("Verify Access Rights")
                if submit_auth:
                    try:
                        target_pass = st.secrets["passwords"]["admin_password"]
                    except Exception:
                        st.error("No server environment secret found.")
                        target_pass = None
                    if target_pass and pass_input == target_pass:
                        st.session_state["admin_authenticated"] = True
                        st.rerun()
                    else:
                        st.error("Authentication handshake failed.")
        else:
            st.sidebar.info("Authorized Workspace Active")
            if st.sidebar.button("Terminate Session (Log Out)"):
                st.session_state["admin_authenticated"] = False
                st.rerun()

            adm_t1, adm_t2 = st.tabs(["👤 Participant Assignment Engine", "⚠️ Database Reset Switch"])

            with adm_t1:
                st.subheader("📝 Live Participant Registry Editor")
                available_teams = sorted(df_teams["Team"].dropna().astype(str).tolist()) if not df_teams.empty else []

                edited_p_df = st.data_editor(
                    df_participants,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Participant Name": st.column_config.TextColumn("Player Name", required=True),
                        "Team Assigned": st.column_config.SelectboxColumn("Assigned Country", options=available_teams, required=True)
                    },
                    key="participant_grid_editor"
                )

                if st.button("💾 Save Participant Grid Changes", type="primary"):
                    try:
                        with st.spinner("Synchronizing database registry..."):
                            edited_p_df["Participant Name"] = edited_p_df["Participant Name"].astype(str).str.strip()
                            edited_p_df["Team Assigned"] = edited_p_df["Team Assigned"].astype(str).str.strip()
                            edited_p_df = edited_p_df.dropna(subset=["Participant Name"])
                            conn.update(worksheet="Participants_Wimbledon", data=edited_p_df)
                            st.cache_data.clear()
                            st.success("🎉 Participant registry updated successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to commit participant updates: {e}")

            with adm_t2:
                st.subheader("Destructive Matrix Synchronization Block")
                st.warning("Clears user-registry rows completely.")
                safety_checkbox = st.checkbox("I explicitly acknowledge that this operational process cannot be undone.")
                if st.button("Wipe & Clear Global Datastores", disabled=not safety_checkbox):
                    blank_p = pd.DataFrame(columns=["Participant Name", "Team Assigned"])
                    try:
                        conn.update(worksheet="Participants_Wimbledon", data=blank_p)
                        st.cache_data.clear()
                        st.success("Google Spreadsheet cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Execution failed: {e}")


if __name__ == "__main__":
    main()
