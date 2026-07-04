"""
World Cup Sweepstake Hub — Streamlit app.

Data sources:
- football-data.org  -> live scores, standings, bracket progression
- The Odds API       -> pre-match "Draw No Bet" odds (2-way outright winner)
- Groq (llama-3.3)   -> AI-written match summaries/previews

Page layout:
- Public dashboard: Leaderboard tab, Bracket tab, Live Action feed tab
- Admin panel: edit participants, wipe the sheet
"""

import streamlit as st
import pandas as pd
import requests
from streamlit_gsheets import GSheetsConnection
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from groq import Groq

st.set_page_config(
    page_title="Tournament Sweepstake Hub",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =================================================================
# CONSTANTS
# =================================================================

STAGE_ORDER = ["Group Stage", "Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Finals", "Champion"]
BRACKET_STAGES = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Finals"]
UNRESOLVED_TEAM_NAMES = ["TBD", "TBC"]
LIVE_OR_DONE_STATUSES = ["FINISHED", "IN_PLAY", "PAUSED", "LIVE"]
DEFAULT_FLAG = "https://flagcdn.com/w40/un.png"

API_STAGE_MAP = {
    "GROUP_STAGE": "Group Stage",
    "ROUND_OF_32": "Round of 32",
    "LAST_32": "Round of 32",
    "ROUND_OF_16": "Round of 16",
    "LAST_16": "Round of 16",
    "QUARTER_FINALS": "Quarter-Finals",
    "SEMI_FINALS": "Semi-Finals",
    "FINAL": "Finals"
}

# Bracket column layout: how much vertical space to pad above the first slot
# and between slots, in "row units" (each unit = 54px), so that matchups line
# up visually with the pair of teams that feeds into them from the round before.
BRACKET_GEOMETRY = {
    "Round of 32":    {"start_pads": 0,    "mid_pads": 0,    "total_slots": 16},
    "Round of 16":    {"start_pads": 1.05, "mid_pads": 2.48, "total_slots": 8},
    "Quarter-Finals": {"start_pads": 3.6,  "mid_pads": 7.8,  "total_slots": 4},
    "Semi-Finals":    {"start_pads": 8.95, "mid_pads": 18.5, "total_slots": 2},
    "Finals":         {"start_pads": 20.0, "mid_pads": 0,    "total_slots": 1}
}

# Manually curated one-off tournament facts (data provider doesn't expose these).
TOURNAMENT_MILESTONES = [
    ("🎯 First Goal from a Penalty", "Switzerland", "Successfully converted against Qatar."),
    ("🤦‍♂️ First Own Goal", "Paraguay", "Deflected into their own net against USA."),
    ("🟥 First Red Card", "South Africa", "Sent off during the tournament opener sequence."),
    ("📉 Worst Team (Exited in Group Stage)", "Iraq", "Eliminated with 0 Points and a -11 Goal Difference."),
]

FOOTER_CAPTION = "Created By Devansh Gupta using Gemini"


# =================================================================
# SMALL GENERAL-PURPOSE HELPERS
# =================================================================

def normalize_team_name(name):
    """football-data.org and The Odds API both call the USA 'United States' — align it
    with the shorter name used everywhere else in the app."""
    return "USA" if name == "United States" else name

def shorten_name(full_name_str):
    """Converts a full name like 'James O'Doherty' into 'James O'."""
    if not full_name_str or str(full_name_str).upper() in ["NAN", "NONE"]:
        return ""
    parts = str(full_name_str).strip().split()
    if not parts:
        return ""
    first_name = parts[0]
    if len(parts) > 1 and parts[1]:
        return f"{first_name} {parts[1][0].upper()}"
    return first_name

def check_secrets():
    required = {
        "football_api": ["api_token"],
        "odds_api": ["odds_api_key"],
        "passwords": ["admin_password"],
        "connections": ["gsheets"],
        "groq_api": ["groq_api_key"]
    }
    for section, keys in required.items():
        if section not in st.secrets:
            return False, f"Missing section: [{section}] in secrets.toml"
        for key in keys:
            if key not in st.secrets[section]:
                return False, f"Missing key: {key} in [{section}]"
    return True, ""

def get_team_flag(df_teams, team_name):
    if df_teams.empty:
        return DEFAULT_FLAG
    match = df_teams[df_teams["Team"] == team_name]
    return match["Flag"].values[0] if not match.empty else DEFAULT_FLAG

def get_team_player(df_teams, team_name):
    """Returns the sweepstake participant's formatted name for a team, or the team
    name itself if nobody's been assigned to it (or no team data is loaded yet)."""
    if df_teams.empty:
        return team_name
    match = df_teams[df_teams["Team"] == team_name]
    return match["Player"].values[0] if not match.empty else team_name

def get_outright_odds(odds_lookup, team_name, opponent_name):
    """Returns team_name's Draw No Bet price for its upcoming fixture vs opponent_name,
    checking both possible home/away orderings. None if we don't have odds for it."""
    fixture = odds_lookup.get(f"{team_name}_vs_{opponent_name}")
    if fixture:
        return fixture.get("home_win")
    fixture = odds_lookup.get(f"{opponent_name}_vs_{team_name}")
    if fixture:
        return fixture.get("away_win")
    return None

def flag_img_html(flag_url, width=20, margin_side="right"):
    return f"<img src='{flag_url}' width='{width}' style='vertical-align: middle; margin-{margin_side}: 6px;'>"


# =================================================================
# AI MATCH COMMENTARY (via Groq's llama-3.3-70b)
# =================================================================


AI_COMMENTATOR_STYLE = (
    "You are a charismatic, playful, and incredibly witty football commentator. "
    "Your tone should be clever, clever, and highly creative, but always remaining "
    "positive and fun. Strictly avoid mean-spirited roasts, dark humor, or cynicism."
)

def _ask_groq(user_prompt, fallback=None):
    """Shared call path for both AI commentary functions below."""
    groq_api_key = st.secrets.get("groq_api", {}).get("groq_api_key")
    if not groq_api_key or not Groq:
        return fallback
    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": AI_COMMENTATOR_STYLE},
                {"role": "user", "content": user_prompt}
            ]
        )
        if response and response.choices:
            return response.choices[0].message.content.strip().replace('"', '').replace('*', '')
        return fallback
    except Exception:
        return fallback

@st.cache_data(ttl=1800, show_spinner=False)
def get_ai_match_summary(match_id, h_player, a_player, score_str, goal_info):
    prompt = (
        f"Context: Football tournament match result.\n"
        f"Participants: {h_player} vs {a_player}.\n"
        f"Final Score: {score_str}.\n"
        f"Goal details: {goal_info}.\n"
        f"Instruction: Generate a one-sentence fun, dramatic, and witty post-match commentary. "
        f"Do not include any emojis. Do not format with markdown bolding or asterisks. Be punchy."
    )
    return _ask_groq(prompt, fallback="What an incredible finish to this matchup!")

@st.cache_data(ttl=1800, show_spinner=False)
def get_ai_match_preview(match_id, h_player, a_player, h_odd, a_odd):
    odds_context = (
        f"Decimal odds to win: {h_player} is priced at {h_odd:.2f}, while {a_player} is priced at {a_odd:.2f}."
        if h_odd and a_odd else "Odds are currently closely matched or pending."
    )
    prompt = (
        f"Context: Upcoming tournament sweepstake football match.\n"
        f"Matchup: {h_player} vs {a_player}.\n"
        f"{odds_context}\n"
        f"Instruction: Generate a short, single-sentence dramatic narrative intro line hype setup based on who the bookmakers favor. "
        f"Do not include the numeric odds values in the response. "
        f"No emojis. No asterisks. The response should be funny."
    )
    return _ask_groq(prompt, fallback=None)


# =================================================================
# FOOTBALL-DATA.ORG INGESTION
# =================================================================


@st.cache_data(ttl=30)
def fetch_live_tournament_data(api_token):
    """
    Pulls every World Cup match from football-data.org and derives, from
    that raw match list, everything the dashboard needs:
      - `stats`: per-team record (stage reached, goals, knocked-out status...)
      - `stage_matchups`: which two teams played each other, per bracket stage
      - `all_matches`: the raw match list (used by the live-feed tab)
      - `golden_boot`: goals scored per player
      - `biggest_wins`: finished matches sorted later by goal margin
    """
    stats = {}
    stage_matchups = {stage: [] for stage in STAGE_ORDER if stage != "Champion"}
    all_matches = []
    golden_boot = {}
    biggest_wins = []

    if not api_token or str(api_token).strip() == "":
        st.sidebar.error("⚠️ API Token is missing. Please check your secrets configuration.")
        return stats, stage_matchups, all_matches, {}, []

    try:
        headers = {"X-Auth-Token": str(api_token).strip()}
        matches_res = requests.get(
            "https://api.football-data.org/v4/competitions/WC/matches", headers=headers, timeout=10
        )

        if matches_res.status_code == 200:
            matches = sorted(matches_res.json().get("matches", []), key=lambda x: x.get("id", 0))

            for match in matches:
                stage = match["stage"]
                status = match["status"]
                home_obj = match.get("homeTeam")
                away_obj = match.get("awayTeam")

                home_team = normalize_team_name((home_obj.get("name") if home_obj else "TBD") or "TBD")
                away_team = normalize_team_name((away_obj.get("name") if away_obj else "TBD") or "TBD")
                if home_obj:
                    match["homeTeam"]["name"] = home_team
                if away_obj:
                    match["awayTeam"]["name"] = away_team
                all_matches.append(match)

                # Register any newly-seen team.
                for team_name, team_meta in [(home_team, home_obj), (away_team, away_obj)]:
                    if team_name not in UNRESOLVED_TEAM_NAMES and team_meta and team_name not in stats:
                        stats[team_name] = {
                            "Team": team_name, "Flag": team_meta.get("crest", DEFAULT_FLAG),
                            "Won": 0, "Lost": 0, "Points": 0, "Goals Scored": 0,
                            "Stage": "Group Stage", "Status": "Active",
                            "Match Scores": {}, "Live Stages": []
                        }

                current_stage = API_STAGE_MAP.get(stage)
                if current_stage:
                    # Advance each team's "furthest stage reached" marker.
                    for team_name in [home_team, away_team]:
                        if team_name in UNRESOLVED_TEAM_NAMES:
                            continue
                        reached_rank = STAGE_ORDER.index(stats[team_name]["Stage"]) if stats[team_name]["Stage"] in STAGE_ORDER else 0
                        if STAGE_ORDER.index(current_stage) > reached_rank:
                            stats[team_name]["Stage"] = current_stage

                    if current_stage in stage_matchups:
                        pair = (home_team, away_team)
                        if pair not in stage_matchups[current_stage]:
                            stage_matchups[current_stage].append(pair)

                if status in LIVE_OR_DONE_STATUSES:
                    full_time = match.get("score", {}).get("fullTime", {})
                    home_goals = full_time.get("home", 0) or 0
                    away_goals = full_time.get("away", 0) or 0

                    if home_team not in UNRESOLVED_TEAM_NAMES:
                        stats[home_team]["Goals Scored"] += home_goals
                    if away_team not in UNRESOLVED_TEAM_NAMES:
                        stats[away_team]["Goals Scored"] += away_goals

                    if current_stage:
                        if home_team not in UNRESOLVED_TEAM_NAMES:
                            stats[home_team]["Match Scores"][current_stage] = home_goals
                        if away_team not in UNRESOLVED_TEAM_NAMES:
                            stats[away_team]["Match Scores"][current_stage] = away_goals

                        if status in ["IN_PLAY", "PAUSED", "LIVE"]:
                            if home_team not in UNRESOLVED_TEAM_NAMES:
                                stats[home_team]["Live Stages"].append(current_stage)
                            if away_team not in UNRESOLVED_TEAM_NAMES:
                                stats[away_team]["Live Stages"].append(current_stage)

                # Golden boot tally from in-match goal events.
                for goal in (match.get("goals") or match.get("score", {}).get("goals", [])):
                    scorer_name = goal.get("scorer", {}).get("name")
                    if not scorer_name:
                        continue
                    scorer_team = normalize_team_name(goal.get("team", {}).get("name") or "Unknown")
                    golden_boot.setdefault(scorer_name, {"Scorer": scorer_name, "Team": scorer_team, "Goals": 0})
                    golden_boot[scorer_name]["Goals"] += 1

                # Once a match is finished: record it as a possible "biggest win",
                # and progress the winner/loser through the bracket.
                is_valid_finish = (
                    status == "FINISHED"
                    and (current_stage or stage == "GROUP_STAGE")
                    and home_team not in UNRESOLVED_TEAM_NAMES
                    and away_team not in UNRESOLVED_TEAM_NAMES
                )
                if is_valid_finish:
                    full_time = match.get("score", {}).get("fullTime", {})
                    h_g = full_time.get("home", 0) or 0
                    a_g = full_time.get("away", 0) or 0
                    margin = abs(h_g - a_g)
                    if margin > 0:
                        biggest_wins.append({
                            "Match": f"{home_team} vs {away_team}", "Score": f"{h_g} - {a_g}",
                            "Margin": margin, "Goals": max(h_g, a_g),
                            "Winner": home_team if h_g > a_g else away_team
                        })

                    winner = match.get("score", {}).get("winner")
                    if current_stage and current_stage != "Group Stage":
                        next_stage = current_stage
                        if current_stage in STAGE_ORDER:
                            curr_idx = STAGE_ORDER.index(current_stage)
                            if curr_idx + 1 < len(STAGE_ORDER):
                                next_stage = STAGE_ORDER[curr_idx + 1]

                        if winner == "HOME_TEAM":
                            stats[away_team]["Status"] = "Knocked Out"
                            stats[away_team]["Stage"] = current_stage
                            stats[home_team]["Stage"] = next_stage
                        elif winner == "AWAY_TEAM":
                            stats[home_team]["Status"] = "Knocked Out"
                            stats[home_team]["Stage"] = current_stage
                            stats[away_team]["Stage"] = next_stage

                        if stage == "FINAL" and winner:
                            champion = home_team if winner == "HOME_TEAM" else away_team
                            stats[champion]["Stage"] = "Champion"
                            stats[champion]["Status"] = "Winner"

        # Top scorers endpoint overrides the goal counts derived above with the
        # authoritative tournament-wide tally (covers scorers from matches we
        # may not have fully parsed goal-events for).
        scorers_res = requests.get(
            "https://api.football-data.org/v4/competitions/WC/scorers", headers=headers, timeout=10
        )
        if scorers_res.status_code == 200:
            for entry in scorers_res.json().get("scorers", []):
                scorer_name = entry.get("player", {}).get("name")
                if not scorer_name:
                    continue
                scorer_team = normalize_team_name(entry.get("team", {}).get("name") or "Unknown")
                golden_boot[scorer_name] = {"Scorer": scorer_name, "Team": scorer_team, "Goals": entry.get("goals", 0)}

    except Exception as e:
        st.sidebar.error(f"API Connection Issue: {e}")

    return stats, stage_matchups, all_matches, golden_boot, biggest_wins


# =================================================================
# THE ODDS API — DRAW NO BET (2-WAY OUTRIGHT WINNER) ODDS
# =================================================================


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_odds_data(odds_api_key, target_pairs):
    """
    Fetches "Draw No Bet" odds for FIFA World Cup fixtures from The Odds API
    (https://the-odds-api.com). Draw No Bet is a genuine 2-way market ("Odds
    for the match winner, excluding the draw outcome. A draw will result in
    a returned bet.") — this gives an outright home/away winner price
    without a draw leg, which is what we want to display since draws don't
    stand in the World Cup bracket.

    `target_pairs` is a frozenset of (home_team, away_team) tuples for the
    fixtures we actually need odds for (i.e. not yet finished/live), so we
    only spend API quota on matches that need it. Draw No Bet is an
    "additional market", so it must be fetched one event at a time via the
    /events/{eventId}/odds endpoint (unlike the bulk h2h/1X2 endpoint).
    """
    odds_lookup = {}
    if not odds_api_key or str(odds_api_key).strip() == "" or not target_pairs:
        return odds_lookup

    api_key = str(odds_api_key).strip()
    base_url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup"

    try:
        # Step 1: list current events. This endpoint does NOT count against quota.
        events_res = requests.get(f"{base_url}/events", params={"apiKey": api_key}, timeout=10)

        if events_res.status_code == 401:
            st.sidebar.error("⚠️ Odds API key is invalid or missing.")
            return odds_lookup
        if events_res.status_code != 200:
            st.sidebar.warning(f"⚠️ Odds API returned status {events_res.status_code} for events.")
            return odds_lookup

        for event in events_res.json():
            home_team = normalize_team_name(event.get("home_team", ""))
            away_team = normalize_team_name(event.get("away_team", ""))
            event_id = event.get("id")
            if (home_team, away_team) not in target_pairs or not event_id:
                continue

            # Step 2: per-event Draw No Bet odds. This DOES count against quota
            # (1 credit per region specified, for this one market).
            odds_res = requests.get(
                f"{base_url}/events/{event_id}/odds",
                params={"apiKey": api_key, "regions": "eu", "markets": "draw_no_bet", "oddsFormat": "decimal"},
                timeout=10
            )
            if odds_res.status_code == 429:
                st.sidebar.warning("⚠️ Odds API monthly quota exceeded.")
                break
            if odds_res.status_code != 200:
                continue

            home_prices, away_prices = [], []
            for bookmaker in odds_res.json().get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "draw_no_bet":
                        continue
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price")
                        if price is None:
                            continue
                        outcome_name = normalize_team_name(outcome.get("name", ""))
                        if outcome_name == home_team:
                            home_prices.append(price)
                        elif outcome_name == away_team:
                            away_prices.append(price)

            if home_prices or away_prices:
                odds_lookup[f"{home_team}_vs_{away_team}"] = {
                    "home_win": round(sum(home_prices) / len(home_prices), 2) if home_prices else None,
                    "away_win": round(sum(away_prices) / len(away_prices), 2) if away_prices else None,
                }

    except Exception as e:
        st.sidebar.warning(f"Odds API connection issue: {e}")

    return odds_lookup

def get_odds_target_pairs(raw_matches):
    """Fixtures still worth spending Odds API quota on: anything that hasn't
    kicked off yet. Once a match starts we show a score instead of odds."""
    pairs = set()
    for m in raw_matches:
        if m.get("status") in LIVE_OR_DONE_STATUSES:
            continue
        h_name = m.get("homeTeam", {}).get("name")
        a_name = m.get("awayTeam", {}).get("name")
        if h_name and a_name and h_name not in UNRESOLVED_TEAM_NAMES and a_name not in UNRESOLVED_TEAM_NAMES:
            pairs.add((h_name, a_name))
    return frozenset(pairs)


# =================================================================
# GOOGLE SHEETS — PARTICIPANTS & TEAM ASSIGNMENTS
# =================================================================


conn = st.connection("gsheets", type=GSheetsConnection)

def empty_participants_df():
    return pd.DataFrame(columns=["Participant Name", "Participant Name 2", "Team Assigned"])

def build_team_to_player_map(df_participants):
    """One team can have 1 or 2 participants attached to it (e.g. a couple
    sharing an entry) — this merges them into a single '/'-joined display string,
    keyed by the team name in upper case."""
    team_to_player = {}
    for _, row in df_participants.iterrows():
        team = str(row.get("Team Assigned", "")).strip().upper()
        if not team or team in ["NAN", "NONE"]:
            continue

        players = []
        for col in ["Participant Name", "Participant Name 2"]:
            if col in row:
                name = shorten_name(row.get(col, "")).strip()
                if name and name.upper() not in ["NAN", "NONE"]:
                    players.append(name)
        if not players:
            continue

        if team in team_to_player:
            existing = [p.strip() for p in team_to_player[team].split(" / ")]
            for p in players:
                if p not in existing:
                    team_to_player[team] += f" / {p}"
        else:
            team_to_player[team] = " / ".join(players)

    return team_to_player

def load_sweepstake_data(worksheet_name):
    """Reads the participant sheet, merges it with live tournament data from
    football-data.org, and returns everything the dashboard needs to render."""
    try:
        df_participants = conn.read(worksheet=worksheet_name, ttl=0)
        if df_participants is None:
            df_participants = empty_participants_df()
    except Exception as e:
        st.sidebar.warning(f"Connection to Sheets failed: {e}")
        df_participants = empty_participants_df()

    if df_participants.empty or "Participant Name" not in df_participants.columns or "Team Assigned" not in df_participants.columns:
        df_participants = empty_participants_df()
    else:
        df_participants = df_participants.dropna(subset=["Participant Name"])

    api_token = st.secrets.get("football_api", {}).get("api_token", "")
    live_stats, stage_matchups, raw_matches, golden_boot, biggest_wins = fetch_live_tournament_data(api_token)

    df_teams = pd.DataFrame(list(live_stats.values()))
    if df_teams.empty:
        df_teams = pd.DataFrame(columns=["Team", "Flag", "Won", "Lost", "Points", "Goals Scored", "Stage", "Status"])

    team_to_player = build_team_to_player_map(df_participants) if not df_participants.empty else {}

    def format_player_column(team_name):
        player = team_to_player.get(str(team_name).strip().upper())
        return f"{player} ({team_name})" if player else team_name

    df_teams["Player"] = df_teams["Team"].apply(format_player_column) if not df_teams.empty else []

    df_golden_boot = pd.DataFrame(list(golden_boot.values()))
    if df_golden_boot.empty:
        df_golden_boot = pd.DataFrame(columns=["Scorer", "Team", "Goals"])
    else:
        df_golden_boot = df_golden_boot.sort_values(by="Goals", ascending=False).reset_index(drop=True)

    df_biggest_wins = pd.DataFrame(biggest_wins)
    if df_biggest_wins.empty:
        df_biggest_wins = pd.DataFrame(columns=["Match", "Score", "Margin", "Goals", "Winner"])
    else:
        df_biggest_wins = df_biggest_wins.sort_values(by=["Margin", "Goals"], ascending=False).reset_index(drop=True)

    return df_participants, df_teams, stage_matchups, raw_matches, team_to_player, df_golden_boot, df_biggest_wins


# =================================================================
# DASHBOARD UI — SHARED CHROME (styling + sidebar)
# =================================================================


def inject_custom_css():
    st.markdown("""
        <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; margin-top: 0px !important; }
        [data-testid="stHeader"] { height: 0px !important; background: transparent !important; }

        @media (max-width: 800px) {
            div[data-testid="stHorizontalBlock"] { overflow-x: auto; flex-wrap: nowrap !important; gap: 1rem !important; padding-bottom: 15px; }
            div[data-testid="column"] { min-width: 250px !important; flex: 0 0 auto !important; }
        }

        button[data-baseweb="tab"] {
            background-color: #f1f5f9 !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px 8px 0px 0px !important;
            padding: 10px 16px !important;
            margin-right: 4px !important;
            transition: all 0.2s ease-in-out;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #0284c7 !important;
            color: white !important;
            font-weight: bold !important;
            border-color: #0284c7 !important;
        }

        @keyframes blinker { 50% { opacity: 0; } }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df_teams):
    """Renders the sidebar chrome and returns which top-level view the user picked."""
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
        return st.radio("Switch Dashboard View", ["📊 Public Fan Dashboard", "🔐 Admin Control Panel"])


# =================================================================
# DASHBOARD UI — LEADERBOARD TAB
# =================================================================

def render_leaderboard_tab(df_teams, df_golden_boot, df_biggest_wins, team_to_player):
    st.subheader("Overall Standings")

    if not df_teams.empty:
        df_lead = df_teams.copy()
        df_lead["Goals Scored"] = df_lead["Goals Scored"].fillna(0).astype(int)
        df_lead["Flag"] = df_lead["Flag"].fillna(DEFAULT_FLAG)
        df_lead["Stage"] = df_lead["Stage"].fillna("Group Stage")

        df_display = df_lead[["Flag", "Player", "Stage", "Goals Scored"]].rename(columns={
            "Stage": "Current Progress Stage", "Goals Scored": "Total Goals Scored"
        })
        stage_rank = lambda val: STAGE_ORDER.index(val) if val in STAGE_ORDER else -1
        df_display = df_display.sort_values(
            by=["Current Progress Stage", "Total Goals Scored", "Player"],
            key=lambda col: col.apply(stage_rank) if col.name == "Current Progress Stage" else col,
            ascending=[False, False, True]
        ).reset_index(drop=True)

        st.dataframe(
            df_display, use_container_width=True, height=450,
            column_config={"Flag": st.column_config.ImageColumn("🏳️", width="small")},
            hide_index=True
        )
    else:
        st.info("No standings data available yet.")

    st.divider()

    def format_owner(team_name):
        player = team_to_player.get(str(team_name).strip().upper())
        return f"{player} ({team_name})" if player else ""

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("⚽ Golden Boot")
        if not df_golden_boot.empty:
            df_golden_boot = df_golden_boot.copy()
            df_golden_boot["Owner"] = df_golden_boot["Team"].apply(format_owner)
            st.dataframe(df_golden_boot[["Scorer", "Team", "Owner", "Goals"]].head(10), use_container_width=True, hide_index=True)
        else:
            st.write("Data currently unavailable.")

    with col_b:
        st.subheader("🔥 Biggest Win")
        if not df_biggest_wins.empty:
            df_biggest_wins = df_biggest_wins.copy()
            df_biggest_wins["Owner"] = df_biggest_wins["Winner"].apply(format_owner)
            st.dataframe(
                df_biggest_wins[["Winner", "Match", "Score", "Owner"]].head(5),
                use_container_width=True, hide_index=True,
                column_config={"Owner": "Sweepstake Owner"}
            )
        else:
            st.write("Data currently unavailable.")

    render_milestones_section(df_teams, team_to_player)

    st.divider()
    st.subheader("🚩 Most Corners")
    st.info("Corner kick statistics are not provided by the current tournament data provider (football-data.org).")

    st.space()
    st.caption(FOOTER_CAPTION)

def render_milestones_section(df_teams, team_to_player):
    st.divider()
    st.subheader("🏆 Tournament Milestone Bounties")
    st.caption("Special sweepstake milestones achieved during the tournament (determined chronologically by non-simultaneous match order).")

    def render_milestone_card(title, team_name, metric_detail):
        entrant = team_to_player.get(str(team_name).strip().upper(), "Unassigned")
        if entrant != "Unassigned":
            entrant = f"{entrant} ({team_name})"

        match_row = df_teams[df_teams["Team"] == team_name] if not df_teams.empty else pd.DataFrame()
        if not match_row.empty and match_row.iloc[0].get("Flag"):
            flag_html = f"<img src='{match_row.iloc[0]['Flag']}' width='26' style='vertical-align: middle; margin-right: 8px; border: 1px solid #e2e8f0; border-radius: 3px; flex-shrink: 0;'>"
        else:
            flag_html = "🏳️"

        st.markdown(
            f"""
            <div style="background-color: #ffffff; padding: 16px; border-radius: 10px; border: 1px solid #e2e8f0; border-left: 5px solid #38bdf8; margin-bottom: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h5 style="margin: 0 0 6px 0; color: #475569; font-size: 0.95rem; font-weight: 600; letter-spacing: -0.01em;">{title}</h5>
                <p style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #0f172a; display: flex; align-items: center; flex-wrap: nowrap;">
                    {flag_html}
                    <span style="margin-right: 6px;">{team_name}</span>
                    <span style="color: #64748b; font-weight: 400; margin: 0 4px;">—</span>
                    <span style="color: #0284c7; margin-left: 2px;">{entrant}</span>
                </p>
                <small style="color: #64748b; font-size: 0.85rem; display: block; margin-top: 6px; line-height: 1.3;">{metric_detail}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    columns = st.columns(2)
    for i, (title, team_name, detail) in enumerate(TOURNAMENT_MILESTONES):
        with columns[i % 2]:
            render_milestone_card(title, team_name, detail)


# =================================================================
# DASHBOARD UI — BRACKET (KNOCKOUT TREE) TAB
# =================================================================

def build_ordered_bracket(df_teams, global_matchups):
    """
    Produces {stage: [(team_a, team_b), ...]} with a fixed number of slots per
    stage (see BRACKET_GEOMETRY), so the bracket grid always has a consistent
    shape to render even before every fixture is confirmed.

    Two passes fill it in:
      1. Known matchups (from the raw match list) get slotted in, aligned
         under whichever pair of Round-of-32 teams feeds into them.
      2. Teams who've reached a stage but don't have a fixture yet (opponent
         still TBD) get placed into whichever bracket slot follows from their
         previous-round position.
    """
    ordered_bracket = {
        stage: [("TBD", "TBD") for _ in range(BRACKET_GEOMETRY[stage]["total_slots"])]
        for stage in BRACKET_STAGES
    }

    for idx, pair in enumerate(global_matchups.get("Round of 32", [])):
        if idx < len(ordered_bracket["Round of 32"]):
            ordered_bracket["Round of 32"][idx] = pair

    for stage_idx in range(1, len(BRACKET_STAGES)):
        current_stage = BRACKET_STAGES[stage_idx]
        prev_stage = BRACKET_STAGES[stage_idx - 1]

        for t_a, t_b in global_matchups.get(current_stage, []):
            target_slot = None
            for prev_idx, (p1, p2) in enumerate(ordered_bracket[prev_stage]):
                if (t_a != "TBD" and t_a in [p1, p2]) or (t_b != "TBD" and t_b in [p1, p2]):
                    target_slot = prev_idx // 2
                    break

            if target_slot is not None and target_slot < len(ordered_bracket[current_stage]):
                ordered_bracket[current_stage][target_slot] = (t_a, t_b)
            else:
                for idx, slot in enumerate(ordered_bracket[current_stage]):
                    if slot == ("TBD", "TBD"):
                        ordered_bracket[current_stage][idx] = (t_a, t_b)
                        break

    if not df_teams.empty:
        for _, row in df_teams.iterrows():
            team_name, team_stage = row["Team"], row["Stage"]
            if team_name in UNRESOLVED_TEAM_NAMES or team_stage not in ordered_bracket:
                continue

            already_placed = any(
                team_name in pair for stage in BRACKET_STAGES for pair in ordered_bracket.get(stage, [])
            )
            if already_placed:
                continue

            target_slot, is_bottom_position = None, False
            try:
                prev_stage_idx = BRACKET_STAGES.index(team_stage) - 1
                if prev_stage_idx >= 0:
                    prev_stage = BRACKET_STAGES[prev_stage_idx]
                    for prev_idx, (p1, p2) in enumerate(ordered_bracket[prev_stage]):
                        if team_name in (p1, p2):
                            target_slot = prev_idx // 2
                            is_bottom_position = (prev_idx % 2 != 0)
                            break
            except ValueError:
                pass

            if target_slot is not None and target_slot < len(ordered_bracket[team_stage]):
                t1, t2 = ordered_bracket[team_stage][target_slot]
                ordered_bracket[team_stage][target_slot] = (t1, team_name) if is_bottom_position else (team_name, t2)
            else:
                for idx, (t1, t2) in enumerate(ordered_bracket[team_stage]):
                    if t1 in UNRESOLVED_TEAM_NAMES:
                        ordered_bracket[team_stage][idx] = (team_name, t2)
                        break
                    elif t2 in UNRESOLVED_TEAM_NAMES:
                        ordered_bracket[team_stage][idx] = (t1, team_name)
                        break

    return ordered_bracket

def render_bracket_slot_html(df_teams, odds_lookup, team_name, current_stage_title, opponent_name=None):
    """HTML for one team's row within a single bracket-slot card: flag, name,
    and either their score (if the match has a result), their live badge, a
    trophy if they're champion, their Draw No Bet odds (if still upcoming),
    or a plain '--' placeholder."""
    if team_name in UNRESOLVED_TEAM_NAMES or not team_name:
        return "<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap;'><span style='color:gray; font-style:italic;'>🏳️ TBD</span></div>"

    match_row = df_teams[df_teams["Team"] == team_name]
    if match_row.empty:
        return f"<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap;'><span>🏳️ {team_name}</span></div>"

    row = match_row.iloc[0]
    display_name = row.get("Player", team_name)
    flag_html = f"<img src='{row['Flag']}' width='20' style='vertical-align: middle; margin-right: 5px; flex-shrink: 0;'>"

    score = row.get("Match Scores", {}).get(current_stage_title)
    is_live = current_stage_title in row.get("Live Stages", [])
    live_badge = "<span style='color: #2ecc71; font-size: 10px; font-weight: bold; margin-left: 5px; animation: blinker 1s linear infinite;'>● LIVE</span>" if is_live else ""
    trophy_suffix = " <span style='font-size:14px; margin-left:3px;'>🏆</span>" if (current_stage_title == "Finals" and row["Stage"] == "Champion") else ""

    if score is not None:
        score_html = f"<span style='font-size: 18px; font-weight: bold; margin-left: 8px; flex-shrink: 0;'>{score}</span>"
    elif opponent_name and opponent_name not in UNRESOLVED_TEAM_NAMES:
        team_odd = get_outright_odds(odds_lookup, team_name, opponent_name)
        if team_odd:
            score_html = f"<span style='font-size: 12px; color: #0284c7; font-weight: bold; margin-left: 8px;' title='Draw No Bet — outright winner odds, excludes the draw'>{team_odd:.2f}</span>"
        else:
            score_html = "<span style='font-size: 11px; color: #888; margin-left: 8px; font-style: italic;'>--</span>"
    else:
        score_html = ""

    if row["Status"] == "Knocked Out" and row["Stage"] == current_stage_title:
        name_html = f"<span style='color:gray; text-decoration:line-through;'>{display_name}</span><span style='font-size:11px; color:red; margin-left:3px;'>❌</span>"
    else:
        name_html = f"<span style='font-weight: bold;'>{display_name}</span>{live_badge}{trophy_suffix}"

    return (
        "<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap; width: 100%;'>"
        f"<div style='display: flex; align-items: center; min-width: 0; overflow: hidden; text-overflow: ellipsis;'>{flag_html}{name_html}</div>"
        f"{score_html}</div>"
    )

def render_bracket_tab(df_teams, global_matchups, odds_lookup):
    st.subheader("Tournament Knockout Progression")
    st.caption("⬅️ Swipe horizontally to navigate the bracket stages ➡️")

    ordered_bracket = build_ordered_bracket(df_teams, global_matchups)
    grid_cols = st.columns(len(BRACKET_STAGES))

    for col_idx, stage_title in enumerate(BRACKET_STAGES):
        with grid_cols[col_idx]:
            st.markdown(f"⚡ **{stage_title}**")
            geometry = BRACKET_GEOMETRY[stage_title]
            pairs = ordered_bracket.get(stage_title, [])

            if geometry["start_pads"] > 0:
                st.html(f"<div style='height: {geometry['start_pads'] * 54}px;'></div>")

            for slot_idx in range(geometry["total_slots"]):
                team_a, team_b = pairs[slot_idx] if slot_idx < len(pairs) else ("TBD", "TBD")

                with st.container(border=True):
                    st.html(render_bracket_slot_html(df_teams, odds_lookup, team_a, stage_title, opponent_name=team_b))
                    st.html("<div style='margin: 3px 0; border-top: 1px dashed #ddd;'></div>")
                    st.html(render_bracket_slot_html(df_teams, odds_lookup, team_b, stage_title, opponent_name=team_a))

                if slot_idx < (geometry["total_slots"] - 1) and geometry["mid_pads"] > 0:
                    st.html(f"<div style='height: {geometry['mid_pads'] * 54}px;'></div>")

    st.space()
    st.caption(FOOTER_CAPTION)


# =================================================================
# DASHBOARD UI — LIVE ACTION / FEED TAB
# =================================================================


def get_match_display_context(df_teams, match):
    """Common per-match display fields needed by every feed row: team names,
    their sweepstake player, and their flag."""
    h_team = match["homeTeam"]["name"]
    a_team = match["awayTeam"]["name"]
    return {
        "h_team": h_team, "a_team": a_team,
        "h_player": get_team_player(df_teams, h_team), "a_player": get_team_player(df_teams, a_team),
        "h_flag": get_team_flag(df_teams, h_team), "a_flag": get_team_flag(df_teams, a_team),
    }

def get_matches_in_window(raw_matches, start_window, end_window):
    matches = []
    for m in raw_matches:
        m_utc = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).replace(tzinfo=None)
        m_bst = m_utc + timedelta(hours=1)
        if start_window <= m_bst <= end_window:
            matches.append((m, m_bst))
    return matches

def render_finished_match_row(df_teams, match, prefix="", spinner_text="✨ AI is wrapping up the match summary..."):
    ctx = get_match_display_context(df_teams, match)
    score_str = f"{match['score']['fullTime']['home']} - {match['score']['fullTime']['away']}"

    st.markdown(
        f"{prefix}{flag_img_html(ctx['h_flag'])}**{ctx['h_player']}** &nbsp; `{score_str}` &nbsp; "
        f"**{ctx['a_player']}**{flag_img_html(ctx['a_flag'], margin_side='left')}",
        unsafe_allow_html=True
    )

    goals = match.get("goals", [])
    goal_info = ", ".join([f"{g['minute']}'" for g in goals]) if goals else "not provided"
    with st.spinner(spinner_text):
        summary = get_ai_match_summary(match.get("id", 0), ctx["h_player"], ctx["a_player"], score_str, goal_info)
    if summary:
        st.write(summary)

def render_live_match_row(df_teams, match):
    ctx = get_match_display_context(df_teams, match)
    score_str = f"{match['score']['fullTime'].get('home', 0)} - {match['score']['fullTime'].get('away', 0)}"

    st.markdown(
        f"🔴 **LIVE** | {flag_img_html(ctx['h_flag'])}**{ctx['h_player']}** &nbsp; `{score_str}` &nbsp; "
        f"**{ctx['a_player']}**{flag_img_html(ctx['a_flag'], margin_side='left')}",
        unsafe_allow_html=True
    )
    
def render_scheduled_match_row(df_teams, odds_lookup, match, m_time):
    ctx = get_match_display_context(df_teams, match)

    match_odds = odds_lookup.get(f"{ctx['h_team']}_vs_{ctx['a_team']}") or {}
    h_odd, a_odd = match_odds.get("home_win"), match_odds.get("away_win")
    
    odds_display = (
        f"To Win — {ctx['h_player']}: **{h_odd:.2f}** | {ctx['a_player']}: **{a_odd:.2f}**"
        if h_odd and a_odd else "Odds pending fixture finalization"
    )

    st.markdown(
        f"{flag_img_html(ctx['h_flag'])}**{ctx['h_player']}** &nbsp; vs &nbsp; "
        f"**{ctx['a_player']}**{flag_img_html(ctx['a_flag'], margin_side='left')}",
        unsafe_allow_html=True
    )
    st.markdown(f"<small style='color: #64748b; display: block; margin-top: -5px; margin-bottom: 5px;'>📊 {odds_display}</small>", unsafe_allow_html=True)
    if m_time:
        st.caption(f"Kickoff: {m_time.strftime('%H:%M')} (BST)")

    with st.spinner("🤖 AI is generating the match preview..."):
        # Pass odds to Groq instead of win probabilities
        preview = get_ai_match_preview(match.get("id", 0), ctx["h_player"], ctx["a_player"], h_odd, a_odd)
    if preview:
        st.write(preview)

def render_feed_tab(df_teams, raw_matches, odds_lookup):
    now_bst = datetime.utcnow() + timedelta(hours=1)
    target_date = now_bst.date() if now_bst.hour >= 6 else now_bst.date() - timedelta(days=1)

    def night_window(target_d):
        start = datetime.combine(target_d, datetime.min.time()).replace(hour=16)
        return get_matches_in_window(raw_matches, start, start + timedelta(hours=14))

    tonight_matches = night_window(target_date)
    last_night_matches = night_window(target_date - timedelta(days=1))

    feed_matches, feed_title = last_night_matches, "🌙 Last Night's Action"
    if not feed_matches:
        feed_title = "🕒 Recent Finished Match Results"
        recent_finished = [m for m in raw_matches if m.get("status") == "FINISHED"]
        feed_matches = [(m, None) for m in recent_finished[-5:]]

    st.subheader(feed_title)
    if not feed_matches:
        st.write("No matches found in this window.")
    else:
        for match, _ in feed_matches:
            render_finished_match_row(df_teams, match, spinner_text="✨ AI is analyzing last night's action...")

    st.divider()
    st.subheader("🕒 Tonight's Schedule")
    if not tonight_matches:
        st.write("No fixtures carded for tonight.")
    else:
        for match, m_time in tonight_matches:
            status = match["status"]
            if status == "FINISHED":
                render_finished_match_row(df_teams, match, prefix="✅ ")
            elif status in ["IN_PLAY", "PAUSED", "LIVE"]:
                render_live_match_row(df_teams, match)
            else:
                render_scheduled_match_row(df_teams, odds_lookup, match, m_time)

    st.space()
    st.caption(FOOTER_CAPTION)


# =================================================================
# ADMIN PANEL
# =================================================================


def render_admin_login():
    st.markdown("### Authorization Required")
    with st.form("auth_form"):
        pass_input = st.text_input("Master Verification Key", type="password")
        if st.form_submit_button("Verify Access Rights"):
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

def render_admin_panel(worksheet_name, df_teams, df_participants):
    st.title("🔐 Admin Controller Dashboard")
    st.session_state.setdefault("admin_authenticated", False)

    if not st.session_state["admin_authenticated"]:
        render_admin_login()
        return

    st.sidebar.info("Authorized Workspace Active")
    if st.sidebar.button("Terminated Session (Log Out)"):
        st.session_state["admin_authenticated"] = False
        st.rerun()

    tab_edit, tab_reset = st.tabs(["👤 Participant Assignment Engine", "⚠️ Database Reset Switch"])

    with tab_edit:
        st.subheader(f"📝 Live Participant Registry Editor ({worksheet_name})")
        available_teams = sorted(df_teams["Team"].dropna().astype(str).tolist()) if not df_teams.empty else []

        if "Participant Name 2" not in df_participants.columns:
            df_participants["Participant Name 2"] = ""

        edited_df = st.data_editor(
            df_participants,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Participant Name": st.column_config.TextColumn("Player 1 Name", required=True),
                "Participant Name 2": st.column_config.TextColumn("Player 2 Name", required=False),
                "Team Assigned": st.column_config.SelectboxColumn("Assigned Country", options=available_teams, required=True)
            },
            key=f"participant_grid_editor_{worksheet_name}"
        )

        if st.button("💾 Save Participant Grid Changes", type="primary"):
            try:
                with st.spinner("Synchronizing database registry..."):
                    edited_df["Participant Name"] = edited_df["Participant Name"].astype(str).str.strip()
                    edited_df = edited_df.dropna(subset=["Participant Name"])
                    conn.update(worksheet=worksheet_name, data=edited_df)
                    st.cache_data.clear()
                    st.success("🎉 Participant registry updated successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to commit participant updates: {e}")

    with tab_reset:
        st.subheader("Destructive Matrix Synchronization Block")
        st.warning(f"Clears user-registry rows completely from {worksheet_name}.")
        confirmed = st.checkbox("I explicitly acknowledge that this operational process cannot be undone.")
        if st.button("Wipe & Clear Global Datastores", disabled=not confirmed):
            try:
                conn.update(worksheet=worksheet_name, data=empty_participants_df())
                st.cache_data.clear()
                st.success("Google Spreadsheet cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Execution failed: {e}")


# =================================================================
# TOP-LEVEL PAGE ORCHESTRATION
# =================================================================


def render_dashboard(worksheet_name, dashboard_title):
    st_autorefresh(interval=30000, key=f"datarefresh_{worksheet_name}")

    secrets_ok, error_msg = check_secrets()
    if not secrets_ok:
        st.error("🚨 Missing Configuration")
        st.info(f"Details: {error_msg}")
        st.stop()

    df_participants, df_teams, global_matchups, raw_matches, team_to_player, df_golden_boot, df_biggest_wins = \
        load_sweepstake_data(worksheet_name)

    odds_api_key = st.secrets.get("odds_api", {}).get("odds_api_key", "")
    odds_lookup = fetch_odds_data(odds_api_key, get_odds_target_pairs(raw_matches))

    inject_custom_css()
    app_view = render_sidebar(df_teams)

    if app_view == "📊 Public Fan Dashboard":
        st.title(dashboard_title)
        st.caption("Updated automatically from official game knockout data feeds.")

        tab_feed, tab_leaderboard, tab_bracket = st.tabs(["📱 Live Action", "🏆 Leaderboard", "🌳 Draw"])

        with tab_leaderboard:
            render_leaderboard_tab(df_teams, df_golden_boot, df_biggest_wins, team_to_player)
        with tab_bracket:
            render_bracket_tab(df_teams, global_matchups, odds_lookup)
        with tab_feed:
            render_feed_tab(df_teams, raw_matches, odds_lookup)

    else:
        render_admin_panel(worksheet_name, df_teams, df_participants)


# =================================================================
# STREAMLIT PAGE ROUTING
# =================================================================

def page_wimbledon():
    render_dashboard(worksheet_name="Participants_Wimbledon", dashboard_title="📊 The Wimbledon World Cup")

def page_office():
    render_dashboard(worksheet_name="Participants_NB", dashboard_title="📊 The World Cup Sweepstake")

pg = st.navigation([
    st.Page(page_wimbledon, title="Wimbledon Hub", icon="🎾", url_path="wimbledon"),
    st.Page(page_office, title="Office Hub", icon="🏢", url_path="office")
], position="hidden")

if __name__ == "__main__":
    pg.run()
