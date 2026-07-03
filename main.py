import streamlit as st
import pandas as pd
import requests
from streamlit_gsheets import GSheetsConnection
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from groq import Groq

# Set page config MUST be the very first Streamlit command
st.set_page_config(
    page_title="Tournament Sweepstake Hub",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# GLOBAL CONSTANTS & UTILITIES
# -------------------------------------------------------------

STAGE_ORDER = ["Group Stage", "Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Finals", "Champion"]

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

def shorten_name(full_name_str):
    """
    Converts a full name like 'James O'Doherty' into 'James O' 
    and handles double-barrelled or regular surnames safely.
    """
    if not full_name_str or str(full_name_str).upper() in ["NAN", "NONE"]:
        return ""
    
    parts = str(full_name_str).strip().split()
    if not parts:
        return ""
    
    first_name = parts[0]
    
    if len(parts) > 1:
        surname_part = parts[1]
        if surname_part:
            return f"{first_name} {surname_part[0].upper()}"
            
    return first_name

def check_secrets():
    required = {
        "football_api": ["api_token"],
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

# -------------------------------------------------------------
# AI CORE UTILITIES (GROQ)
# -------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def get_gemini_summary(match_id, h_player, a_player, score_str, goal_info):
    groq_api_key = st.secrets.get("groq_api", {}).get("groq_api_key")
    if not groq_api_key or not Groq:
        return "AI integration offline."
        
    try:
        client = Groq(api_key=groq_api_key)
        system_instruction = (
            "You are a charismatic, playful, and incredibly witty football commentator. "
            "Your tone should be clever, clever, and highly creative, but always remaining "
            "positive and fun. Strictly avoid mean-spirited roasts, dark humor, or cynicism."
        )
        user_prompt = (
            f"Context: Football tournament match result.\n"
            f"Participants: {h_player} vs {a_player}.\n"
            f"Final Score: {score_str}.\n"
            f"Goal details: {goal_info}.\n"
            f"Instruction: Generate a one-sentence fun, dramatic, and witty post-match commentary. "
            f"Do not include any emojis. Do not format with markdown bolding or asterisks. Be punchy."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
        )
        if response and response.choices:
            ai_text = response.choices[0].message.content
            return ai_text.strip().replace('"', '').replace('*', '')
        return "The match result left the AI speechless!"
    except Exception:
        return "What an incredible finish to this matchup!"

@st.cache_data(ttl=1800, show_spinner=False)
def get_gemini_preview(match_id, h_player, a_player, h_prob, a_prob):
    groq_api_key = st.secrets.get("groq_api", {}).get("groq_api_key")
    if not groq_api_key or not Groq:
        return None
        
    try:
        client = Groq(api_key=groq_api_key)
        system_instruction = (
            "You are a charismatic, playful, and incredibly witty football commentator. "
            "Your tone should be clever, clever, and highly creative, but always remaining "
            "positive and fun. Strictly avoid mean-spirited roasts, dark humor, or cynicism."
        )
        user_prompt = (
            f"Context: Upcoming tournament sweepstake football match.\n"
            f"Matchup: {h_player} vs {a_player}.\n"
            f"Calculated Win Probabilities: {h_player} has a {h_prob:.0%} chance, while {a_player} has a {a_prob:.0%} chance.\n"
            f"Instruction: Generate a short, single-sentence dramatic narrative intro line hype setup. "
            f"Use predictions, but do not include percentages in response."
            f"No emojis. No asterisks. The response should be funny."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
        )
        if response and response.choices:
            ai_text = response.choices[0].message.content
            return ai_text.strip().replace('"', '').replace('*', '')
    except Exception:
        return None

# -------------------------------------------------------------
# DYNAMIC FOOTBALL API DATA INGESTION ENGINE
# -------------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_live_tournament_data(api_token):
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
        matches_url = "https://api.football-data.org/v4/competitions/WC/matches"
        matches_res = requests.get(matches_url, headers=headers, timeout=10)
        
        if matches_res.status_code == 200:
            matches_data = matches_res.json()
            matches = sorted(matches_data.get("matches", []), key=lambda x: x.get("id", 0))
            
            for match in matches:
                stage = match["stage"]
                home_obj = match.get("homeTeam")
                away_obj = match.get("awayTeam")
                
                home_team = (home_obj.get("name") if home_obj else "TBD") or "TBD"
                away_team = (away_obj.get("name") if away_obj else "TBD") or "TBD"

                if home_team == "United States": home_team = "USA"
                if away_team == "United States": away_team = "USA"
                
                if home_obj: match["homeTeam"]["name"] = home_team
                if away_obj: match["awayTeam"]["name"] = away_team

                all_matches.append(match)
                status = match["status"]
                
                for t, team_meta in [(home_team, home_obj), (away_team, away_obj)]:
                    if t not in ["TBD", "TBC"] and team_meta and t not in stats:
                        flag_url = team_meta.get("crest", "https://flagcdn.com/w40/un.png")
                        stats[t] = {
                            "Team": t, "Flag": flag_url, "Won": 0, "Lost": 0, "Points": 0,
                            "Goals Scored": 0, "Stage": "Group Stage", "Status": "Active",
                            "Match Scores": {}, "Live Stages": []
                        }
                
                current_stage_mapped = API_STAGE_MAP.get(stage)
                if current_stage_mapped:
                    for t in [home_team, away_team]:
                        if t not in ["TBD", "TBC"]:
                            current_team_rank = STAGE_ORDER.index(stats[t]["Stage"]) if stats[t]["Stage"] in STAGE_ORDER else 0
                            match_stage_rank = STAGE_ORDER.index(current_stage_mapped)
                            if match_stage_rank > current_team_rank:
                                stats[t]["Stage"] = current_stage_mapped
               
                    if current_stage_mapped in stage_matchups:
                        pair = (home_team, away_team)
                        if pair not in stage_matchups[current_stage_mapped]:
                            stage_matchups[current_stage_mapped].append(pair)
                
                if status in ["FINISHED", "IN_PLAY", "PAUSED", "LIVE"]:
                    score_data = match.get("score", {})
                    full_time = score_data.get("fullTime", {})
                    home_goals = full_time.get("home", 0) or 0
                    away_goals = full_time.get("away", 0) or 0
               
                    if home_team not in ["TBD", "TBC"]: stats[home_team]["Goals Scored"] += home_goals
                    if away_team not in ["TBD", "TBC"]: stats[away_team]["Goals Scored"] += away_goals
                
                    if current_stage_mapped:
                        if home_team not in ["TBD", "TBC"]: stats[home_team]["Match Scores"][current_stage_mapped] = home_goals
                        if away_team not in ["TBD", "TBC"]: stats[away_team]["Match Scores"][current_stage_mapped] = away_goals
         
                        if status in ["IN_PLAY", "PAUSED", "LIVE"]:
                            if home_team not in ["TBD", "TBC"]:
                                stats[home_team].setdefault("Live Stages", []).append(current_stage_mapped)
                            if away_team not in ["TBD", "TBC"]:
                                stats[away_team].setdefault("Live Stages", []).append(current_stage_mapped)

                match_goals = match.get("goals") or match.get("score", {}).get("goals", [])
                for goal in match_goals:
                    s_name = goal.get("scorer", {}).get("name")
                    g_team = (goal.get("team", {}).get("name") or "Unknown")
                    if g_team == "United States": g_team = "USA"
                    if s_name:
                        if s_name not in golden_boot:
                            golden_boot[s_name] = {"Scorer": s_name, "Team": g_team, "Goals": 0}
                        golden_boot[s_name]["Goals"] += 1
                
                if status == "FINISHED" and (current_stage_mapped or stage == "GROUP_STAGE") and home_team not in ["TBD", "TBC"] and away_team not in ["TBD", "TBC"]:
                    score_data = match.get("score", {})
                    ft = score_data.get("fullTime", {})
                    h_g = ft.get("home", 0) or 0
                    a_g = ft.get("away", 0) or 0
                    margin = abs(h_g - a_g)
                    if margin > 0:
                        biggest_wins.append({
                            "Match": f"{home_team} vs {away_team}",
                            "Score": f"{h_g} - {a_g}",
                            "Margin": margin,
                            "Goals": max(h_g, a_g),
                            "Winner": home_team if h_g > a_g else away_team
                        })

                    winner = match.get("score", {}).get("winner")
                    
                    if current_stage_mapped and current_stage_mapped != "Group Stage":
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

        scorers_url = "https://api.football-data.org/v4/competitions/WC/scorers"
        scorers_res = requests.get(scorers_url, headers=headers, timeout=10)
        if scorers_res.status_code == 200:
            scorers_data = scorers_res.json().get("scorers", [])
            for entry in scorers_data:
                s_name = entry.get("player", {}).get("name")
                g_team = entry.get("team", {}).get("name") or "Unknown"
                if g_team == "United States": g_team = "USA"
                g_count = entry.get("goals", 0)
                
                if s_name:
                    golden_boot[s_name] = {"Scorer": s_name, "Team": g_team, "Goals": g_count}

    except Exception as e:
        st.sidebar.error(f"API Connection Issue: {e}")
       
    return stats, stage_matchups, all_matches, golden_boot, biggest_wins

# Initialize Google Sheets Connection
conn = st.connection("gsheets", type=GSheetsConnection)

def database_load_pipeline(worksheet_name):
    """Parameterized database pipeline pointing to specific worksheets."""
    try:
        df_p = conn.read(worksheet=worksheet_name, ttl=0)
        if df_p is None:
            df_p = pd.DataFrame(columns=["Participant Name", "Participant Name 2", "Team Assigned"])
    except Exception as e:
        st.sidebar.warning(f"Connection to Sheets failed: {e}")
        df_p = pd.DataFrame(columns=["Participant Name", "Participant Name 2", "Team Assigned"])
        
    if df_p.empty or "Participant Name" not in df_p.columns or "Team Assigned" not in df_p.columns:
        df_p = pd.DataFrame(columns=["Participant Name", "Participant Name 2", "Team Assigned"])
    else:
        df_p = df_p.dropna(subset=["Participant Name"])
        
    api_token = st.secrets.get("football_api", {}).get("api_token", "")
    live_stats, stage_matchups, matches_list, golden_boot, biggest_wins = fetch_live_tournament_data(api_token)
    df_t = pd.DataFrame(list(live_stats.values()))
    
    if df_t.empty:
        df_t = pd.DataFrame(columns=["Team", "Flag", "Won", "Lost", "Points", "Goals Scored", "Stage", "Status"])
    
    # UNIFIED LOOKUP MAPPING ENGINE
    team_to_player = {}
    if not df_p.empty:
        for _, row in df_p.iterrows():
            team = str(row.get("Team Assigned", "")).strip().upper()
            if not team or team in ["NAN", "NONE"]:
                continue
                
            p1 = shorten_name(row.get("Participant Name", ""))
            p2 = shorten_name(row.get("Participant Name 2", "")) if "Participant Name 2" in row else ""
            p2 = p2.strip()
            
            row_players = []
            if p1 and p1.upper() not in ["NAN", "NONE"]:
                row_players.append(p1)
            if p2 and p2.upper() not in ["NAN", "NONE"]:
                row_players.append(p2)
                
            if not row_players:
                continue
                
            combined_row_string = " / ".join(row_players)
            
            if team in team_to_player:
                existing_players = [p.strip() for p in team_to_player[team].split(" / ")]
                for player in row_players:
                    if player not in existing_players:
                        team_to_player[team] += f" / {player}"
            else:
                team_to_player[team] = combined_row_string

    if not df_t.empty:
        def get_formatted_player(team_name):
            p_name = team_to_player.get(str(team_name).strip().upper())
            if p_name:
                return f"{p_name} ({team_name})"
            return team_name
        df_t["Player"] = df_t["Team"].apply(get_formatted_player)
    else:
        df_t["Player"] = []
        
    df_boot = pd.DataFrame(list(golden_boot.values()))
    df_wins = pd.DataFrame(biggest_wins)
    
    if df_wins.empty:
        df_wins = pd.DataFrame(columns=["Match", "Score", "Margin", "Goals", "Winner"])
    else:
        df_wins = df_wins.sort_values(by=["Margin", "Goals"], ascending=False).reset_index(drop=True)

    if df_boot.empty:
        df_boot = pd.DataFrame(columns=["Scorer", "Team", "Goals"])
    else:
        df_boot = df_boot.sort_values(by="Goals", ascending=False).reset_index(drop=True)

    return df_p, df_t, stage_matchups, matches_list, team_to_player, df_boot, df_wins

# -------------------------------------------------------------
# CORE DASHBOARD RENDERER
# -------------------------------------------------------------

def render_dashboard(worksheet_name, dashboard_title):
    st_autorefresh(interval=30000, key=f"datarefresh_{worksheet_name}")

    secrets_ok, error_msg = check_secrets()
    if not secrets_ok:
        st.error("🚨 Missing Configuration")
        st.info(f"Details: {error_msg}")
        st.stop()

    df_participants, df_teams, global_matchups, raw_matches, team_to_player, df_boot, df_wins = database_load_pipeline(worksheet_name)

    odds_lookup = {}
    for m in raw_matches:
        m_id = m.get("id")
        h = m.get("homeTeam", {}).get("name")
        a = m.get("awayTeam", {}).get("name")
        if h and a:
            o_node = m.get("odds", {}) or {}
            odds_data = {
                "home_win": o_node.get("homeWin"),
                "away_win": o_node.get("awayWin"),
                "draw": o_node.get("draw")
            }
            # Map both by ID and a straightforward string check
            if m_id:
                odds_lookup[m_id] = odds_data
            odds_lookup[f"{h}_vs_{a}"] = odds_data

    st.markdown("""
        <style>
        /* 1. Eliminate dead space at the top of the page */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            margin-top: 0px !important;
        }
        [data-testid="stHeader"] {
            height: 0px !important;
            background: transparent !important;
        }
        
        /* 2. Responsive Horizontal scrolling for columns on mobile */
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
        
        /* 3. Make Tabs prominent with colored backgrounds (especially on mobile) */
        button[data-baseweb="tab"] {
            background-color: #f1f5f9 !important; /* Light slate gray background for inactive tabs */
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px 8px 0px 0px !important;
            padding: 10px 16px !important;
            margin-right: 4px !important;
            transition: all 0.2s ease-in-out;
        }
        
        /* Style for the active/selected tab */
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #0284c7 !important; /* Vibrant primary blue for the active tab */
            color: white !important;
            font-weight: bold !important;
            border-color: #0284c7 !important;
        }

        @keyframes blinker {
            50% { opacity: 0; }
        }
        </style>
    """, unsafe_allow_html=True)
    
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
        st.title(dashboard_title)
        st.caption("Updated automatically from official game knockout data feeds.")
        
        tab_feed, tab_lead, tab_bracket = st.tabs([
            "📱 Live Action", 
            "🏆 Leaderboard",
            "🌳 Draw"
        ])
        
        with tab_lead:
            st.subheader("Overall Standings")
            df_lead = df_teams.copy()
            
            if not df_lead.empty:
                df_lead["Goals Scored"] = df_lead["Goals Scored"].fillna(0).astype(int)
                df_lead["Flag"] = df_lead["Flag"].fillna("https://flagcdn.com/w40/un.png")
                df_lead["Stage"] = df_lead["Stage"].fillna("Group Stage")
                
                df_display = df_lead[[
                    "Flag", "Player", "Stage", "Goals Scored"
                ]].rename(columns={
                    "Stage": "Current Progress Stage",
                    "Goals Scored": "Total Goals Scored"
                })
                
                df_display = df_display.sort_values(
                    by=["Current Progress Stage", "Total Goals Scored", "Player"], 
                    key=lambda col: col.apply(lambda val: STAGE_ORDER.index(val) if val in STAGE_ORDER else -1) if col.name == "Current Progress Stage" else col,
                    ascending=[False, False, True]
                ).reset_index(drop=True)

                st.dataframe(
                    df_display, 
                    use_container_width=True, 
                    height=450,
                    column_config={"Flag": st.column_config.ImageColumn("🏳️", width="small")},
                    hide_index=True
                )
            else:
                st.info("No standings data available yet.")
            
            st.divider()
            col_a, col_b = st.columns(2)
            
            def get_owner_formatted(team_name):
                player = team_to_player.get(str(team_name).strip().upper())
                if player:
                    return f"{player} ({team_name})"
                return ""

            with col_a:
                st.subheader("⚽ Golden Boot")
                if not df_boot.empty:
                    df_boot["Owner"] = df_boot["Team"].apply(get_owner_formatted)
                    st.dataframe(df_boot[["Scorer", "Team", "Owner", "Goals"]].head(10), use_container_width=True, hide_index=True)
                else:
                    st.write("Data currently unavailable.")
                    
            with col_b:
                st.subheader("🔥 Biggest Win")
                if not df_wins.empty:
                    df_wins["Owner"] = df_wins["Winner"].apply(get_owner_formatted)
                    st.dataframe(
                        df_wins[["Winner", "Match", "Score", "Owner"]].head(5), 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={"Owner": "Sweepstake Owner"}
                    )
                else:
                    st.write("Data currently unavailable.")

            # --- TOURNAMENT MILESTONES ---
            st.divider()
            st.subheader("🏆 Tournament Milestone Bounties")
            st.caption("Special sweepstake milestones achieved during the tournament (determined chronologically by non-simultaneous match order).")

            def render_milestone_card(title, team_name, metric_detail):
                team_key = str(team_name).strip().upper()
                entrant = team_to_player.get(team_key, "Unassigned")
                if entrant != "Unassigned":
                    entrant = f"{entrant} ({team_name})"
                
                flag_html = "🏳️" 
                
                if not df_teams.empty:
                    flag_match = df_teams[df_teams["Team"].str.upper() == team_key]
                    if not flag_match.empty and "Flag" in flag_match.columns:
                        flag_url = flag_match.iloc[0]["Flag"]
                        if flag_url:
                            flag_html = f"<img src='{flag_url}' width='26' style='vertical-align: middle; margin-right: 8px; border: 1px solid #e2e8f0; border-radius: 3px; flex-shrink: 0;'>"

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

            col_m1, col_m2 = st.columns(2)

            with col_m1:
                render_milestone_card("🎯 First Goal from a Penalty", "Switzerland", "Successfully converted against Qatar.")
                render_milestone_card("🤦‍♂️ First Own Goal", "Paraguay", "Deflected into their own net against USA.")

            with col_m2:
                render_milestone_card("🟥 First Red Card", "South Africa", "Sent off during the tournament opener sequence.")
                render_milestone_card("📉 Worst Team (Exited in Group Stage)", "Iraq", "Eliminated with 0 Points and a -11 Goal Difference.")

            st.divider()
            st.subheader("🚩 Most Corners")
            st.info("Corner kick statistics are not provided by the current tournament data provider (football-data.org).")

            st.space()
            st.caption("Created By Devansh Gupta using Gemini")
                
        with tab_bracket:
            st.subheader("Tournament Knockout Progression")
            st.caption("⬅️ Swipe horizontally to navigate the bracket stages ➡️")
            
            bracket_stages = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Finals"]
            grid_cols = st.columns(len(bracket_stages))
            
            stage_geometry = {
                "Round of 32":      {"start_pads": 0, "mid_pads": 0,  "total_slots": 16},
                "Round of 16":      {"start_pads": 1.05, "mid_pads": 2.48,  "total_slots": 8},
                "Quarter-Finals":   {"start_pads": 3.6, "mid_pads": 7.8,  "total_slots": 4},
                "Semi-Finals":      {"start_pads": 8.95, "mid_pads": 18.5, "total_slots": 2},
                "Finals":           {"start_pads": 20.0, "mid_pads": 0, "total_slots": 1}
            }
       
            def render_team_markup(team_name, current_stage_title, opponent_name=None):
                if team_name in ["TBD", "TBC"] or not team_name:
                    return "<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap;'><span style='color:gray; font-style:italic;'>🏳️ TBD</span></div>"
                
                match_row = df_teams[df_teams["Team"] == team_name]
                if match_row.empty:
                    return f"<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap;'><span>🏳️ {team_name}</span></div>"
                
                row = match_row.iloc[0]
                display_name = row.get("Player", team_name)
                
                flag_html = f"<img src='{row['Flag']}' width='20' style='vertical-align: middle; margin-right: 5px; flex-shrink: 0;'>"
                
                match_scores = row.get("Match Scores", {})
                score = match_scores.get(current_stage_title)
                
                is_live = current_stage_title in row.get("Live Stages", [])
                live_badge = "<span style='color: #2ecc71; font-size: 10px; font-weight: bold; margin-left: 5px; animation: blinker 1s linear infinite;'>● LIVE</span>" if is_live else ""
                
                trophy_suffix = " <span style='font-size:14px; margin-left:3px;'>🏆</span>" if (current_stage_title == "Finals" and row["Stage"] == "Champion") else ""
                
                score_html = ""
                if score is not None:
                    score_html = f"<span style='font-size: 18px; font-weight: bold; margin-left: 8px; flex-shrink: 0;'>{score}</span>"
                elif team_name != "TBD" and opponent_name and opponent_name != "TBD":
                    # Try lookups in both potential fixture configurations
                    match_odds = odds_lookup.get(f"{team_name}_vs_{opponent_name}") or odds_lookup.get(f"{opponent_name}_vs_{team_name}") or {}
                    
                    # Verify if team is playing at home or away in this fixture mapping
                    if odds_lookup.get(f"{team_name}_vs_{opponent_name}"):
                        team_odd = match_odds.get("home_win")
                    else:
                        team_odd = match_odds.get("away_win")
                    
                    if team_odd is not None and team_odd > 0:
                        score_html = f"<span style='font-size: 12px; color: #0284c7; font-weight: bold; margin-left: 8px;' title='API Pre-Match Decimal Odds'>{team_odd:.2f}</span>"
                    else:
                        score_html = f"<span style='font-size: 11px; color: #888; margin-left: 8px; font-style: italic;'>--</span>"


                if row["Status"] == "Knocked Out" and row["Stage"] == current_stage_title:
                    return f"<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap; width: 100%;'><div style='display: flex; align-items: center; min-width: 0; overflow: hidden; text-overflow: ellipsis;'>{flag_html}<span style='color:gray; text-decoration:line-through;'>{display_name}</span><span style='font-size:11px; color:red; margin-left:3px;'>❌</span></div>{score_html}</div>"
                else:
                    return f"<div style='display: flex; justify-content: space-between; align-items: center; white-space: nowrap; width: 100%;'><div style='display: flex; align-items: center; min-width: 0; overflow: hidden; text-overflow: ellipsis;'>{flag_html}<span style='font-weight: bold;'>{display_name}</span>{live_badge}{trophy_suffix}</div>{score_html}</div>"

            ordered_bracket = {}
            for stage in bracket_stages:
                slots = stage_geometry[stage]["total_slots"]
                ordered_bracket[stage] = [("TBD", "TBD") for _ in range(slots)]

            r32_raw = global_matchups.get("Round of 32", [])
            for idx, pair in enumerate(r32_raw):
                if idx < len(ordered_bracket["Round of 32"]):
                    ordered_bracket["Round of 32"][idx] = pair

            for stage_idx in range(1, len(bracket_stages)):
                current_stage = bracket_stages[stage_idx]
                prev_stage = bracket_stages[stage_idx - 1]
                raw_matches_this_stage = global_matchups.get(current_stage, [])

                for t_a, t_b in raw_matches_this_stage:
                    target_slot_idx = None
                    for prev_slot_idx, (p1, p2) in enumerate(ordered_bracket[prev_stage]):
                        if (t_a != "TBD" and t_a in [p1, p2]) or (t_b != "TBD" and t_b in [p1, p2]):
                            target_slot_idx = prev_slot_idx // 2
                            break
                    
                    if target_slot_idx is not None and target_slot_idx < len(ordered_bracket[current_stage]):
                        ordered_bracket[current_stage][target_slot_idx] = (t_a, t_b)
                    else:
                        for idx in range(len(ordered_bracket[current_stage])):
                            if ordered_bracket[current_stage][idx] == ("TBD", "TBD"):
                                ordered_bracket[current_stage][idx] = (t_a, t_b)
                                break

            if not df_teams.empty:
                for _, row in df_teams.iterrows():
                    team_name = row["Team"]
                    team_stage = row["Stage"]
                    if team_name in ["TBD", "TBC"]:
                        continue
             
                    if team_stage in ordered_bracket:
                        already_visible = False
                        for stg in bracket_stages:
                            if any((team_name == t1 or team_name == t2) for t1, t2 in ordered_bracket.get(stg, [])):
                                already_visible = True
                                break
                        
                        if not already_visible:
                            target_slot_idx = None
                            is_bottom_position = False
                            
                            try:
                                prev_stage_idx = bracket_stages.index(team_stage) - 1
                                if prev_stage_idx >= 0:
                                    prev_stage = bracket_stages[prev_stage_idx]
                                    for prev_slot_idx, (p1, p2) in enumerate(ordered_bracket[prev_stage]):
                                        if p1 == team_name or p2 == team_name:
                                            target_slot_idx = prev_slot_idx // 2
                                            is_bottom_position = (prev_slot_idx % 2 != 0)
                                            break
                            except ValueError:
                                pass
                            
                            if target_slot_idx is not None and target_slot_idx < len(ordered_bracket[team_stage]):
                                t1, t2 = ordered_bracket[team_stage][target_slot_idx]
                                if is_bottom_position:
                                    ordered_bracket[team_stage][target_slot_idx] = (t1, team_name)
                                else:
                                    ordered_bracket[team_stage][target_slot_idx] = (team_name, t2)
                            else:
                                for idx in range(len(ordered_bracket[team_stage])):
                                    t1, t2 = ordered_bracket[team_stage][idx]
                                    if t1 in ["TBD", "TBC"]:
                                        ordered_bracket[team_stage][idx] = (team_name, t2)
                                        break
                                    elif t2 in ["TBD", "TBC"]:
                                        ordered_bracket[team_stage][idx] = (t1, team_name)
                                        break

            for col_idx, stage_title in enumerate(bracket_stages):
                with grid_cols[col_idx]:
                    st.markdown(f"⚡ **{stage_title}**")
                    geom = stage_geometry[stage_title]
                    pairs = ordered_bracket.get(stage_title, [])
                
                    if geom["start_pads"] > 0:
                        height_px = geom["start_pads"] * 54
                        st.html(f"<div style='height: {height_px}px;'></div>")
                   
                    for slot_idx in range(geom["total_slots"]):
                        if slot_idx < len(pairs):
                            team_a, team_b = pairs[slot_idx]
                        else:
                            team_a, team_b = "TBD", "TBD"
                            
                        with st.container(border=True):
                            st.html(render_team_markup(team_a, stage_title, opponent_name=team_b))
                            st.html("<div style='margin: 3px 0; border-top: 1px dashed #ddd;'></div>")
                            st.html(render_team_markup(team_b, stage_title, opponent_name=team_a))
                            
                        if slot_idx < (geom["total_slots"] - 1) and geom["mid_pads"] > 0:
                            mid_height_px = geom["mid_pads"] * 54
                            st.html(f"<div style='height: {mid_height_px}px;'></div>")

            st.space()
            st.caption("Created By Devansh Gupta using Gemini")

        with tab_feed:
            now_utc = datetime.utcnow()
            now_bst = now_utc + timedelta(hours=1)
            
            if now_bst.hour >= 6:
                target_date = now_bst.date()
            else:
                target_date = now_bst.date() - timedelta(days=1)
            
            def get_night_matches(target_d):
                start_window = datetime.combine(target_d, datetime.min.time()).replace(hour=16)
                end_window = start_window + timedelta(hours=14)
                
                matches = []
                for m in raw_matches:
                    m_utc = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).replace(tzinfo=None)
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
                    h_player = df_teams[df_teams["Team"] == h_team]["Player"].values[0] if not df_teams.empty and not df_teams[df_teams["Team"] == h_team].empty else h_team
                    a_player = df_teams[df_teams["Team"] == a_team]["Player"].values[0] if not df_teams.empty and not df_teams[df_teams["Team"] == a_team].empty else a_team
                    
                    h_flag = df_teams[df_teams["Team"] == h_team]["Flag"].values[0] if not df_teams.empty and not df_teams[df_teams["Team"] == h_team].empty else "https://flagcdn.com/w40/un.png"
                    a_flag = df_teams[df_teams["Team"] == a_team]["Flag"].values[0] if not df_teams.empty and not df_teams[df_teams["Team"] == a_team].empty else "https://flagcdn.com/w40/un.png"
          
                    if m["status"] == "FINISHED":
                        score_str = f"{m['score']['fullTime']['home']} - {m['score']['fullTime']['away']}"
                        goals = m.get("goals", [])
                        goal_info = ", ".join([str(g['minute']) + "'" for g in goals]) if goals else "not provided"
                    
                        st.markdown(
                            f"<img src='{h_flag}' width='20' style='vertical-align: middle; margin-right: 6px;'>**{h_player}** &nbsp; `{score_str}` &nbsp; **{a_player}**<img src='{a_flag}' width='20' style='vertical-align: middle; margin-left: 6px;'>", 
                            unsafe_allow_html=True
                        )
                        
                        with st.spinner("✨ AI is analyzing last night's action..."):
                            summary = get_gemini_summary(m.get("id", 0), h_player, a_player, score_str, goal_info)
                        st.write(summary)

            st.divider()
            
            st.subheader("🕒 Tonight's Schedule")
            if not tonight_matches:
                 st.write("No fixtures carded for tonight.")
            else:
                for m, m_time in tonight_matches:
                    h_team = m["homeTeam"]["name"]
                    a_team = m["awayTeam"]["name"]
                    h_player = df_teams[df_teams["Team"] == h_team]["Player"].values[0] if not df_teams.empty and not df_teams[df_teams["Team"] == h_team].empty else h_team
                    a_player = df_teams[df_teams["Team"] == a_team]["Player"].values[0] if not df_teams.empty and not df_teams[df_teams["Team"] == a_team].empty else a_team
                    
                    h_flag = df_teams[df_teams["Team"] == h_team]["Flag"].values[0] if not df_teams.empty and not df_teams[df_teams["Team"] == h_team].empty else "https://flagcdn.com/w40/un.png"
                    a_flag = df_teams[df_teams["Team"] == a_team]["Flag"].values[0] if not df_teams.empty and not df_teams[df_teams["Team"] == a_team].empty else "https://flagcdn.com/w40/un.png"
                    
                    if m["status"] == "FINISHED":
                        score_str = f"{m['score']['fullTime']['home']} - {m['score']['fullTime']['away']}"
                        
                        st.markdown(
                            f"✅ <img src='{h_flag}' width='20' style='vertical-align: middle; margin-right: 6px;'>**{h_player}** &nbsp; `{score_str}` &nbsp; **{a_player}**<img src='{a_flag}' width='20' style='vertical-align: middle; margin-left: 6px;'>", 
                            unsafe_allow_html=True
                        )
                        
                        goals = m.get("goals", [])
                        goal_info = ", ".join([str(g['minute']) + "'" for g in goals]) if goals else "not provided"
                        
                        with st.spinner("✨ AI is wrapping up the match summary..."):
                            summary = get_gemini_summary(m.get("id", 0), h_player, a_player, score_str, goal_info)
                        if summary:
                            st.write(summary)
             
                    elif m["status"] in ["IN_PLAY", "PAUSED", "LIVE"]:
                        score_str = f"{m['score']['fullTime'].get('home', 0)} - {m['score']['fullTime'].get('away', 0)}"
                        
                        h_goals = df_teams[df_teams["Team"] == h_team]["Goals Scored"].sum() if not df_teams.empty else 0
                        a_goals = df_teams[df_teams["Team"] == a_team]["Goals Scored"].sum() if not df_teams.empty else 0
                        total_w = h_goals + a_goals + 2
    
                        h_prob = (h_goals + 1) / total_w
                        a_prob = (a_goals + 1) / total_w
                        
                        st.markdown(
                            f"🔴 **LIVE** | <img src='{h_flag}' width='20' style='vertical-align: middle; margin-right: 6px;'>**{h_player}** &nbsp; "
                            f"`{score_str}` &nbsp; **{a_player}**<img src='{a_flag}' width='20' style='vertical-align: middle; margin-left: 6px;'>", 
                            unsafe_allow_html=True
                        )
                        st.caption(f"📊 Live Prediction: {h_player} {h_prob:.0%} chance | {a_player} {a_prob:.0%} chance")
                        
                    else:
                        match_odds = odds_lookup.get(m.get("id")) or odds_lookup.get(f"{h_team}_vs_{a_team}") or {}
                        h_odd = match_odds.get("home_win")
                        a_odd = match_odds.get("away_win")
                        draw_odd = match_odds.get("draw")
                        
                        if h_odd and a_odd:
                            odds_display = f"Odds — Home: **{h_odd:.2f}** | Draw: **{draw_odd:.2f}** | Away: **{a_odd:.2f}**"
                        else:
                            odds_display = "Odds pending fixture finalization"
                        
                        st.markdown(
                            f"<img src='{h_flag}' width='20' style='vertical-align: middle; margin-right: 6px;'>**{h_player}** &nbsp; vs &nbsp; **{a_player}**<img src='{a_flag}' width='20' style='vertical-align: middle; margin-left: 6px;'>", 
                            unsafe_allow_html=True
                        )
                        st.markdown(f"<small style='color: #64748b; display: block; margin-top: -5px; margin-bottom: 5px;'>📊 {odds_display}</small>", unsafe_allow_html=True)

                        if m_time:
                            st.caption(f"Kickoff: {m_time.strftime('%H:%M')} (BST)")
                        
                        with st.spinner("🤖 AI is generating the match preview..."):
                            preview = get_gemini_preview(m.get("id", 0), h_player, a_player, h_prob, a_prob)
                        if preview:
                             st.write(preview)
            st.space()
            st.caption("Created By Devansh Gupta using Gemini")

    # -------------------------------------------------------------
    # ADMIN PANEL
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
            if st.sidebar.button("Terminated Session (Log Out)"):
                st.session_state["admin_authenticated"] = False
                st.rerun()
        
            adm_t1, adm_t2 = st.tabs(["👤 Participant Assignment Engine", "⚠️ Database Reset Switch"])
        
            with adm_t1:
                st.subheader(f"📝 Live Participant Registry Editor ({worksheet_name})")
                available_teams = sorted(df_teams["Team"].dropna().astype(str).tolist()) if not df_teams.empty else []
          
                if "Participant Name 2" not in df_participants.columns:
                    df_participants["Participant Name 2"] = ""

                edited_p_df = st.data_editor(
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
                            edited_p_df["Participant Name"] = edited_p_df["Participant Name"].astype(str).str.strip()
                            edited_p_df = edited_p_df.dropna(subset=["Participant Name"])
                            conn.update(worksheet=worksheet_name, data=edited_p_df)
                            st.cache_data.clear()
                            st.success("🎉 Participant registry updated successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to commit participant updates: {e}")
                        
            with adm_t2:
                st.subheader("Destructive Matrix Synchronization Block")
                st.warning(f"Clears user-registry rows completely from {worksheet_name}.")
                safety_checkbox = st.checkbox("I explicitly acknowledge that this operational process cannot be undone.")
                if st.button("Wipe & Clear Global Datastores", disabled=not safety_checkbox):
                    blank_p = pd.DataFrame(columns=["Participant Name", "Participant Name 2", "Team Assigned"])
                    try:
                        conn.update(worksheet=worksheet_name, data=blank_p)
                        st.cache_data.clear()
                        st.success("Google Spreadsheet cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Execution failed: {e}")


# -------------------------------------------------------------
# STREAMLIT PAGE ROUTING ENGINE
# -------------------------------------------------------------

def page_wimbledon():
    render_dashboard(
        worksheet_name="Participants_Wimbledon", 
        dashboard_title="📊 The Wimbledon World Cup"
    )

def page_office():
    render_dashboard(
        worksheet_name="Participants_NB", 
        dashboard_title="📊 The World Cup Sweepstake"
    )

# Use Streamlit's native navigation to create specific URL endpoints
pg = st.navigation([
    st.Page(page_wimbledon, title="Wimbledon Hub", icon="🎾", url_path="wimbledon"),
    st.Page(page_office, title="Office Hub", icon="🏢", url_path="office")
], position="hidden")

if __name__ == "__main__":
    pg.run()
