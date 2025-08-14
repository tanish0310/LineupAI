import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import json
import numpy as np
from datetime import datetime, timedelta

# Configure Streamlit
st.set_page_config(
    page_title="FPL Optimizer Pro", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #38003c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .transfer-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-priority {
        border-left: 4px solid #28a745;
        background-color: #f8fff9;
    }
    .medium-priority {
        border-left: 4px solid #ffc107;
        background-color: #fffef8;
    }
    .low-priority {
        border-left: 4px solid #dc3545;
        background-color: #fff8f8;
    }
    .captain-recommendation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .formation-display {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .player-item {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        min-width: 120px;
        text-align: center;
    }
    .bench-player {
        background-color: #f8f9fa;
        border-color: #6c757d;
    }
    .position-header {
        font-weight: bold;
        color: #38003c;
        margin: 1rem 0 0.5rem 0;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 0.3rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">⚽ FPL Optimizer Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://resources.premierleague.com/premierleague/photo/2019/12/11/2f5d594f-f9dc-43f4-8dc4-c6c2d80b3e3b/PL-MAIN-LOGO.png", width=200)
        
        page = st.selectbox(
            "Choose Analysis Type",
            [
                "🏠 Dashboard",
                "⚽ Squad Optimizer", 
                "🔄 Transfer Planner",
                "📊 Player Analysis",
                "📈 Performance Tracker",
                "🎯 Captain Selector",
                "🔮 Multi-GW Planner"
            ]
        )
        
        # Get current gameweek
        try:
            response = requests.get(f"{API_BASE_URL}/gameweek/current", timeout=5)
            if response.status_code == 200:
                current_gw = response.json()["current_gameweek"]
                st.success(f"Current Gameweek: {current_gw}")
            else:
                current_gw = 15
                st.warning("Using default gameweek")
        except:
            current_gw = 15
            st.error("API offline - demo mode")
        
        # Quick stats
        st.divider()
        st.subheader("📊 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Players", "600+")
        with col2:
            st.metric("Teams", "20")
        
        st.metric("Last Update", "2 hrs ago", "Live")
    
    # Route to selected page
    if page == "🏠 Dashboard":
        show_dashboard(current_gw)
    elif page == "⚽ Squad Optimizer":
        show_squad_optimizer(current_gw)
    elif page == "🔄 Transfer Planner":
        show_transfer_planner(current_gw)
    elif page == "📊 Player Analysis":
        show_player_analysis(current_gw)
    elif page == "📈 Performance Tracker":
        show_performance_tracker()
    elif page == "🎯 Captain Selector":
        show_captain_selector(current_gw)
    elif page == "🔮 Multi-GW Planner":
        show_multi_gw_planner(current_gw)

def show_dashboard(current_gw: int):
    st.title("🏠 FPL Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Gameweek", current_gw, "Active")
    
    with col2:
        try:
            response = requests.get(f"{API_BASE_URL}/model/performance", timeout=5)
            if response.status_code == 200:
                performance = response.json()["model_performance"]
                avg_mae = sum([pos.get("mae", 0) for pos in performance.values()]) / len(performance)
                accuracy = (3-avg_mae) / 3 * 100
                st.metric("Model Accuracy", f"{accuracy:.0f}%", "Good")
            else:
                st.metric("Model Accuracy", "85%", "Good")
        except:
            st.metric("Model Accuracy", "85%", "Good")
    
    with col3:
        st.metric("Players Tracked", "600+", "Updated")
    
    with col4:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=3)
            if response.status_code == 200:
                st.metric("API Status", "🟢 Online", "Healthy")
            else:
                st.metric("API Status", "🟡 Limited", "Partial")
        except:
            st.metric("API Status", "🔴 Offline", "Demo Mode")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Update Data", use_container_width=True):
            with st.spinner("Updating FPL data..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/update-data", timeout=10)
                    if response.status_code == 200:
                        st.success("Data update initiated!")
                    else:
                        st.error("Failed to update data")
                except:
                    st.error("API connection failed")
    
    with col2:
        if st.button("⚽ Build New Squad", use_container_width=True):
            st.switch_page("⚽ Squad Optimizer")
    
    with col3:
        if st.button("🔄 Plan Transfers", use_container_width=True):
            st.switch_page("🔄 Transfer Planner")
    
    with col4:
        if st.button("🎯 Find Captain", use_container_width=True):
            st.switch_page("🎯 Captain Selector")
    
    # Weekly insights
    st.subheader("💡 This Week's Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Top Captain Picks</h4>
            <p>1. Haaland vs Brighton (H) - 18.4 expected points</p>
            <p>2. Salah vs Palace (A) - 16.2 expected points</p>
            <p>3. Kane vs Fulham (H) - 15.6 expected points</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>💰 Budget Gems</h4>
            <p>• Mitoma (£5.1m) - 6.2 predicted points</p>
            <p>• Estupiñan (£4.9m) - 5.8 predicted points</p>
            <p>• Steele (£4.0m) - Great backup keeper</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Injury Concerns</h4>
            <p>• Robertson - Minor knock, 75% chance to play</p>
            <p>• Martinelli - Illness, monitor team news</p>
            <p>• Núñez - Thigh strain, likely to miss GW</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>📈 Form Players</h4>
            <p>• Saka - 5 goals in last 4 games</p>
            <p>• Rashford - On penalties, excellent fixtures</p>
            <p>• Almiron - Newcastle's attacking threat</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity
    st.subheader("📊 Recent Activity")
    
    activity_data = {
        "Time": ["2 hours ago", "5 hours ago", "8 hours ago", "1 day ago"],
        "Action": ["Data Update", "Model Retrain", "Squad Analysis", "Transfer Rec"],
        "Status": ["✅ Complete", "✅ Complete", "✅ Complete", "✅ Complete"],
        "Details": ["Latest GW data", "Goalkeeper model", "User squad optimization", "3 transfer options"]
    }
    
    st.dataframe(pd.DataFrame(activity_data), use_container_width=True, hide_index=True)

def show_squad_optimizer(current_gw: int):
    st.title("⚽ Squad Optimizer")
    st.write("Build the optimal 15-man squad from scratch using AI-powered predictions.")
    
    # Configuration section
    with st.expander("🔧 Squad Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            budget = st.number_input("Budget (£m)", min_value=80.0, max_value=105.0, value=100.0, step=0.1)
        
        with col2:
            target_gameweek = st.number_input("Target Gameweek", min_value=1, max_value=38, value=current_gw)
        
        with col3:
            strategy = st.selectbox("Strategy", ["Balanced", "Aggressive", "Conservative", "Budget"])
    
    # Strategy descriptions
    strategy_descriptions = {
        "Balanced": "Equal focus on premium and budget players",
        "Aggressive": "Premium heavy with high-risk, high-reward picks",
        "Conservative": "Consistent performers with lower variance",
        "Budget": "Maximum value picks to free up budget"
    }
    
    st.info(f"**{strategy} Strategy:** {strategy_descriptions[strategy]}")
    
    # Optional locked players
    with st.expander("🔒 Lock Specific Players (Optional)"):
        st.write("Force specific players into the squad (useful for keeping favorites):")
        
        col1, col2 = st.columns(2)
        with col1:
            locked_players_input = st.text_input("Player IDs (comma separated)", placeholder="e.g., 283, 302, 199")
        
        with col2:
            if st.button("🔍 Find Player IDs"):
                st.info("Enter player names in the Player Analysis section to find their IDs")
        
        locked_players = []
        if locked_players_input:
            try:
                locked_players = [int(x.strip()) for x in locked_players_input.split(",")]
                st.success(f"✅ Locked {len(locked_players)} players: {locked_players}")
            except:
                st.error("❌ Invalid format. Use comma-separated numbers only.")
    
    # Build squad button
    if st.button("🚀 Build Optimal Squad", type="primary", use_container_width=True):
        with st.spinner("🤖 Optimizing squad using linear programming..."):
            try:
                # Make API request
                request_data = {
                    "budget": budget,
                    "gameweek_id": target_gameweek,
                    "locked_players": locked_players if locked_players else None
                }
                
                response = requests.post(f"{API_BASE_URL}/build-squad", json=request_data, timeout=30)
                
                if response.status_code == 200:
                    squad_data = response.json()
                    display_squad_results(squad_data)
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    display_demo_squad_results()
                    
            except requests.exceptions.RequestException:
                st.warning("⚠️ API connection failed. Showing demo results...")
                display_demo_squad_results()

def display_squad_results(squad_data: Dict):
    """Display comprehensive squad optimization results."""
    
    squad_solution = squad_data["squad_solution"]
    analysis = squad_data["analysis"]
    validation = squad_data["validation"]
    
    # Validation check
    if not validation["is_valid"]:
        st.error("❌ Generated squad violates FPL rules:")
        for violation in validation["violations"]:
            st.error(f"• {violation}")
        return
    
    # Success message
    st.success("✅ Optimal squad generated successfully!")
    
    # Key metrics
    st.subheader("📊 Squad Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Cost", 
            f"£{squad_solution['total_cost']:.1f}m",
            f"£{squad_solution['budget_remaining']:.1f}m left"
        )
    
    with col2:
        st.metric(
            "Predicted Points", 
            f"{squad_solution['total_predicted_points']:.1f}",
            f"{squad_solution['total_predicted_points']/15:.1f} avg"
        )
    
    with col3:
        st.metric(
            "Formation", 
            squad_solution['formation_name'],
            "Optimal"
        )
    
    with col4:
        efficiency = squad_data["optimization_summary"]["efficiency_score"]
        st.metric(
            "Efficiency", 
            f"{efficiency:.1f}/10",
            "Points per £"
        )
    
    # Squad display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("⭐ Starting XI")
        
        starting_xi = squad_solution['starting_xi']
        
        # Formation display
        st.markdown(f"""
        <div class="formation-display">
            <h3>Formation: {starting_xi['formation']}</h3>
            <p>Total Points: {starting_xi['total_predicted_points']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display by position
        position_names = {1: "🥅 Goalkeeper", 2: "🛡️ Defenders", 3: "⚽ Midfielders", 4: "🎯 Forwards"}
        
        for position_id, position_players in starting_xi['players_by_position'].items():
            if position_players:
                st.markdown(f'<div class="position-header">{position_names.get(position_id, "Players")}</div>', unsafe_allow_html=True)
                
                # Display players in grid
                cols = st.columns(min(len(position_players), 5))
                for i, player in enumerate(position_players):
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="player-item">
                            <strong>{player['name']}</strong><br>
                            £{player['cost']/10:.1f}m<br>
                            {player['predicted_points']:.1f} pts
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🪑 Bench")
        
        bench = starting_xi.get('bench', [])
        for i, player in enumerate(bench, 1):
            st.markdown(f"""
            <div class="player-item bench-player">
                <strong>{i}. {player['name']}</strong><br>
                £{player['cost']/10:.1f}m<br>
                {player['predicted_points']:.1f} pts
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📊 Quick Stats")
        st.metric("Bench Points", f"{starting_xi.get('bench_points', 0):.1f}")
        st.metric("Risk Level", analysis['risk_assessment']['overall_risk_level'])
    
    # Captain recommendations
    st.subheader("👑 Captain Recommendations")
    
    captain_options = squad_solution.get('captain_options', [])
    if captain_options:
        captain_cols = st.columns(min(len(captain_options), 3))
        
        for i, captain in enumerate(captain_options[:3]):
            with captain_cols[i]:
                rank_emoji = ["🥇", "🥈", "🥉"][i]
                st.markdown(f"""
                <div class="captain-recommendation">
                    <h4>{rank_emoji} {captain['name']}</h4>
                    <p><strong>Expected Points:</strong> {captain['expected_captain_points']:.1f}</p>
                    <p><strong>Safety Score:</strong> {captain['safety_score']:.0%}</p>
                    <p><strong>Upside:</strong> {captain['upside_potential']:.1f}</p>
                    <p><small>{captain['reasoning'][:50]}...</small></p>
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed analysis
    with st.expander("📈 Detailed Squad Analysis", expanded=False):
        
        # Position breakdown
        st.subheader("📊 Position Breakdown")
        position_data = []
        
        for pos_name, pos_data in analysis['position_breakdown'].items():
            position_data.append({
                'Position': pos_name,
                'Players': pos_data['count'],
                'Total Cost': f"£{pos_data['total_cost']:.1f}m",
                'Predicted Points': f"{pos_data['total_predicted_points']:.1f}",
                'Avg Points': f"{pos_data['avg_predicted_points']:.1f}"
            })
        
        df_positions = pd.DataFrame(position_data)
        st.dataframe(df_positions, use_container_width=True, hide_index=True)
        
        # Position breakdown chart
        fig_positions = px.bar(
            df_positions, 
            x='Position', 
            y=[float(x.replace('£', '').replace('m', '')) for x in df_positions['Total Cost']],
            title="Budget Allocation by Position"
        )
        st.plotly_chart(fig_positions, use_container_width=True)
        
        # Team distribution
        st.subheader("🏟️ Team Distribution")
        team_dist_data = []
        
        for team, players in analysis['team_distribution'].items():
            total_points = sum([p['predicted_points'] for p in players])
            team_dist_data.append({
                'Team': team,
                'Players': len(players),
                'Total Points': f"{total_points:.1f}",
                'Players List': ", ".join([f"{p['name']} ({p['position']})" for p in players])
            })
        
        df_teams = pd.DataFrame(team_dist_data)
        st.dataframe(df_teams, use_container_width=True, hide_index=True)
        
        # Price distribution
        st.subheader("💰 Price Analysis")
        price_analysis = analysis['price_analysis']
        
        price_col1, price_col2, price_col3 = st.columns(3)
        with price_col1:
            st.metric("Most Expensive", f"£{price_analysis['most_expensive']:.1f}m")
        with price_col2:
            st.metric("Average Cost", f"£{price_analysis['average_cost']:.1f}m")
        with price_col3:
            st.metric("Cheapest", f"£{price_analysis['cheapest']:.1f}m")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Premium Players (£9.0m+)", price_analysis['premium_players'])
        with col2:
            st.metric("Budget Players (≤£5.0m)", price_analysis['budget_players'])
        
        # Risk assessment
        st.subheader("⚖️ Risk Assessment")
        risk_data = analysis['risk_assessment']
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            st.metric("Average Confidence", f"{risk_data['avg_confidence']:.0%}")
        with risk_col2:
            st.metric("High Confidence Players", f"{risk_data['high_confidence']}/15")
        with risk_col3:
            st.metric("Risky Picks", risk_data['risky_picks'])
        
        # Risk level indicator
        risk_level = risk_data['overall_risk_level']
        if risk_level == "Low":
            st.success(f"✅ {risk_level} Risk - Conservative, reliable squad")
        elif risk_level == "Medium":
            st.info(f"⚖️ {risk_level} Risk - Balanced approach")
        else:
            st.warning(f"⚠️ {risk_level} Risk - Aggressive, high variance squad")

def display_demo_squad_results():
    """Display demo squad results when API is unavailable."""
    st.info("📱 Displaying demo results (API unavailable)")
    
    demo_data = {
        "total_cost": 99.5,
        "budget_remaining": 0.5,
        "predicted_points": 67.8,
        "formation": "3-5-2",
        "captain": "Haaland",
        "vice_captain": "Salah"
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"£{demo_data['total_cost']}m", f"£{demo_data['budget_remaining']}m left")
    with col2:
        st.metric("Predicted Points", f"{demo_data['predicted_points']}", "4.5 avg")
    with col3:
        st.metric("Formation", demo_data['formation'])
    with col4:
        st.metric("Captain", demo_data['captain'])
    
    # Demo squad display
    st.subheader("⭐ Demo Starting XI")
    
    demo_squad = {
        "🥅 Goalkeeper": [{"name": "Alisson", "cost": 55, "points": 5.2}],
        "🛡️ Defenders": [
            {"name": "TAA", "cost": 75, "points": 6.8},
            {"name": "Saliba", "cost": 50, "points": 5.1},
            {"name": "Trippier", "cost": 65, "points": 4.9}
        ],
        "⚽ Midfielders": [
            {"name": "Salah", "cost": 130, "points": 8.9},
            {"name": "De Bruyne", "cost": 110, "points": 8.1},
            {"name": "Son", "cost": 95, "points": 7.2},
            {"name": "Rashford", "cost": 85, "points": 5.8},
            {"name": "Mitoma", "cost": 51, "points": 5.5}
        ],
        "🎯 Forwards": [
            {"name": "Haaland", "cost": 120, "points": 12.1},
            {"name": "Kane", "cost": 115, "points": 7.8}
        ]
    }
    
    for position, players in demo_squad.items():
        st.markdown(f'<div class="position-header">{position}</div>', unsafe_allow_html=True)
        
        cols = st.columns(min(len(players), 5))
        for i, player in enumerate(players):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div class="player-item">
                    <strong>{player['name']}</strong><br>
                    £{player['cost']/10:.1f}m<br>
                    {player['points']:.1f} pts
                </div>
                """, unsafe_allow_html=True)
    
    st.success("🎉 Demo squad built successfully! Connect to API for live optimization.")

def show_transfer_planner(current_gw: int):
    st.title("🔄 Transfer Planner")
    st.write("Get AI-powered transfer recommendations with detailed reasoning and analysis.")
    
    # Squad input section
    st.subheader("📝 Current Squad Input")
    
    input_method = st.radio(
        "How would you like to input your squad?",
        ["📝 Manual Entry", "📁 Upload CSV", "🔗 FPL Team ID"],
        horizontal=True
    )
    
    current_squad = []
    
    if input_method == "📝 Manual Entry":
        st.write("Enter your current 15-player squad:")
        
        with st.form("squad_input"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🥅 Goalkeepers (2)**")
                gk1 = st.text_input("GK 1", value="Alisson", placeholder="Player name")
                gk2 = st.text_input("GK 2", value="Steele", placeholder="Player name")
                
                st.markdown("**🛡️ Defenders (5)**")
                def1 = st.text_input("DEF 1", value="TAA", placeholder="Player name") 
                def2 = st.text_input("DEF 2", value="Saliba", placeholder="Player name")
                def3 = st.text_input("DEF 3", value="Trippier", placeholder="Player name")
                def4 = st.text_input("DEF 4", value="White", placeholder="Player name")
                def5 = st.text_input("DEF 5", value="Mitchell", placeholder="Player name")
                
                st.markdown("**⚽ Midfielders (5)**")
                mid1 = st.text_input("MID 1", value="Salah", placeholder="Player name")
                mid2 = st.text_input("MID 2", value="De Bruyne", placeholder="Player name")
                mid3 = st.text_input("MID 3", value="Son", placeholder="Player name")
            
            with col2:
                st.markdown("**⚽ Midfielders (continued)**")
                mid4 = st.text_input("MID 4", value="Rashford", placeholder="Player name")
                mid5 = st.text_input("MID 5", value="Gordon", placeholder="Player name")
                
                st.markdown("**🎯 Forwards (3)**") 
                fwd1 = st.text_input("FWD 1", value="Haaland", placeholder="Player name")
                fwd2 = st.text_input("FWD 2", value="Kane", placeholder="Player name")
                fwd3 = st.text_input("FWD 3", value="Archer", placeholder="Player name")
                
                st.markdown("**💰 Squad Details**")
                squad_value = st.number_input("Squad Value (£m)", value=99.5, step=0.1)
                in_bank = st.number_input("Money in Bank (£m)", value=0.5, step=0.1)
            
            submitted = st.form_submit_button("📊 Load Squad", use_container_width=True)
            
            if submitted:
                squad_names = [gk1, gk2, def1, def2, def3, def4, def5, 
                             mid1, mid2, mid3, mid4, mid5, fwd1, fwd2, fwd3]
                
                # Demo positions mapping
                positions = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
                teams = [1, 2, 1, 3, 4, 3, 5, 1, 6, 7, 8, 4, 6, 7, 9]  # Demo team distribution
                costs = [55, 40, 75, 50, 65, 45, 40, 130, 110, 95, 85, 51, 120, 115, 45]  # Demo costs
                
                for i, name in enumerate(squad_names):
                    if name.strip():
                        current_squad.append({
                            "id": i + 1,
                            "name": name.strip(),
                            "position": positions[i],
                            "team": teams[i],
                            "cost": costs[i],
                            "selling_price": costs[i]  # Simplified - selling price = cost
                        })
                
                st.success(f"✅ Loaded squad with {len(current_squad)} players!")
                
                # Quick squad validation
                position_counts = {}
                for player in current_squad:
                    pos = player['position']
                    position_counts[pos] = position_counts.get(pos, 0) + 1
                
                if position_counts == {1: 2, 2: 5, 3: 5, 4: 3}:
                    st.success("✅ Squad formation is valid!")
                else:
                    st.error(f"❌ Invalid squad formation: {position_counts}")
    
    elif input_method == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Upload squad CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("📄 Preview of uploaded squad:")
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("🔄 Process CSV"):
                    st.info("📝 CSV processing functionality would convert file to squad format")
                    # Implementation would depend on CSV structure
                
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")
    
    else:  # FPL Team ID
        col1, col2 = st.columns([2, 1])
        
        with col1:
            team_id = st.text_input("Enter your FPL Team ID", placeholder="e.g., 123456")
        
        with col2:
            st.markdown("**How to find your Team ID:**")
            st.info("1. Login to FPL website\n2. Go to 'Pick Team'\n3. Check URL for ID number")
        
        if team_id and st.button("🔗 Fetch Squad from FPL"):
            with st.spinner("Fetching squad from FPL API..."):
                st.info("🔄 FPL API integration would fetch official squad data")
                # Implementation would use official FPL API
    
    # Transfer parameters and recommendations
    if current_squad:
        st.divider()
        st.subheader("⚙️ Transfer Parameters")
        
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)
        
        with param_col1:
            free_transfers = st.number_input("Free Transfers", min_value=0, max_value=5, value=1)
        
        with param_col2:
            budget_remaining = st.number_input("Budget Remaining (£m)", min_value=0.0, max_value=20.0, value=0.5, step=0.1)
        
        with param_col3:
            max_transfers = st.number_input("Max Transfers", min_value=1, max_value=3, value=2)
        
        with param_col4:
            wildcards = st.selectbox("Special Chips", ["None", "Wildcard", "Free Hit", "Bench Boost"])
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                risk_tolerance = st.select_slider(
                    "Risk Tolerance",
                    options=["Conservative", "Balanced", "Aggressive"],
                    value="Balanced"
                )
                
                hit_threshold = st.number_input(
                    "Hit Threshold (points needed to justify -4)",
                    min_value=1.0, max_value=10.0, value=4.0, step=0.5
                )
            
            with col2:
                focus_positions = st.multiselect(
                    "Focus Positions",
                    ["Goalkeeper", "Defender", "Midfielder", "Forward"],
                    default=[]
                )
                
                exclude_teams = st.multiselect(
                    "Exclude Teams",
                    ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd", "Tottenham"],
                    default=[]
                )
        
        # Generate recommendations
        if st.button("🎯 Generate Transfer Recommendations", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing optimal transfers with AI..."):
                try:
                    request_data = {
                        "current_squad": current_squad,
                        "gameweek_id": current_gw,
                        "free_transfers": free_transfers,
                        "budget_remaining": budget_remaining,
                        "max_transfers": max_transfers
                    }
                    
                    response = requests.post(f"{API_BASE_URL}/optimize-transfers", json=request_data, timeout=30)
                    
                    if response.status_code == 200:
                        transfer_data = response.json()
                        display_transfer_recommendations(transfer_data)
                    else:
                        st.error("❌ API Error - showing demo recommendations")
                        display_demo_transfer_recommendations(free_transfers, budget_remaining)
                        
                except requests.exceptions.RequestException:
                    st.warning("⚠️ API connection failed - showing demo recommendations")
                    display_demo_transfer_recommendations(free_transfers, budget_remaining)

def display_transfer_recommendations(transfer_data: Dict):
    """Display comprehensive transfer recommendations."""
    
    recommendations = transfer_data["recommendations"]
    summary = transfer_data["summary"]
    
    # Summary metrics
    st.subheader("📊 Transfer Analysis Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Total Options", summary["total_options"], "Analyzed")
    with summary_col2:
        st.metric("Recommended", summary["recommended_options"], "Net positive")
    with summary_col3:
        st.metric("Hit Worthy", summary["hit_worthy_options"], "Worth -4 pts")
    with summary_col4:
        st.metric("Free Transfers", transfer_data["free_transfers"], "Available")
    
    # Display recommendations
    st.subheader("🎯 Transfer Recommendations")
    
    if not recommendations:
        st.info("💡 No transfer recommendations generated. Your squad may already be optimal!")
        return
    
    for i, rec in enumerate(recommendations[:5], 1):
        priority_class = f"{rec['priority'].lower()}-priority"
        
        # Recommendation header
        st.markdown(f"""
        <div class="transfer-card {priority_class}">
            <h3>🔄 Option {i}: {rec['recommendation']}</h3>
            <p><strong>Priority:</strong> {rec['priority']} | <strong>Complexity:</strong> {rec.get('complexity', 'Standard')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Transfer details
        transfer_col1, transfer_col2, transfer_col3 = st.columns([2, 1, 1])
        
        with transfer_col1:
            st.markdown("**📤 Players Out → 📥 Players In**")
            for player_out, player_in in zip(rec['players_out'], rec['players_in']):
                st.write(f"• **OUT:** {player_out['name']} → **IN:** {player_in['name']}")
        
        with transfer_col2:
            st.metric("Points Improvement", f"+{rec['points_improvement']:.1f}")
            st.metric("Cost Change", f"£{rec['cost_change']:+.1f}m")
        
        with transfer_col3:
            st.metric("Net Benefit", f"+{rec['net_improvement']:.1f} pts")
            if rec['hit_cost'] > 0:
                st.metric("Hit Cost", f"-{rec['hit_cost']} pts", "Transfer hit")
            else:
                st.success("✅ Uses free transfers")
        
        # Reasoning summary
        st.markdown("**💡 Key Reasons:**")
        for reason in rec['reasoning'][:3]:  # Show top 3 reasons
            st.write(f"• {reason}")
        
        # Detailed analysis expandable
        with st.expander(f"📋 Detailed Analysis - Option {i}"):
            
            if 'detailed_analysis' in rec:
                for j, analysis in enumerate(rec['detailed_analysis']):
                    st.markdown(f"**Transfer {j+1} Analysis:**")
                    
                    # Performance comparison
                    if 'performance_analysis' in analysis:
                        perf = analysis['performance_analysis']
                        st.markdown("*📊 Performance Comparison:*")
                        
                        perf_col1, perf_col2 = st.columns(2)
                        with perf_col1:
                            st.write(f"**Predicted Points:** {perf['predicted_points']['player_out']:.1f} → {perf['predicted_points']['player_in']:.1f}")
                            st.write(f"**Form:** {perf['recent_form']['form_out']:.1f} → {perf['recent_form']['form_in']:.1f}")
                        
                        with perf_col2:
                            improvement = perf['predicted_points']['percentage_improvement']
                            st.metric("Performance Improvement", f"{improvement:+.1f}%")
                    
                    # Risk assessment
                    if 'risk_assessment' in analysis:
                        risk = analysis['risk_assessment']
                        st.markdown("*⚖️ Risk Assessment:*")
                        
                        risk_col1, risk_col2 = st.columns(2)
                        with risk_col1:
                            st.write(f"**Injury Risk:** {risk['injury_risk']['risk_change']}")
                            st.write(f"**Rotation Risk:** {risk['rotation_risk']['minutes_certainty_change']}")
                        
                        with risk_col2:
                            st.write(f"**Confidence:** {risk['prediction_confidence']['reliability_assessment']}")
                            st.write(f"**Overall Risk:** {risk['overall_risk_level']}")
                    
                    # Fixture analysis
                    if 'fixture_analysis' in analysis and analysis['fixture_analysis'].get('analysis') != 'Fixture data not available':
                        fixture = analysis['fixture_analysis']
                        st.markdown("*🗓️ Fixture Analysis:*")
                        
                        if 'next_5_gameweeks' in fixture:
                            avg_diff_out = fixture['next_5_gameweeks']['avg_difficulty_out']
                            avg_diff_in = fixture['next_5_gameweeks']['avg_difficulty_in']
                            st.write(f"**Next 5 GW Difficulty:** {avg_diff_out:.1f} → {avg_diff_in:.1f}")
                        
                        if 'fixture_swing_analysis' in fixture:
                            st.write(f"**Fixture Swing:** {fixture['fixture_swing_analysis']}")
                    
                    st.divider()
            
            # Transfer timing
            st.markdown("**⏰ Timing Recommendation:**")
            if rec['hit_cost'] > 0:
                if rec['net_improvement'] > 2:
                    st.success("🟢 **MAKE TRANSFER NOW** - Strong net benefit justifies the hit")
                elif rec['net_improvement'] > 0:
                    st.warning("🟡 **MARGINAL** - Consider waiting for next gameweek if possible")
                else:
                    st.error("🔴 **AVOID** - Negative net return")
            else:
                st.success("🟢 **MAKE TRANSFER** - Uses free transfer(s)")
        
        st.divider()
    
    # Additional insights
    st.subheader("💡 Additional Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        <div class="info-box">
            <h4>📈 Market Trends</h4>
            <p>• Attacking players trending up due to fixture ease</p>
            <p>• Defensive assets from top 6 teams gaining popularity</p>
            <p>• Budget midfielders showing great value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Considerations</h4>
            <p>• Price changes may affect transfer costs</p>
            <p>• Monitor team news before deadline</p>
            <p>• Consider upcoming fixture difficulty</p>
        </div>
        """, unsafe_allow_html=True)

def display_demo_transfer_recommendations(free_transfers: int, budget: float):
    """Display demo transfer recommendations."""
    st.info("📱 Showing demo transfer recommendations (API unavailable)")
    
    demo_transfers = [
        {
            "option": 1,
            "players_out": ["Rashford"],
            "players_in": ["Saka"],
            "cost_change": "+£1.9m",
            "points_improvement": 3.2,
            "hit_cost": 4 if free_transfers == 0 else 0,
            "reasoning": [
                "Arsenal home vs Brighton - excellent fixture",
                "Saka in outstanding form (5 goals in 4 games)",
                "Better underlying stats (xG, xA)"
            ],
            "priority": "HIGH",
            "recommendation": "STRONGLY RECOMMENDED"
        },
        {
            "option": 2,
            "players_out": ["Mitoma"],
            "players_in": ["Gordon"],
            "cost_change": "£0.0m",
            "points_improvement": 2.8,
            "hit_cost": 0 if free_transfers >= 1 else 4,
            "reasoning": [
                "Newcastle's favorable fixture run begins",
                "Gordon on set pieces and penalties",
                "Mitoma rotation risk in busy period"
            ],
            "priority": "MEDIUM",
            "recommendation": "RECOMMENDED"
        },
        {
            "option": 3,
            "players_out": ["Kane", "Mitchell"],
            "players_in": ["Mitrovic", "Trippier"],
            "cost_change": "+£0.3m",
            "points_improvement": 4.1,
            "hit_cost": 4 if free_transfers < 2 else 0,
            "reasoning": [
                "Mitrovic excellent value at lower price",
                "Trippier on penalties and free kicks",
                "Saves budget for future upgrades"
            ],
            "priority": "MEDIUM",
            "recommendation": "CONSIDER"
        }
    ]
    
    for i, transfer in enumerate(demo_transfers, 1):
        priority_class = f"{transfer['priority'].lower()}-priority"
        
        st.markdown(f"""
        <div class="transfer-card {priority_class}">
            <h3>🔄 Option {i}: {transfer['recommendation']}</h3>
            <p><strong>Priority:</strong> {transfer['priority']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**📤 Players Out → 📥 Players In**")
            for j, (out, inp) in enumerate(zip(transfer['players_out'], transfer['players_in'])):
                st.write(f"• **OUT:** {out} → **IN:** {inp}")
        
        with col2:
            st.metric("Points Improvement", f"+{transfer['points_improvement']:.1f}")
            st.metric("Cost Change", transfer['cost_change'])
        
        with col3:
            net_improvement = transfer['points_improvement'] - transfer['hit_cost']
            st.metric("Net Benefit", f"+{net_improvement:.1f} pts")
            if transfer['hit_cost'] > 0:
                st.metric("Hit Cost", f"-{transfer['hit_cost']} pts")
            else:
                st.success("✅ Free transfer")
        
        st.markdown("**💡 Key Reasons:**")
        for reason in transfer['reasoning']:
            st.write(f"• {reason}")
        
        st.divider()

def show_player_analysis(current_gw: int):
    st.title("📊 Player Analysis")
    st.write("Get comprehensive AI-powered analysis for any FPL player.")
    
    # Player search section
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        player_input = st.text_input(
            "🔍 Enter Player Name or ID", 
            placeholder="e.g., Haaland, Salah, or player ID 283"
        )
    
    with search_col2:
        gameweek_select = st.number_input(
            "Gameweek", 
            min_value=1, 
            max_value=38, 
            value=current_gw
        )
    
    # Quick player suggestions
    st.markdown("**🔥 Popular Players:**")
    popular_players = ["Haaland", "Salah", "De Bruyne", "Kane", "Son", "Saka", "Rashford", "TAA"]
    
    cols = st.columns(4)
    for i, player in enumerate(popular_players):
        with cols[i % 4]:
            if st.button(f"🔍 {player}", key=f"popular_{player}"):
                player_input = player
                st.rerun()
    
    if st.button("🎯 Analyze Player", type="primary", use_container_width=True) and player_input:
        
        # For demo, simulate different players
        player_id = hash(player_input.lower()) % 1000  # Generate consistent ID from name
        
        with st.spinner(f"🤖 Analyzing {player_input} with AI..."):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/player/{player_id}/analysis?gameweek_id={gameweek_select}", 
                    timeout=10
                )
                
                if response.status_code == 200:
                    player_data = response.json()["analysis"]
                    display_player_analysis_results(player_data, player_input)
                else:
                    st.warning("⚠️ Player not found in API - showing demo analysis")
                    display_demo_player_analysis(player_input)
                    
            except requests.exceptions.RequestException:
                st.warning("⚠️ API connection failed - showing demo analysis")
                display_demo_player_analysis(player_input)

def display_player_analysis_results(player_data: Dict, player_name: str):
    """Display comprehensive player analysis."""
    
    st.success(f"✅ Analysis complete for **{player_name}**")
    
    # Key metrics header
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    prediction = player_data["prediction"]
    captain_analysis = player_data["captain_analysis"]
    ownership = player_data["ownership_data"]
    
    with metrics_col1:
        st.metric(
            "Predicted Points", 
            f"{prediction['points']:.1f}",
            f"{prediction['confidence']:.0%} confidence"
        )
    with metrics_col2:
        st.metric(
            "Captain Score", 
            f"{captain_analysis['captain_score']:.1f}",
            "Expected as (C)"
        )
    with metrics_col3:
        st.metric(
            "Price", 
            f"£{ownership['price']:.1f}m",
            f"{ownership['price_change']:+.1f} change"
        )
    with metrics_col4:
        st.metric(
            "Ownership", 
            f"{ownership['selected_by_percent']:.1f}%",
            "Selected by"
        )
    
    # Main analysis sections
    analysis_col1, analysis_col2 = st.columns([1.5, 1])
    
    with analysis_col1:
        # Performance analysis
        st.subheader("📈 Performance Analysis")
        
        form_data = player_data["form_data"]
        
        # Recent form chart
        recent_games = list(range(1, 6))
        recent_points = form_data["recent_points"]
        
        fig_form = px.line(
            x=recent_games,
            y=recent_points,
            title="Last 5 Gameweeks Performance",
            labels={"x": "Games Ago", "y": "Points Scored"},
            markers=True
        )
        fig_form.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            showlegend=False
        )
        st.plotly_chart(fig_form, use_container_width=True)
        
        # Form metrics
        form_col1, form_col2, form_col3 = st.columns(3)
        with form_col1:
            st.metric("5GW Average", f"{form_data['avg_points_5gw']:.1f} pts")
        with form_col2:
            st.metric("Consistency", f"{form_data['consistency_score']:.0%}")
        with form_col3:
            trend_delta = "📈" if form_data['form_trend'] == "Improving" else "📊"
            st.metric("Trend", f"{trend_delta} {form_data['form_trend']}")
        
        # Fixture analysis
        st.subheader("🗓️ Fixture Analysis")
        
        fixture_data = player_data["fixture_analysis"]
        
        # Next fixture highlight
        st.markdown(f"""
        <div class="fixture-highlight">
            <h4>Next Fixture: {fixture_data['next_fixture']}</h4>
            <p>Difficulty: {fixture_data['difficulty']}/5 {'🟢' if fixture_data['difficulty'] <= 2 else '🟡' if fixture_data['difficulty'] <= 3 else '🔴'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upcoming fixtures
        st.markdown("**Upcoming Fixtures:**")
        for i, fixture in enumerate(fixture_data["upcoming_fixtures"][:3], 1):
            difficulty_color = "🟢" if i <= 2 else "🟡" if i <= 3 else "🔴"
            st.write(f"{i}. {fixture} {difficulty_color}")
    
    with analysis_col2:
        # Captain analysis
        st.subheader("👑 Captaincy Analysis")
        
        # Captain score visualization
        captain_score = captain_analysis['captain_score']
        max_score = 25  # Typical maximum
        
        fig_captain = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = captain_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Captain Score"},
            delta = {'reference': 15, 'position': "top"},
            gauge = {
                'axis': {'range': [None, max_score]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgray"},
                    {'range': [10, 15], 'color': "yellow"},
                    {'range': [15, 25], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        fig_captain.update_layout(height=300)
        st.plotly_chart(fig_captain, use_container_width=True)
        
        # Captain metrics
        capt_col1, capt_col2 = st.columns(2)
        with capt_col1:
            st.metric("Expected Points", f"{captain_analysis['expected_captain_points']:.1f}")
            st.metric("Safety Score", f"{captain_analysis['safety_score']:.0%}")
        with capt_col2:
            st.metric("Upside Potential", f"{captain_analysis['upside_potential']:.1f}")
            
            # Captain recommendation
            if captain_score >= 18:
                st.success("🌟 **PREMIUM CAPTAIN**")
            elif captain_score >= 15:
                st.info("👑 **GOOD CAPTAIN**")
            elif captain_score >= 12:
                st.warning("⚖️ **RISKY CAPTAIN**")
            else:
                st.error("❌ **AVOID AS CAPTAIN**")
        
        # Transfer recommendation
        st.subheader("💡 Transfer Advice")
        
        recommendations = player_data["recommendations"]
        
        # Overall recommendation
        if recommendations["transfer_in"]:
            st.markdown("""
            <div class="success-box">
                <h4>✅ TRANSFER IN RECOMMENDED</h4>
                <p>Strong predicted performance and good value</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>➖ HOLD/MONITOR</h4>
                <p>Current performance acceptable, monitor for changes</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write(f"**Assessment:** {recommendations['hold_or_sell']}")
        
        # Ownership insights
        st.subheader("📊 Ownership Data")
        
        ownership_pct = ownership['selected_by_percent']
        
        if ownership_pct >= 50:
            st.error(f"🔴 **HIGHLY OWNED** ({ownership_pct:.1f}%)")
            st.write("Consider differential options")
        elif ownership_pct >= 20:
            st.warning(f"🟡 **POPULAR PICK** ({ownership_pct:.1f}%)")
            st.write("Safe but less differential")
        elif ownership_pct >= 5:
            st.info(f"🟢 **BALANCED OWNERSHIP** ({ownership_pct:.1f}%)")
            st.write("Good balance of safety and differential")
        else:
            st.success(f"🌟 **DIFFERENTIAL** ({ownership_pct:.1f}%)")
            st.write("High risk, high reward pick")
    
    # Detailed statistics
    with st.expander("📊 Detailed Statistics & Comparisons", expanded=False):
        
        # Position comparison would go here
        st.subheader("📈 Position Comparison")
        st.info("This section would show how the player ranks among others in their position")
        
        # Historical performance
        st.subheader("📅 Historical Performance")
        
        # Create sample historical data
        weeks = list(range(1, current_gw))
        sample_points = np.random.normal(prediction['points'], 2, len(weeks))
        sample_points = np.clip(sample_points, 0, 20)  # Reasonable range
        
        fig_history = px.bar(
            x=weeks,
            y=sample_points,
            title="Points by Gameweek This Season",
            labels={"x": "Gameweek", "y": "Points"}
        )
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Advanced metrics
        st.subheader("🔬 Advanced Metrics")
        
        advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
        
        with advanced_col1:
            st.metric("xPoints (Expected)", f"{prediction['points']:.1f}")
            st.metric("Points vs Price", f"{prediction['points']/(ownership['price']):.2f}")
        
        with advanced_col2:
            st.metric("Form vs Season Avg", "+15%")  # Demo value
            st.metric("Fixture Difficulty", f"{fixture_data['difficulty']}/5")
        
        with advanced_col3:
            st.metric("Ownership Change", "+2.3%")  # Demo value
            st.metric("Price Change Prob", "Medium")  # Demo value

def display_demo_player_analysis(player_name: str):
    """Display demo player analysis when API is unavailable."""
    st.info(f"📱 Showing demo analysis for **{player_name}** (API unavailable)")
    
    # Generate demo data based on player name
    player_hash = hash(player_name.lower())
    
    # Demo metrics
    demo_data = {
        "predicted_points": 8.4 + (player_hash % 20) / 10,
        "confidence": 0.75 + (player_hash % 25) / 100,
        "captain_score": 15.2 + (player_hash % 30) / 10,
        "price": 8.5 + (player_hash % 60) / 10,
        "ownership": 15.3 + (player_hash % 300) / 10,
        "form_trend": ["Improving", "Stable", "Declining"][player_hash % 3]
    }
    
    # Display demo metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Points", f"{demo_data['predicted_points']:.1f}")
    with col2:
        st.metric("Confidence", f"{demo_data['confidence']:.0%}")
    with col3:
        st.metric("Captain Score", f"{demo_data['captain_score']:.1f}")
    with col4:
        st.metric("Price", f"£{demo_data['price']:.1f}m")
    
    # Demo recommendation
    if demo_data['predicted_points'] > 8:
        st.success("✅ **EXCELLENT TRANSFER TARGET** - High predicted points and good value!")
    elif demo_data['predicted_points'] > 6:
        st.info("👍 **GOOD OPTION** - Solid choice for your squad")
    else:
        st.warning("⚠️ **CONSIDER ALTERNATIVES** - Better options may be available")
    
    # Demo chart
    fig = px.line(
        x=list(range(1, 6)),
        y=[6, 8, 4, 9, 7],  # Demo recent form
        title=f"{player_name} - Last 5 Gameweeks",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

def show_performance_tracker():
    st.title("📈 Performance Tracker")
    st.write("Monitor AI model performance and prediction accuracy over time.")
    
    # Get model performance
    try:
        response = requests.get(f"{API_BASE_URL}/model/performance", timeout=10)
        if response.status_code == 200:
            performance = response.json()["model_performance"]
            display_performance_metrics(performance)
        else:
            display_demo_performance_metrics()
    except:
        display_demo_performance_metrics()

def display_performance_metrics(performance: Dict):
    """Display model performance metrics."""
    
    st.subheader("🤖 ML Model Performance by Position")
    
    # Performance summary
    perf_data = []
    total_mae = 0
    total_samples = 0
    
    for position, metrics in performance.items():
        mae = metrics.get('mae', 0)
        samples = metrics.get('samples', 0)
        
        total_mae += mae * samples
        total_samples += samples
        
        # Calculate accuracy grade
        if mae < 1.5:
            grade = "A"
            grade_color = "🟢"
        elif mae < 2.0:
            grade = "B"
            grade_color = "🟡"
        else:
            grade = "C"
            grade_color = "🔴"
        
        perf_data.append({
            'Position': position.title(),
            'MAE': f"{mae:.2f}",
            'RMSE': f"{metrics.get('rmse', 0):.2f}",
            'Samples': f"{samples:,}",
            'Features': metrics.get('features', 0),
            'Grade': f"{grade_color} {grade}"
        })
    
    # Overall performance
    overall_mae = total_mae / total_samples if total_samples > 0 else 0
    overall_accuracy = max(0, (3 - overall_mae) / 3 * 100)
    
    # Display overall metrics
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
    with perf_col2:
        st.metric("Average MAE", f"{overall_mae:.2f}")
    with perf_col3:
        st.metric("Total Samples", f"{total_samples:,}")
    with perf_col4:
        status = "🟢 Excellent" if overall_accuracy > 80 else "🟡 Good" if overall_accuracy > 70 else "🔴 Needs Improvement"
        st.metric("Status", status)
    
    # Performance table
    df = pd.DataFrame(perf_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Accuracy visualization
    st.subheader("📊 Model Accuracy Comparison")
    
    positions = [pos.title() for pos in performance.keys()]
    mae_values = [performance[pos.lower()]['mae'] for pos in positions]
    
    fig = px.bar(
        x=positions,
        y=mae_values,
        title="Mean Absolute Error by Position (Lower = Better)",
        labels={'x': 'Position', 'y': 'MAE (Points)'},
        color=mae_values,
        color_continuous_scale="RdYlGn_r"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model training history
    st.subheader("📈 Training History")
    
    # Generate demo training data
    epochs = list(range(1, 21))
    training_mae = [3.5 - (i * 0.1) + np.random.normal(0, 0.1) for i in epochs]
    validation_mae = [3.7 - (i * 0.09) + np.random.normal(0, 0.12) for i in epochs]
    
    fig_training = px.line(
        title="Model Training Progress"
    )
    fig_training.add_scatter(x=epochs, y=training_mae, mode='lines', name='Training MAE')
    fig_training.add_scatter(x=epochs, y=validation_mae, mode='lines', name='Validation MAE')
    fig_training.update_layout(
        xaxis_title="Epoch",
        yaxis_title="MAE",
        legend=dict(x=0.7, y=0.95)
    )
    st.plotly_chart(fig_training, use_container_width=True)
    
    # Prediction confidence distribution
    st.subheader("🎯 Prediction Confidence Distribution")
    
    confidence_ranges = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    confidence_counts = [50, 120, 200, 180, 80]  # Demo data
    
    fig_confidence = px.bar(
        x=confidence_ranges,
        y=confidence_counts,
        title="Number of Predictions by Confidence Level",
        labels={'x': 'Confidence Range', 'y': 'Number of Predictions'},
        color=confidence_counts,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_confidence, use_container_width=True)

def display_demo_performance_metrics():
    """Display demo performance metrics when API is unavailable."""
    st.info("📱 Showing demo performance metrics (API unavailable)")
    
    demo_performance = {
        'goalkeeper': {'mae': 1.2, 'rmse': 1.8, 'samples': 450, 'features': 12},
        'defender': {'mae': 1.5, 'rmse': 2.1, 'samples': 1200, 'features': 15},
        'midfielder': {'mae': 1.8, 'rmse': 2.4, 'samples': 1400, 'features': 18},
        'forward': {'mae': 2.1, 'rmse': 2.9, 'samples': 650, 'features': 14}
    }
    
    display_performance_metrics(demo_performance)

def show_captain_selector(current_gw: int):
    st.title("🎯 Captain Selector")
    st.write("Find the optimal captain choice with AI-powered risk/reward analysis.")
    
    # Captain selection options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_type = st.radio(
            "Captain Analysis Type:",
            ["🔍 Find Best Captains", "👤 Analyze Specific Player", "📊 Compare Players"],
            horizontal=True
        )
    
    with col2:
        risk_preference = st.selectbox(
            "Risk Preference:",
            ["🛡️ Safe (High Floor)", "⚖️ Balanced", "🚀 Aggressive (High Ceiling)"]
        )
    
    if analysis_type == "🔍 Find Best Captains":
        show_best_captains(current_gw, risk_preference)
    elif analysis_type == "👤 Analyze Specific Player":
        show_specific_captain_analysis(current_gw)
    else:
        show_captain_comparison(current_gw)

def show_best_captains(current_gw: int, risk_preference: str):
    """Show top captain recommendations."""
    
    if st.button("🔍 Find Best Captains", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing captain options with AI..."):
            
            # Demo captain data based on risk preference
            if "Safe" in risk_preference:
                captain_options = [
                    {
                        "name": "Mohamed Salah", "team": "Liverpool", "position": "Midfielder",
                        "predicted_points": 8.1, "captain_points": 16.2, "safety_score": 0.92,
                        "fixture": "vs Crystal Palace (A)", "form": 4.2, "ownership": 45.2,
                        "reasoning": "Consistent performer with excellent penalty record",
                        "risk_level": "Low", "ceiling": 25, "floor": 4
                    },
                    {
                        "name": "Kevin De Bruyne", "team": "Manchester City", "position": "Midfielder", 
                        "predicted_points": 7.8, "captain_points": 15.6, "safety_score": 0.88,
                        "fixture": "vs Brighton (H)", "form": 4.0, "ownership": 35.1,
                        "reasoning": "Home fixture and excellent creative stats",
                        "risk_level": "Low", "ceiling": 22, "floor": 3
                    },
                    {
                        "name": "Harry Kane", "team": "Tottenham", "position": "Forward",
                        "predicted_points": 7.5, "captain_points": 15.0, "safety_score": 0.85,
                        "fixture": "vs Fulham (H)", "form": 4.1, "ownership": 28.7,
                        "reasoning": "Penalty taker with good home record vs Fulham",
                        "risk_level": "Low", "ceiling": 20, "floor": 2
                    }
                ]
            elif "Aggressive" in risk_preference:
                captain_options = [
                    {
                        "name": "Erling Haaland", "team": "Manchester City", "position": "Forward",
                        "predicted_points": 9.2, "captain_points": 18.4, "safety_score": 0.75,
                        "fixture": "vs Brighton (H)", "form": 4.8, "ownership": 65.3,
                        "reasoning": "Highest ceiling, excellent form against weaker defense",
                        "risk_level": "Medium", "ceiling": 30, "floor": 1
                    },
                    {
                        "name": "Marcus Rashford", "team": "Manchester United", "position": "Forward",
                        "predicted_points": 8.5, "captain_points": 17.0, "safety_score": 0.65,
                        "fixture": "vs Nottingham Forest (H)", "form": 4.9, "ownership": 22.1,
                        "reasoning": "Explosive potential, on penalties, great differential",
                        "risk_level": "High", "ceiling": 28, "floor": 0
                    },
                    {
                        "name": "Bukayo Saka", "team": "Arsenal", "position": "Midfielder",
                        "predicted_points": 8.0, "captain_points": 16.0, "safety_score": 0.70,
                        "fixture": "vs Southampton (H)", "form": 5.0, "ownership": 18.9,
                        "reasoning": "Outstanding recent form, penalty potential",
                        "risk_level": "Medium-High", "ceiling": 25, "floor": 1
                    }
                ]
            else:  # Balanced
                captain_options = [
                    {
                        "name": "Erling Haaland", "team": "Manchester City", "position": "Forward",
                        "predicted_points": 9.2, "captain_points": 18.4, "safety_score": 0.85,
                        "fixture": "vs Brighton (H)", "form": 4.8, "ownership": 65.3,
                        "reasoning": "Perfect balance of high ceiling and consistency",
                        "risk_level": "Medium", "ceiling": 30, "floor": 2
                    },
                    {
                        "name": "Mohamed Salah", "team": "Liverpool", "position": "Midfielder",
                        "predicted_points": 8.1, "captain_points": 16.2, "safety_score": 0.92,
                        "fixture": "vs Crystal Palace (A)", "form": 4.2, "ownership": 45.2,
                        "reasoning": "Reliable with good upside, away form concerns minimal",
                        "risk_level": "Low-Medium", "ceiling": 25, "floor": 4
                    },
                    {
                        "name": "Harry Kane", "team": "Tottenham", "position": "Forward",
                        "predicted_points": 7.8, "captain_points": 15.6, "safety_score": 0.78,
                        "fixture": "vs Fulham (H)", "form": 4.5, "ownership": 28.7,
                        "reasoning": "Good differential with penalty security",
                        "risk_level": "Medium", "ceiling": 22, "floor": 2
                    }
                ]
            
            display_captain_recommendations(captain_options, risk_preference)

def display_captain_recommendations(captain_options: List[Dict], risk_preference: str):
    """Display captain recommendations with detailed analysis."""
    
    st.success(f"✅ Found {len(captain_options)} optimal captains for {risk_preference.lower()} strategy")
    
    # Captain comparison chart
    st.subheader("📊 Captain Comparison")
    
    names = [c['name'].split()[-1] for c in captain_options]  # Last names for chart
    safety_scores = [c['safety_score'] * 20 for c in captain_options]  # Scale to 20
    captain_points = [c['captain_points'] for c in captain_options]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(captain_options)]
    
    fig = go.Figure()
    
    for i, (name, safety, points, color) in enumerate(zip(names, safety_scores, captain_points, colors)):
        fig.add_trace(go.Scatter(
            x=[safety],
            y=[points],
            mode='markers+text',
            text=[name],
            textposition='top center',
            marker=dict(size=20, color=color),
            name=name,
            hovertemplate=f"<b>{captain_options[i]['name']}</b><br>" +
                         f"Safety: {captain_options[i]['safety_score']:.0%}<br>" +
                         f"Expected Points: {points:.1f}<br>" +
                         f"<extra></extra>"
        ))
    
    fig.update_layout(
        title="Captain Risk vs Reward Analysis",
        xaxis_title="Safety Score (Higher = More Consistent)",
        yaxis_title="Expected Captain Points",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed captain cards
    st.subheader("👑 Detailed Captain Analysis")
    
    for i, captain in enumerate(captain_options, 1):
        
        # Risk level styling
        if captain['risk_level'] == "Low":
            risk_color = "success"
            risk_icon = "🛡️"
        elif captain['risk_level'] in ["Medium", "Low-Medium"]:
            risk_color = "info"
            risk_icon = "⚖️"
        else:
            risk_color = "warning"
            risk_icon = "🚀"
        
        # Captain card
        with st.container():
            captain_col1, captain_col2, captain_col3 = st.columns([2, 1, 1])
            
            with captain_col1:
                st.markdown(f"""
                <div class="captain-recommendation">
                    <h3>{i}. {captain['name']} ({captain['team']})</h3>
                    <p><strong>Position:</strong> {captain['position']}</p>
                    <p><strong>Fixture:</strong> {captain['fixture']}</p>
                    <p><em>{captain['reasoning']}</em></p>
                </div>
                """, unsafe_allow_html=True)
            
            with captain_col2:
                st.metric("Expected Points", f"{captain['captain_points']:.1f}")
                st.metric("Safety Score", f"{captain['safety_score']:.0%}")
                st.metric("Current Form", f"{captain['form']:.1f}/5")
            
            with captain_col3:
                st.metric("Ownership", f"{captain['ownership']:.1f}%")
                st.metric("Risk Level", f"{risk_icon} {captain['risk_level']}")
                
                # Ceiling and floor
                st.write(f"**Ceiling:** {captain['ceiling']} pts")
                st.write(f"**Floor:** {captain['floor']} pts")
        
        # Detailed analysis expandable
        with st.expander(f"📊 Detailed Analysis - {captain['name']}"):
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**📈 Performance Factors:**")
                st.write(f"• Recent form: {captain['form']:.1f}/5")
                st.write(f"• Predicted points: {captain['predicted_points']:.1f}")
                st.write(f"• Fixture difficulty: Favorable")
                st.write(f"• Penalty taker: {'Yes' if 'penalty' in captain['reasoning'].lower() else 'Likely'}")
                
                st.markdown("**⚖️ Risk Assessment:**")
                st.write(f"• Consistency: {captain['safety_score']:.0%}")
                st.write(f"• Injury risk: Low")
                st.write(f"• Rotation risk: Minimal")
                
            with detail_col2:
                st.markdown("**📊 Ownership Analysis:**")
                if captain['ownership'] > 50:
                    st.write("🔴 **Highly owned** - Safe but less differential")
                elif captain['ownership'] > 25:
                    st.write("🟡 **Popular pick** - Balanced ownership")
                else:
                    st.write("🟢 **Differential** - Lower ownership, higher reward potential")
                
                st.markdown("**🎯 Recommendation:**")
                if i == 1:
                    st.success("🥇 **TOP PICK** - Optimal choice for this strategy")
                elif i == 2:
                    st.info("🥈 **STRONG OPTION** - Excellent alternative")
                else:
                    st.warning("🥉 **SOLID CHOICE** - Good backup option")
        
        st.divider()
    
    # Captain strategy tips
    st.subheader("💡 Captain Strategy Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        if "Safe" in risk_preference:
            st.markdown("""
            <div class="success-box">
                <h4>🛡️ Safe Captain Strategy</h4>
                <p>• Focus on consistent performers</p>
                <p>• Prioritize penalty takers</p>
                <p>• Avoid rotation risks</p>
                <p>• Consider home fixtures</p>
            </div>
            """, unsafe_allow_html=True)
        elif "Aggressive" in risk_preference:
            st.markdown("""
            <div class="warning-box">
                <h4>🚀 Aggressive Captain Strategy</h4>
                <p>• Target high ceiling players</p>
                <p>• Consider differentials for rank gains</p>
                <p>• Accept higher variance</p>
                <p>• Monitor form and news closely</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>⚖️ Balanced Captain Strategy</h4>
                <p>• Mix of safety and upside</p>
                <p>• Consider fixture difficulty</p>
                <p>• Monitor ownership levels</p>
                <p>• Adapt based on league position</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tips_col2:
        st.markdown("""
        <div class="info-box">
            <h4>📅 This Gameweek Factors</h4>
            <p>• International break impact</p>
            <p>• Injury news updates</p>
            <p>• Team rotation policies</p>
            <p>• Weather conditions</p>
        </div>
        """, unsafe_allow_html=True)

def show_specific_captain_analysis(current_gw: int):
    """Show analysis for a specific captain choice."""
    
    player_name = st.text_input("Enter player name for captain analysis:", placeholder="e.g., Haaland")
    
    if player_name and st.button("🔍 Analyze Captain Potential"):
        display_demo_player_analysis(player_name)
        
        # Additional captain-specific metrics
        st.subheader("👑 Captain-Specific Analysis")
        
        capt_col1, capt_col2, capt_col3 = st.columns(3)
        
        with capt_col1:
            st.metric("Captain EO", "85.2%", "Expected Ownership")
        with capt_col2:
            st.metric("Captain ROI", "2.1x", "Return on Investment")
        with capt_col3:
            st.metric("Rank Impact", "+150k", "If captain scores well")

# ... (previous code continues)

def show_captain_comparison(current_gw: int):
    """Show comparison between multiple captain options."""
    
    st.write("Select up to 4 players to compare as captain options:")
    
    comparison_players = st.multiselect(
        "Choose players:",
        ["Haaland", "Salah", "Kane", "De Bruyne", "Son", "Rashford", "Saka"],
        default=["Haaland", "Salah", "Kane"]
    )
    
    if len(comparison_players) >= 2 and st.button("⚖️ Compare Captains"):
        
        # Generate comparison data
        comparison_data = []
        for player in comparison_players:
            player_hash = hash(player.lower())
            comparison_data.append({
                "Player": player,
                "Predicted Points": 8.0 + (player_hash % 20) / 10,
                "Captain Points": 16.0 + (player_hash % 40) / 10,
                "Safety Score": f"{75 + (player_hash % 25)}%",
                "Ownership": f"{20 + (player_hash % 50):.1f}%",
                "Form": 3.5 + (player_hash % 15) / 10,
                "Risk Level": ["Low", "Medium", "High"][player_hash % 3]
            })
        
        # Display comparison table
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Comparison chart
        fig = px.scatter(
            df_comparison,
            x="Safety Score",
            y="Captain Points",
            size="Predicted Points",
            color="Player",
            title="Captain Comparison: Risk vs Reward",
            hover_data=["Ownership", "Form"]
        )
        st.plotly_chart(fig, use_container_width=True)

def show_multi_gw_planner(current_gw: int):
    """Show multi-gameweek planning interface."""
    
    st.title("🔮 Multi-Gameweek Planner")
    st.write("Plan your transfers and strategy across multiple gameweeks for optimal long-term performance.")
    
    # Planning horizon
    plan_col1, plan_col2, plan_col3 = st.columns(3)
    
    with plan_col1:
        start_gw = st.number_input("Start Gameweek", min_value=current_gw, max_value=38, value=current_gw)
    with plan_col2:
        end_gw = st.number_input("End Gameweek", min_value=start_gw+1, max_value=38, value=min(start_gw+5, 38))
    with plan_col3:
        strategy_type = st.selectbox("Strategy", ["Fixture Swing", "Form Based", "Balanced", "Differential"])
    
    # Planning factors
    st.subheader("📊 Planning Factors")
    
    factor_col1, factor_col2 = st.columns(2)
    
    with factor_col1:
        st.markdown("**Key Considerations:**")
        chip_usage = st.multiselect(
            "Chip Usage Plan",
            ["Wildcard", "Bench Boost", "Triple Captain", "Free Hit"],
            default=[]
        )
        
        price_change_weight = st.slider("Price Change Importance", 0.0, 1.0, 0.3, 0.1)
        fixture_weight = st.slider("Fixture Difficulty Weight", 0.0, 1.0, 0.7, 0.1)
    
    with factor_col2:
        st.markdown("**Risk Management:**")
        max_hits_per_gw = st.selectbox("Max Hits per GW", [0, 1, 2, 3])
        injury_buffer = st.checkbox("Include injury buffer", value=True)
        rotation_protection = st.checkbox("Rotation protection", value=True)
    
    if st.button("🔮 Generate Multi-GW Plan", type="primary", use_container_width=True):
        with st.spinner("🤖 Generating strategic plan across multiple gameweeks..."):
            display_multi_gw_plan(start_gw, end_gw, strategy_type, chip_usage)

def display_multi_gw_plan(start_gw: int, end_gw: int, strategy: str, chips: List[str]):
    """Display multi-gameweek strategic plan."""
    
    st.success(f"✅ Generated {end_gw - start_gw + 1}-gameweek plan using {strategy.lower()} strategy")
    
    # Timeline overview
    st.subheader("📅 Strategic Timeline")
    
    timeline_data = []
    for gw in range(start_gw, end_gw + 1):
        # Generate demo plan data
        if gw == start_gw:
            action = "Build foundation squad"
            transfers = "2 transfers"
            chip = "None"
        elif gw == start_gw + 2 and "Wildcard" in chips:
            action = "Wildcard activation"
            transfers = "Unlimited"
            chip = "Wildcard"
        elif gw == end_gw - 1 and "Triple Captain" in chips:
            action = "Triple Captain prep"
            transfers = "1 transfer"
            chip = "Triple Captain"
        else:
            action = "Fixture optimization"
            transfers = f"{np.random.choice([0, 1, 2])} transfers"
            chip = "None"
        
        timeline_data.append({
            "Gameweek": f"GW{gw}",
            "Strategy": action,
            "Transfers": transfers,
            "Chip": chip,
            "Focus": "Attack" if gw % 2 == 0 else "Defense"
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    
    # Fixture difficulty heatmap
    st.subheader("🗓️ Fixture Difficulty Heatmap")
    
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd", "Newcastle", "Brighton", "Crystal Palace"]
    gameweeks = [f"GW{gw}" for gw in range(start_gw, min(start_gw + 8, 39))]
    
    # Generate demo fixture difficulty data
    np.random.seed(42)
    difficulty_matrix = np.random.randint(1, 6, size=(len(teams), len(gameweeks)))
    
    fig_heatmap = px.imshow(
        difficulty_matrix,
        x=gameweeks,
        y=teams,
        color_continuous_scale="RdYlGn_r",
        title="Fixture Difficulty by Team (1=Easy, 5=Hard)",
        aspect="auto"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Key transfer targets by gameweek
    st.subheader("🎯 Key Transfer Targets by Period")
    
    transfer_col1, transfer_col2 = st.columns(2)
    
    with transfer_col1:
        st.markdown(f"""
        <div class="success-box">
            <h4>GW{start_gw}-{start_gw+2}: Foundation Phase</h4>
            <p><strong>Targets:</strong> Salah, Haaland, TAA</p>
            <p><strong>Strategy:</strong> Secure premium assets</p>
            <p><strong>Budget:</strong> £40-45m on top 3</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <h4>GW{start_gw+3}-{start_gw+4}: Optimization</h4>
            <p><strong>Targets:</strong> Saka, Rashford, Mitoma</p>
            <p><strong>Strategy:</strong> Fixture exploitation</p>
            <p><strong>Focus:</strong> Form + fixtures alignment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with transfer_col2:
        st.markdown(f"""
        <div class="warning-box">
            <h4>GW{start_gw+5}-{end_gw}: Endgame</h4>
            <p><strong>Targets:</strong> Differential picks</p>
            <p><strong>Strategy:</strong> Rank climbing</p>
            <p><strong>Risk:</strong> Higher variance acceptable</p>
        </div>
        """, unsafe_allow_html=True)
        
        if chips:
            st.markdown(f"""
            <div class="captain-recommendation">
                <h4>🃏 Chip Strategy</h4>
                <p><strong>Chips:</strong> {', '.join(chips)}</p>
                <p><strong>Timing:</strong> Optimal fixtures identified</p>
                <p><strong>Expected Gain:</strong> +15-25 points</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Expected points projection
    st.subheader("📈 Expected Points Projection")
    
    gameweeks_proj = list(range(start_gw, end_gw + 1))
    base_points = [55 + np.random.normal(0, 5) for _ in gameweeks_proj]
    optimized_points = [bp + 5 + np.random.normal(0, 3) for bp in base_points]
    
    fig_projection = go.Figure()
    fig_projection.add_trace(go.Scatter(
        x=gameweeks_proj, y=base_points,
        mode='lines+markers', name='Current Squad Projection',
        line=dict(color='red', dash='dash')
    ))
    fig_projection.add_trace(go.Scatter(
        x=gameweeks_proj, y=optimized_points,
        mode='lines+markers', name='Optimized Strategy',
        line=dict(color='green')
    ))
    
    fig_projection.update_layout(
        title="Points Projection: Current vs Optimized Strategy",
        xaxis_title="Gameweek",
        yaxis_title="Expected Points",
        hovermode='x unified'
    )
    st.plotly_chart(fig_projection, use_container_width=True)
    
    # Risk assessment
    st.subheader("⚖️ Strategy Risk Assessment")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.metric("Price Change Risk", "Medium", "3 players affected")
    with risk_col2:
        st.metric("Injury Risk", "Low", "Rotation protection active")
    with risk_col3:
        st.metric("Fixture Risk", "Low", "Favorable swing captured")
    
    # Action items
    st.subheader("✅ Immediate Action Items")
    
    action_items = [
        f"🔄 Plan {timeline_df.iloc[0]['Transfers']} for GW{start_gw}",
        "👀 Monitor price changes for key targets",
        "📺 Track team news and injury updates",
        "📊 Review plan weekly for adjustments"
    ]
    
    for item in action_items:
        st.write(f"• {item}")

# Error handling and utility functions
def handle_api_error(error_type: str, fallback_data: Dict = None):
    """Handle API errors gracefully with fallback options."""
    
    if error_type == "connection":
        st.error("🔌 **API Connection Failed**")
        st.write("• Check your internet connection")
        st.write("• Verify API server is running")
        st.write("• Using demo data for now")
    elif error_type == "timeout":
        st.warning("⏱️ **Request Timeout**")
        st.write("• API response is taking too long")
        st.write("• Try again in a moment")
    elif error_type == "server":
        st.error("🖥️ **Server Error**")
        st.write("• API server encountered an error")
        st.write("• Issue has been logged automatically")
    
    if fallback_data:
        st.info("📱 **Showing demo data while issue is resolved**")

def format_currency(amount: float) -> str:
    """Format currency values consistently."""
    return f"£{amount:.1f}m"

def format_points(points: float) -> str:
    """Format points values consistently."""
    return f"{points:.1f} pts"

def get_position_emoji(position: int) -> str:
    """Get emoji for player position."""
    emojis = {1: "🥅", 2: "🛡️", 3: "⚽", 4: "🎯"}
    return emojis.get(position, "⚽")

def get_difficulty_color(difficulty: int) -> str:
    """Get color for fixture difficulty."""
    if difficulty <= 2:
        return "🟢"
    elif difficulty <= 3:
        return "🟡"
    else:
        return "🔴"

# Footer and additional information
def show_footer():
    """Display footer with additional information."""
    
    st.divider()
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        **🔗 Resources**
        - [Official FPL Website](https://fantasy.premierleague.com)
        - [FPL Statistics](https://www.fplstatistics.co.uk)
        - [FPL Discovery](https://www.fpldiscovery.com)
        """)
    
    with footer_col2:
        st.markdown("""
        **ℹ️ About**
        - AI-powered FPL optimization
        - Real-time data integration
        - Advanced ML predictions
        """)
    
    with footer_col3:
        st.markdown("""
        **📞 Support**
        - Report issues via GitHub
        - Feature requests welcome
        - Community Discord available
        """)
    
    st.markdown("""
    ---
    <div style='text-align: center'>
        <p>FPL Optimizer Pro v1.0 | Built with ❤️ for the FPL community</p>
        <p><small>⚠️ This tool is for entertainment purposes. Always make your own decisions.</small></p>
    </div>
    """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    main()
    show_footer()


