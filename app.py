# app.py

import streamlit as st
import pandas as pd

# Load the optimized lineup
DATA_PATH = "data/processed/optimized_cleaned.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["now_cost"] = df["now_cost"].round(1)
    df["predicted_points"] = df["predicted_points"].round(2)
    return df

def main():
    st.set_page_config(page_title="FPL Lineup Optimizer", layout="wide")
    st.title("🏆 FPL Lineup Optimizer")
    st.markdown("Built with 💻 ML + ⚽️ Strategy | [GitHub](https://github.com/yourusername/fpl_optimizer)")

    df = load_data()

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    positions = st.sidebar.multiselect("Filter by Position", df["position"].unique(), default=df["position"].unique())
    clubs = st.sidebar.multiselect("Filter by Club", df["team_name"].unique(), default=df["team_name"].unique())

    filtered_df = df[(df["position"].isin(positions)) & (df["team_name"].isin(clubs))]

    # Squad overview
    st.subheader("📋 Optimized Squad")
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

    # Metrics
    total_points = filtered_df["predicted_points"].sum()
    total_cost = filtered_df["now_cost"].sum()

    col1, col2 = st.columns(2)
    col1.metric("📊 Total Predicted Points", f"{total_points:.2f}")
    col2.metric("💰 Total Squad Cost (£)", f"{total_cost:.1f} / 100")

    st.markdown("---")
    st.info("Data is based on latest FPL stats and your trained ML model.")

if __name__ == "__main__":
    main()
