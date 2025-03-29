import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go

# Load CSV data
df1 = pd.read_csv('ATB cles staph aureus.csv')
df2 = pd.read_csv('staph aureus autre atb.csv')
df3 = pd.read_csv('staph aureus phenotypes R.csv')

# Preprocessing
df3 = df3[~df3["Month"].isin(["Total", "Prevalence %"])]  # Exclude summary rows
df3["Month"] = pd.to_datetime(df3["Month"], format='%B', errors='coerce')  # Convert to datetime
df3 = df3.dropna(subset=["Month"])  # Drop any rows where Month couldn't be converted
df3.sort_values(by="Month", inplace=True)


# Compute metrics for MoM and YoY changes
def calculate_changes(df, column):
    df["MoM Change (%)"] = df[column].pct_change() * 100
    df["YoY Change (%)"] = df[column].pct_change(periods=12) * 100
    return df

df3 = calculate_changes(df3, "Total")

# App layout
st.title("Antibiotic Resistance Dashboard")
st.header("Antibiotic Resistance Trends - June 8, 2024")

tab1, tab2, tab3 = st.tabs(["Overall Trends", "Phenotype Analysis", "Comparative Analysis"])

# Tab 1: Overall Trends
with tab1:
    st.subheader("Month-over-Month and Year-over-Year Analysis")
    fig_mom = px.bar(df3, x="Month", y="MoM Change (%)", title="Month-over-Month Change", text="MoM Change (%)")
    st.plotly_chart(fig_mom)

# Tab 2: Phenotype Analysis
with tab2:
    st.subheader("Resistance Trends and Proportions")
    fig_line = px.line(df3, x="Month", y=["MRSA", "VRSA", "Wild", "others"], title="Resistance Proportions Over Time")
    st.plotly_chart(fig_line)

    fig_stacked = go.Figure()
    for col in ["MRSA", "VRSA", "Wild", "others"]:
        fig_stacked.add_trace(go.Scatter(x=df3["Month"], y=df3[col], mode="lines", stackgroup="one", name=col))
    fig_stacked.update_layout(title="Stacked Area Chart of Phenotypes", xaxis_title="Month", yaxis_title="Cases")
    st.plotly_chart(fig_stacked)

    st.metric(label="Total Cases (Latest Month)", value=df3["Total"].iloc[-1])

# Tab 3: Comparative Analysis
with tab3:
    st.subheader("Antibiotic Performance Comparison")
    comparison_data = df1.groupby("Month").mean().reset_index()
    fig_bar = px.bar(comparison_data, x="Month", y=["%R_Teicoplanine", "%R_Vancomycine", "%R_Oxacilline"], barmode="group", title="Grouped Resistance Rates")
    st.plotly_chart(fig_bar)

    st.subheader("Top and Worst Performing Antibiotics")
    resistance_summary = {
        "Antibiotic": ["Teicoplanine", "Vancomycine", "Oxacilline", "Gentamycine"],
        "Resistance Rate (%)": [
            df1["%R_Teicoplanine"].mean(),
            df1["%R_Vancomycine"].mean(),
            df1["%R_Oxacilline"].mean(),
            df1["%R_Gentamycine"].mean(),
        ]
    }
    summary_df = pd.DataFrame(resistance_summary).sort_values(by="Resistance Rate (%)")
    st.write("**Top Performing Antibiotic**")
    st.metric(label=summary_df["Antibiotic"].iloc[0], value=f"{summary_df['Resistance Rate (%)'].iloc[0]:.2f}%")
    st.write("**Worst Performing Antibiotic**")
    st.metric(label=summary_df["Antibiotic"].iloc[-1], value=f"{summary_df['Resistance Rate (%)'].iloc[-1]:.2f}%")

# Improved layout with two-column layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("Resistance Metrics")
    st.metric(label="Resistance Rate Teicoplanine", value=f"{df1['%R_Teicoplanine'].iloc[-1]:.2f}%")
    st.metric(label="Resistance Rate Vancomycine", value=f"{df1['%R_Vancomycine'].iloc[-1]:.2f}%")
with col2:
    st.subheader("Resistance Metrics (Contd.)")
    st.metric(label="Resistance Rate Oxacilline", value=f"{df1['%R_Oxacilline'].iloc[-1]:.2f}%")
    st.metric(label="Resistance Rate Gentamycine", value=f"{df1['%R_Gentamycine'].iloc[-1]:.2f}%")
