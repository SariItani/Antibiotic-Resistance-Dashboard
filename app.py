import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(
    page_title="Antibiotic Resistance Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data():
    # Load CSV data
    df1 = pd.read_csv('ATB cles staph aureus.csv')
    df2 = pd.read_csv('staph aureus autre atb.csv')
    df3 = pd.read_csv('staph aureus phenotypes R.csv')

    # Preprocessing for phenotypes data
    df3 = df3[~df3["Month"].isin(["Total", "Prevalence %"])]
    df3["Month"] = pd.to_datetime(
        df3["Month"] + " 2024", 
        format='%B %Y', 
        errors='coerce'
    )
    df3 = df3.dropna(subset=["Month"])
    df3.sort_values(by="Month", inplace=True)
    
    # Rename phenotype columns to be more descriptive
    df3 = df3.rename(columns={
        "Wild": "MSSA",  # Methicillin-Sensitive S. aureus
        "others": "Other Resistance"  # Other resistance patterns
    })
    
    # Compute metrics
    df3["MoM Change (%)"] = df3["Total"].pct_change() * 100
    df3["YoY Change (%)"] = df3["Total"].pct_change(periods=12) * 100
    
    return df1, df2, df3

df1, df2, df3 = load_and_process_data()

# Create tabs with new structure
tab_overview, tab_metrics, tab_trends, tab_phenotypes, tab_clinical = st.tabs([
    "Overview", 
    "Key Metrics", 
    "Temporal Trends", 
    "Phenotype Analysis", 
    "Clinical Guidance"
])

# Add sidebar filters
with st.sidebar:
    st.header("Filters")
    
    min_date = df3["Month"].min()
    max_date = df3["Month"].max()
    selected_range = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    selected_phenotypes = st.multiselect(
        "Select phenotypes to highlight",
        options=["MRSA", "VRSA", "MSSA", "Other Resistance"],
        default=["MRSA", "VRSA"]
    )

# Overview Tab
with tab_overview:
    # Apply filters
    filtered_df3 = df3[
        (df3["Month"] >= pd.to_datetime(selected_range[0])) & 
        (df3["Month"] <= pd.to_datetime(selected_range[1]))
    ]
    
    st.subheader("Dashboard Summary")
    
    # Calculate statistics using filtered data
    latest_month = filtered_df3.iloc[-1]
    mrsa_rate = (latest_month["MRSA"] / latest_month["Total"]) * 100
    vrsa_cases = latest_month["VRSA"]
    total_cases = filtered_df3["Total"].sum()
    
    # Metrics columns (unchanged, but using filtered data)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cases Analyzed", f"{total_cases:,}")
    with col2:
        st.metric("Current MRSA Rate", f"{mrsa_rate:.1f}%")
    with col3:
        st.metric("VRSA Cases Detected", vrsa_cases, 
                 delta="⚠️ Immediate attention" if vrsa_cases > 0 else None)
    
    # Early warning system using filtered data
    st.subheader("Resistance Alerts")
    if mrsa_rate > filtered_df3["MRSA"].sum()/filtered_df3["Total"].sum()*100 * 1.2:
        val = mrsa_rate/(filtered_df3['MRSA'].sum()/filtered_df3['Total'].sum()*100)-1
        st.warning(f"⚠️ MRSA cases are {val:.0%} above average")
    if vrsa_cases > 0:
        st.error("🚨 VRSA cases detected - immediate attention required")
    
    # Filtered stacked chart
    st.subheader("Phenotype Distribution Overview")
    fig_stacked = go.Figure()
    for col in selected_phenotypes:
        fig_stacked.add_trace(go.Scatter(
            x=filtered_df3["Month"], 
            y=filtered_df3[col], 
            mode="lines", 
            stackgroup="one", 
            name=col
        ))
    fig_stacked.update_layout(
        title="Phenotype Distribution Over Time",
        xaxis_title="Month",
        yaxis_title="Cases"
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

# Key Metrics Tab
with tab_metrics:
    st.subheader("Antibiotic Performance Summary")
    
    # Create comprehensive antibiotic summary table
    summary_data = []
    antibiotics = {
        'Teicoplanine': df1,
        'Vancomycine': df1,
        'Oxacilline': df1,
        'Gentamycine': df1,
        'Clindamycine': df2,
        'Linezolide': df2,
        'Daptomycine': df2,
        'Cotrimoxazole': df2
    }
    
    for abx, df in antibiotics.items():
        res_col = f'%R_{abx}'
        if res_col in df.columns:
            latest = df[res_col].iloc[-1]
            mean_res = df[res_col].mean()
            trend = df[res_col].pct_change().iloc[-1] * 100
            summary_data.append({
                'Antibiotic': abx,
                'Current Resistance (%)': latest,
                'Mean Resistance (%)': mean_res,
                'Recent Trend (%)': trend
            })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Mean Resistance (%)')
    
    # Format the values before passing to AgGrid
    formatted_df = summary_df.copy()
    for col in ['Current Resistance (%)', 'Mean Resistance (%)', 'Recent Trend (%)']:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%")
    
    # Display interactive table
    gb = GridOptionsBuilder.from_dataframe(formatted_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gridOptions = gb.build()
    
    AgGrid(
        formatted_df,
        gridOptions=gridOptions,
        height=400,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True
    )
    
    # Top/bottom performers
    st.subheader("Performance Extremes")
    col1, col2 = st.columns(2)
    with col1:
        best = summary_df.iloc[0]
        st.metric(
            label=f"Best: {best['Antibiotic']}",
            value=f"{best['Current Resistance (%)']:.1f}%",
            delta=f"{best['Recent Trend (%)']:.1f}% (trend)"
        )
    with col2:
        worst = summary_df.iloc[-1]
        st.metric(
            label=f"Worst: {worst['Antibiotic']}",
            value=f"{worst['Current Resistance (%)']:.1f}%",
            delta=f"{worst['Recent Trend (%)']:.1f}% (trend)"
        )

with tab_trends:
    st.subheader("Temporal Analysis")
    
    # Month-over-Month changes with filtered data
    fig_mom = px.bar(
        filtered_df3, 
        x="Month", 
        y="MoM Change (%)", 
        title="Month-over-Month Change in Total Cases",
        text="MoM Change (%)"
    )
    fig_mom.update_traces(
        marker_color=np.where(filtered_df3["MoM Change (%)"] > 0, 'red', 'green')
    )
    st.plotly_chart(fig_mom, use_container_width=True)
    
    # Year-over-Year changes if we had enough data
    if len(filtered_df3) > 12:
        fig_yoy = px.bar(
            filtered_df3[12:],  # Skip first year for YoY comparison
            x="Month", 
            y="YoY Change (%)", 
            title="Year-over-Year Change in Total Cases",
            text="YoY Change (%)"
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

with tab_phenotypes:
    st.subheader("Phenotype Distribution")
    
    fig_stacked = go.Figure()
    for col in selected_phenotypes:
        fig_stacked.add_trace(go.Scatter(
            x=filtered_df3["Month"], 
            y=filtered_df3[col], 
            mode="lines", 
            stackgroup="one", 
            name=col
        ))
    fig_stacked.update_layout(
        title="Stacked Area Chart of Phenotypes",
        xaxis_title="Month",
        yaxis_title="Cases"
    )
    st.plotly_chart(fig_stacked, use_container_width=True)
    
    st.subheader("Phenotype Proportions")
    # Calculate percentages using filtered data
    for phenotype in ["MRSA", "VRSA"]:
        filtered_df3[f"{phenotype}_pct"] = filtered_df3[phenotype] / filtered_df3["Total"] * 100
    fig_prop = px.line(
        filtered_df3, 
        x="Month", 
        y=["MRSA_pct", "VRSA_pct"],
        labels={"value": "Percentage of Total Cases"},
        title="Resistant Phenotypes as Percentage of Total Cases"
    )
    st.plotly_chart(fig_prop, use_container_width=True)

with tab_clinical:
    st.subheader("Therapy Recommendations")
    
    current_rates = {
        'Oxacilline': df1['%R_Oxacilline'].iloc[-1],
        'Vancomycine': df1['%R_Vancomycine'].iloc[-1],
        'Daptomycine': df2['%R_Daptomycine'].iloc[-1],
        'Linezolide': df2['%R_Linezolide'].iloc[-1]
    }
    
    mrsa_rate = (df3["MRSA"].iloc[-1] / df3["Total"].iloc[-1]) * 100
    
    st.markdown("### Empirical Therapy Guidelines")
    
    if mrsa_rate > 10:
        st.warning(f"**High MRSA Prevalence ({mrsa_rate:.1f}%)**")
        st.markdown(f"""
        - **First-line for suspected MRSA:** Vancomycin (resistance: {current_rates['Vancomycine']:.1f}%) or Linezolid (resistance: {current_rates['Linezolide']:.1f}%)
        - **Alternatives:** Daptomycin (resistance: {current_rates['Daptomycine']:.1f}%)
        - Reserve other agents for confirmed susceptibilities
        """)
    else:
        st.success(f"**Moderate MRSA Prevalence ({mrsa_rate:.1f}%)**")
        st.markdown(f"""
        - **First-line for MSSA:** Beta-lactams (Oxacillin resistance: {current_rates['Oxacilline']:.1f}%)
        - **MRSA coverage when indicated:** Vancomycin (resistance: {current_rates['Vancomycine']:.1f}%)
        - **Penicillin-allergic patients:** Consult antimicrobial stewardship
        """)
    
    st.markdown("### Key Considerations")
    st.markdown(f"""
    - Oxacillin resistance implies resistance to all beta-lactams
    - Current Daptomycin effectiveness: {100-current_rates['Daptomycine']:.1f}%
    - Linezolid remains effective in {100-current_rates['Linezolide']:.1f}% of cases
    - Local resistance patterns may vary from these aggregate numbers
    """)

