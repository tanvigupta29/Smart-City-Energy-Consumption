import streamlit as st
import pandas as pd
import plotly.express as px

# Load datasets
df_actual = pd.read_csv("data/simulated_energy_consumption.csv")
df_forecast = pd.read_csv("data/all_forecasts.csv")

# Convert month to datetime
df_actual['month'] = pd.to_datetime(df_actual['month'])
df_forecast['month'] = pd.to_datetime(df_forecast['month'])

# Sidebar: Filters
st.sidebar.title("🔍 Filter Options")
states = sorted(df_actual['state'].unique())
selected_state = st.sidebar.selectbox("Select State", states)

wards = sorted(df_actual[df_actual['state'] == selected_state]['ward_id'].unique())
selected_ward = st.sidebar.selectbox("Select Ward", wards)

# Page Header
st.title("📊 Smart City Energy Dashboard")
st.markdown("Analyze and visualize energy usage across city wards to identify trends, high-consumption zones, and future projections.")

# Tabs Layout
tab1, tab2, tab3 = st.tabs(["📈 Ward Trend", "🌟 Summary", "🏆 Top 5 Wards"])

# === TAB 1: Ward Trend ===
with tab1:
    st.subheader(f"Consumption Trend — {selected_state} / {selected_ward}")

    df_filtered = df_actual[(df_actual['state'] == selected_state) & (df_actual['ward_id'] == selected_ward)]
    df_forecast_filtered = df_forecast[(df_forecast['state'] == selected_state) & (df_forecast['ward_id'] == selected_ward)]

    fig = px.line(df_filtered, x='month', y='consumption_kwh', title='Actual Consumption', markers=True)
    fig.add_scatter(x=df_forecast_filtered['month'], y=df_forecast_filtered['predicted_consumption'],
                    mode='lines+markers', name='Forecast', line=dict(dash='dot'))
    fig.update_layout(xaxis_title='Month', yaxis_title='Consumption (kWh)', legend_title="Legend")
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: Summary Stats ===
with tab2:
    st.subheader(f"⚡ Summary for {selected_state} / {selected_ward}")

    recent_actual = df_filtered.sort_values(by="month", ascending=False).iloc[0]['consumption_kwh']
    next_month_forecast = df_forecast_filtered[df_forecast_filtered['month'] > df_filtered['month'].max()].iloc[0]['predicted_consumption']

    st.metric("Last Month Actual", f"{recent_actual:,.2f} kWh")
    st.metric("Next Month Forecast", f"{next_month_forecast:,.2f} kWh", delta=f"{next_month_forecast - recent_actual:,.2f} kWh")

    avg = df_filtered['consumption_kwh'].mean()
    trend = df_filtered['consumption_kwh'].iloc[-1] - df_filtered['consumption_kwh'].iloc[0]

    st.write(f"**Avg Monthly Consumption**: `{avg:.2f} kWh`")
    st.write(f"**12-Month Net Change**: `{trend:+.2f} kWh`")

# === TAB 3: Top 5 Wards in the State ===
with tab3:
    st.subheader(f"🏆 Top 5 High-Consumption Wards in {selected_state}")
    top_wards = df_actual[df_actual['state'] == selected_state].groupby('ward_id')['consumption_kwh'].sum().nlargest(5).reset_index()

    fig_top = px.bar(top_wards, x='ward_id', y='consumption_kwh',
                     labels={'ward_id': 'Ward', 'consumption_kwh': 'Total Consumption (kWh)'},
                     color='consumption_kwh', color_continuous_scale='reds')
    st.plotly_chart(fig_top, use_container_width=True)
