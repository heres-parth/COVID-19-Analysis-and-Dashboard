import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import streamlit as st

# Load the dataset
st.title("Interactive COVID-19 Dashboard")
st.markdown("This dashboard provides interactive visualizations and forecasts for COVID-19 cases.")

@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_clean_complete.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop("Province/State",axis=1,inplace=True)
    return df

df = load_data()

# Data overview
st.header("Data Overview")
st.write("### Dataset Preview")
st.dataframe(df.head())
#st.write("### Dataset Information")
#st.write(df.info())

# Drop State column due to missing data

# Display summary statistics
st.write("### Dataset Summary")
st.write(df.describe())

# Global Distribution of COVID-19 Cases
st.header("Global Distribution of COVID-19 Cases")
latest_data = df[df['Date'] == df['Date'].max()]
global_totals = latest_data[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum()

fig_global_pie = px.pie(
    values=global_totals,
    names=global_totals.index,
    title='Global Distribution of COVID-19 Cases'
)
st.plotly_chart(fig_global_pie)

# Time-Series Trends of COVID-19 Cases
st.header("Time-Series Trends of COVID-19 Cases")
global_trends = df.groupby('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()

fig_global_trends = px.line(
    global_trends,
    x='Date',
    y=['Confirmed', 'Deaths', 'Recovered', 'Active'],
    title='Time-Series Trends of COVID-19 Cases'
)
st.plotly_chart(fig_global_trends)

# Top 10 Countries by Confirmed Cases
st.header("Top 10 Countries by Confirmed Cases")
top_countries = latest_data.groupby('Country/Region')['Confirmed'].sum().nlargest(10).reset_index()

fig_top_countries = px.bar(
    top_countries,
    x='Confirmed',
    y='Country/Region',
    orientation='h',
    title='Top 10 Countries by Confirmed Cases',
    color='Confirmed'
)
st.plotly_chart(fig_top_countries)

# Correlation Heatmap
st.header("Correlation Heatmap")
numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active']
correlation_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Facebook Prophet Model
st.header("COVID-19 Forecasting Using Facebook Prophet")
prophet_data = global_trends[['Date', 'Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'})

# Initialize and fit the Prophet model
model = Prophet()
model.fit(prophet_data)

future = model.make_future_dataframe(periods=7)  # Predict for the next week
forecast = model.predict(future)

st.write("### Forecasted Data")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast)

st.write("### Forecast Components")
fig_components = plot_components_plotly(model, forecast)
st.plotly_chart(fig_components)

# Forecasted vs Historical Interactive Visualization
st.header("Interactive Historical and Forecasted Trends")
forecast_fig = px.line(
    forecast,
    x='ds',
    y=['yhat', 'yhat_lower', 'yhat_upper'],
    labels={'ds': 'Date', 'value': 'Predicted Cases'},
    title='COVID-19 Predicted Cases for the Next Week'
)
forecast_fig.add_scatter(x=prophet_data['ds'], y=prophet_data['y'], mode='lines', name='Historical Data')
st.plotly_chart(forecast_fig)

st.write("### About the App")
st.markdown("This dashboard leverages **Streamlit**, **Facebook Prophet**, and **Plotly** to analyze and visualize COVID-19 trends interactively.")

st.write("Made by Parth Aland")
