import plotly.express as px
import pandas as pd

def create_climate_timeline(data):
    fig = px.line(data, x='Year', y='Temp_anomaly', 
                 title='Anomalías de Temperatura Global',
                 labels={'Temp_anomaly': 'Δ Temperatura (°C)'},
                 template='plotly_dark')
    
    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(showspikes=True),
        yaxis=dict(showspikes=True)
    )
    return fig

def create_co2_temp_corr(data):
    fig = px.scatter(data, x='CO2', y='Temp_anomaly',
                    trendline='lowess',
                    title='Correlación CO2 vs Temperatura',
                    color='Year',
                    template='plotly_dark')
    return fig