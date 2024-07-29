import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Title of the Streamlit app
st.title('Compost Anomaly Detection')

# Displaying a welcoming message
st.write("Hello, 👋 Welcome to the Compost Anomaly Detection Dashboard. Here you can see the latest anomaly detection plot.")

# Sidebar information
with st.sidebar:
    st.write('*Anomaly Detection in Compost Data*')
    st.caption('''**We use advanced machine learning models to detect anomalies in the compost data.
    The plot below shows the detected anomalies over time. The plot is updated regularly to reflect the latest data.**
    ''')

# Function to display the plot
def display_anomaly_plot():
    st.image('anomaly_detection_plot.png', caption='Anomaly Detection Plot')

# Main content
st.header('Anomaly Detection Plot')
display_anomaly_plot()

# Information about updates
st.write("The plot is updated regularly to reflect the latest data. Last updated at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))