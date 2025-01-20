import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import plotly.express as px
import openai  # Make sure to install the OpenAI package
import time
import google.generativeai as genai

# Set page configuration
st.set_page_config(page_title="Flow Health Dashboard", layout="wide")

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/PlayerR18/River-Water-Purification-Prediction/refs/heads/main/water_potability.csv')

# Define the columns that need to be converted to numeric
columns_to_check = ['Temp Min', 'Temp Max', 'Do Min', 'Con Min', 'Con Max', 
                    'BCOD Min', 'BCOD Max', 'Ni Min', 'Ni Max']
df[columns_to_check] = df[columns_to_check].apply(pd.to_numeric, errors='coerce')

# Handle missing values by imputing with the column mean
imputer = SimpleImputer(strategy='mean')
df[columns_to_check] = imputer.fit_transform(df[columns_to_check])

# Replace infinite values with NaN, then handle missing values again
df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
df[columns_to_check] = imputer.fit_transform(df[columns_to_check])

# Drop unnecessary columns
x = df.drop(['Station Code', 'LOCATION', 'STATE', 'Do Min', 'Do Max', 'Dissolved Oxygen', 
             'Con Min', 'Con Max', 'Conductivity', 'FC Min', 'FC Max', 'Faecal Coliform', 
             'TC Min', 'TC Max', 'Total Coliform', 'Potability', 'Temp Min', 'Temp Max', 
             'pH Min', 'pH Max', 'BCOD Min', 'BCOD Max', 'Ni Min', 'Ni Max'], axis=1)
y = df['Potability']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=84)

# Initialize the Random Forest model
model_rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=200)
if np.all(np.isfinite(x_train)) and np.all(np.isfinite(y_train)):
    model_rf.fit(x_train, y_train)

# Function to generate random data for dynamic graphs
def generate_random_data():
    data = {
        "Year": np.arange(2025, 2035),
        "pH": np.random.uniform(6.5, 8.5, 10),
        "BoD": np.random.uniform(2.0, 5.0, 10),
        "Temperature": np.random.uniform(20, 35, 10),
        "Nitrate": np.random.uniform(0.1, 2.5, 10),
    }
    return pd.DataFrame(data)

# Function to update graphs with random data
def update_graphs():
    df = generate_random_data()

    fig1 = px.line(df, x="Year", y="pH", title="pH Trends", labels={"Year": "Year", "pH": "pH"}, markers=True)
    fig2 = px.line(df, x="Year", y="BoD", title="BoD Trends", labels={"Year": "Year", "BoD": "BoD (mg/L)"}, markers=True)
    fig3 = px.line(df, x="Year", y="Temperature", title="Temperature Trends", labels={"Year": "Year", "Temperature": "Temperature (°C)"}, markers=True)
    fig4 = px.line(df, x="Year", y="Nitrate", title="Nitrate Trends", labels={"Year": "Year", "Nitrate": "Nitrate (mg/L)"}, markers=True)

    return fig1, fig2, fig3, fig4

# AI Assistant Function
def ai_assistant(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant for the Flow Health Dashboard. You provide insights about the project, water quality, and dashboard functionality."},
                {"role": "user", "content": user_input},
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

# Sidebar Navigation
st.sidebar.title("Navigation")
navigation = st.sidebar.radio("Go to:", ["Home", "Graphs", "Water Potability Prediction", "AI Assistant"])

if navigation == "Home":
    st.title("🌊 Welcome to Flow Health Dashboard")
    st.header("""
                 Know Your River's Health at a Glance
                """)
    st.markdown("""
    "Flow Health" is an ambitious project focused on environmental sustainability, specifically targeting the salinity and cleanliness of Indian rivers. By leveraging the power of machine learning, this initiative aims to analyze vast datasets and deliver accurate, real-time insights into river health. The project aspires to provide actionable solutions for preserving water quality and ensuring the long-term sustainability of these vital water resources.  

Flow Health seeks to empower policymakers, researchers, and local communities with data-driven tools to combat river pollution and salinity issues. Through predictive modeling and continuous monitoring, it can identify patterns, detect anomalies, and predict future changes in river conditions. This innovative approach aims to bridge the gap between advanced technology and environmental stewardship.  

In addition to its technical focus, the project highlights the importance of community involvement and sustainable practices. By addressing critical water challenges, Flow Health hopes to contribute to India's broader environmental conservation efforts and support the global goal of sustainable development.
                """)

elif navigation == "Graphs":
    st.title("📊 Predicted Trends for River Health Parameters")

    col1, col2 = st.columns(2)
    fig1, fig2, fig3, fig4 = update_graphs()

    with col1:
        st.subheader("pH")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("BoD")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Temperature")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Nitrate")
        st.plotly_chart(fig4, use_container_width=True)

elif navigation == "Water Potability Prediction":
    st.title("💧 Water Potability Prediction")

    temperature = st.number_input('Enter Temperature (°C):', min_value=-50.0, max_value=50.0, value=28.0)
    ph = st.number_input('Enter pH Level:', min_value=0.0, max_value=14.0, value=7.4)
    bio_chemical_oxygen_demand = st.number_input('Enter Bio Chemical Oxygen Demand (mg/L):', min_value=0.0, max_value=100.0, value=2.7)
    nitrate = st.number_input('Enter Nitrate Level (mg/L):', min_value=0.0, max_value=50.0, value=1.95)

    new_user_data = pd.DataFrame({
        'Temperature': [temperature],
        'pH': [ph],
        'Bio Chemical Oxygen Demand': [bio_chemical_oxygen_demand],
        'Nitrate': [nitrate]
    })

    prediction_rf = model_rf.predict(new_user_data)
    if prediction_rf == 1:
        st.success('The water is potable!')
    else:
        st.error('The water is not potable.')

elif navigation == "AI Assistant":
    st.title("🤖 AI Assistant")
    st.markdown("Ask me anything about this project, river health, or dashboard features.")
    user_input = st.text_input("Enter your question:")
    if user_input:
        genai.configure(api_key="AIzaSyBSBbhgiPU26mS-AQINv7z6PavJsX5vZvU")
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(user_input)
        st.markdown(f"*Assistant:* {response.text}")

st.markdown("---")
st.markdown("Flow Health | Hackathon Project | Made with ❤️ by Team CodeRed")
