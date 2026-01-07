import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(layout="wide")

# Load your pre_trained model
with open('qda_model.pk1', 'rb') as f:
    qm2 = pickle.load(f)

# Load feature importance from an Excel file
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

# Load the feature importance DataFrame
final_fi = load_feature_importance("feature_importance.xlsx")    # Replace with your file path

# Sidebar setup
image_sidebar = Image.open('Pic 1.png')   # Replace with your image file
st.sidebar.image(image_sidebar, width='stretch')
st.sidebar.header('Heart Disease Features')

# Feature selection on sidebar
def get_user_input():
    age = st.sidebar.number_input('Age', min_value=0, max_value=200, step=1, value=30)
    resting_bp = st.sidebar.number_input('Resting Blood Pleasure (No)', min_value=0, max_value=200, step=1, value=120)
    cholestrol = st.sidebar.number_input('Cholestrol (No)', min_value=0, max_value=500, step=1, value=200)
    max_hr = st.sidebar.number_input('Maximum Heart Rate (No)', min_value=0, max_value=400, step=1, value=180)
    old_peak = st.sidebar.number_input('Oldpeak ST depression (No)', min_value=0, max_value=10, step=1, value=2)
    sex = st.sidebar.selectbox('Sex', ['F', 'M'])
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ['Asymptomatic', 'Atypical angina', 'Non-anginal pain', 'Typical angina'])
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar', ['No', 'Yes'])
    resting_ecg = st.sidebar.selectbox('Resting Electrocardiographic Type', ['Left ventricular hypertrophy ', 'Normal', 'ST-T wave abnormality'])
    exercise_angina = st.sidebar.selectbox('Exercise-induced angina (Y/N?)', ['N', 'Y'])
    st_slope = st.sidebar.selectbox('Slope of the peak exercise ', ['Down', 'Flat', 'Up'])
    
    user_data = {
        'Age': age,
        'Resting_BP': resting_bp,
        'Cholestrol': cholestrol,
        'Maximum Heart Rate': max_hr,
        'Oldpeak ST depression': old_peak,
        f'Sex_{sex}': 1,
        f'Chest Pain Type_{chest_pain_type}': 1,
        f'Fasting Blood Sugar_{fasting_bs}': 1,
        f'Resting Electrocardiographic Type_{resting_ecg}': 1,
        f'Exercise-induced angina_{exercise_angina}': 1,
        f'Slope of the peak exercise_{st_slope}': 1,
    }
    return user_data

# Top banner
image_banner = Image.open('Pic 2.png')    # Replace with your image file
st.image(image_banner, width='stretch')

# Centered title
st.markdown("<h1 style='text-align: center;'>Heart Disease Prediction App</h1>", unsafe_allow_html=True)

# Split layout into two columns
left_col, right_col = st.columns(2)

# Left column: Feature Importance Interactive Bar Chart
with left_col:
    st.header("Feature Importance")
    
    # Sort feature importance DataFrame by "Feature Importance Score'
    final_fi_sorted = final_fi.sort_values(by='Feature Importance Score', ascending=True)

    # Create interactive bar chart with plotly
    fig = px.bar(
        final_fi_sorted,
        x='Feature Importance Score',
        y='Variable',
        orientation='h',
        title="Feature Importance",
        labels={'Feature Importance Score': 'Importance', 'Variable': 'Feature'},
        text='Feature Importance Score',
        color_discrete_sequence=['#48a3b4']   # Custom bar color
    )
    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Variable",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, width='stretch')

# Right Column: Prediction Interface
with right_col:
    st.header("Predict Heart Disease")
    
    # User inputs from sidebar
    user_data = get_user_input()

    # Transform the input into the required format
    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])

    # Feature list (same order as used during model training)
    features = [
        'Age', 'Resting_BP', 'Cholestrol', 'Maximum Heart Rate', 'Oldpeak ST depression', 'Sex_F', 'Sex_M',
        'Chest Pain Type_Asymptomatic', 'Chest Pain Type_Atypical angina',
        'Chest Pain Type_Non-anginal pain', 'Chest Pain Type_Typical angina', 'Fasting Blood Sugar_N',
        'Fasting Blood Sugar_Y', 'Resting Electrocardiographic Type_Left ventricular hypertrophy', 
        'Resting Electrocardiographic Type_Normal', 'Resting Electrocardiographic Type_ST-T wave abnormality',
        'Exercise-induced angina_N', 'Exercise-induced angina_Y', 'Slope of the peak exercise_Down',
        'Slope of the peak exercise_Flat', 'Slope of the peak exercise_Up'
    ]

    # Predict button
    if st.button("Predict"):
        input_array = prepare_input(user_data, features)
        prediction = qm2.predict(input_array)
        st.subheader("Predicted Heart Disease")
        st.write(prediction)
        if prediction == 0:
            st.write("No sign of Heart Disease")
        else:
            st.write("Heart Disease Suspected")
        #st.write(f"${prediction[0]:,.2f}")

# streamlit run heart_pred_1.py