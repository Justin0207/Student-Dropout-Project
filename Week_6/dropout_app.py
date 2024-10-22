# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:03:29 2024

@author: Anyanwu Justice

"""


import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import pickle
import base64
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np



class FeatureBinner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the bins and labels
        self.age_bins = [16, 19, 22, 26, 30, 35, 41, 49, 71]
        self.ad_grade_bins = [94.0, 107.2, 116.9, 125.2, 134.1, 144.6, 157.7, 191.0]
        self.prev_grade_bins = [94.0, 112.0, 122.0, 129.0, 136.0, 144.0, 154.0, 166.0, 191.0]
        self.labels_ad = [0, 1, 2, 3, 4, 5, 6]
        self.labels_prev = [0, 1, 2, 3, 4, 5, 6, 7]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Apply the binning using pd.cut
        X['age_bins'] = pd.cut(X['Age at enrollment'], bins=self.age_bins, labels=self.labels_prev).astype('int')
        
        X['Admission grade (bins)'] = pd.cut(X['Admission grade'], bins=self.ad_grade_bins, labels=self.labels_ad).astype('int')
        
        X['Previous grade (bins)'] = pd.cut(X['Previous qualification (grade)'], bins=self.prev_grade_bins, labels=self.labels_prev).astype('int')
        return X

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        X = X.copy()
        
        # Total Units Enrolled
        X['Total Units Enrolled'] = X['Curricular units 1st sem (enrolled)'] + X['Curricular units 2nd sem (enrolled)']
        
        # Total Units Approved
        X['Total Units Approved'] = X['Curricular units 1st sem (approved)'] + X['Curricular units 2nd sem (approved)']
        
        # Average curricular units
        X['Average curricular units'] = (X['Curricular units 1st sem (grade)'] + X['Curricular units 2nd sem (grade)']) / 2
        
        # Approval Rate (Handle division by zero to remove NaN errors)
        X['Approval Rate'] = np.where(X['Total Units Enrolled'] != 0,
                                      X['Total Units Approved'] / X['Total Units Enrolled'],
                                      0)
        
        # Improvement in Grades
        X['Improvement in Grades'] = X['Curricular units 2nd sem (grade)'] - X['Curricular units 1st sem (grade)']
        
        # Economic Hardship
        X['Economic Hardship'] = X['Unemployment rate'] + X['Inflation rate'] - X['GDP']
        
        # Total Units without Evaluations
        X['Total Units without Evaluations'] = X['Curricular units 1st sem (without evaluations)'] + X['Curricular units 2nd sem (without evaluations)']
      
        return X
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = None
        pass

    def fit(self, X, y=None):
        self.columns = X.drop(['Marital status', 'Daytime/evening attendance', 'Application order', 'Nationality', 'Course', 'Application mode',
                               'Previous qualification', "Mother's qualification", "Father's qualification",
                               "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
                               'Gender', 'Scholarship holder', 'International', 'age_bins', 'Admission grade (bins)', 'Previous grade (bins)', 'Target'], axis = 1).columns
        return self

    def transform(self, X):
        X = X.copy()


        # Loop through specified columns
        for col in self.columns:
            # Check skewness of the column
            col_skewness = stats.skew(X[col].dropna())  # Drop NaNs to avoid skew calculation issues

            # Check if skewness is above the threshold and values are non-negative
            if -1 < col_skewness < 1 and (X[col] >= 0).all():
                X[col] = np.log1p(X[col])  # Apply log transformation
                
        return X
    
class PolynomialFeaturesInteraction(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the PolynomialFeatures transformer from sklearn
        self.poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
    def fit(self, X, y=None):
        # Select numerical features to apply polynomial transformation
        self.numerical_features = ['Admission grade', 'Age at enrollment', 'Curricular units 1st sem (credited)',
                                'Curricular units 1st sem (enrolled)',
                                'Curricular units 1st sem (evaluations)',
                                'Curricular units 1st sem (approved)',
                                'Curricular units 1st sem (without evaluations)',
                                'Curricular units 2nd sem (credited)',
                                'Curricular units 2nd sem (enrolled)',
                                'Curricular units 2nd sem (evaluations)',
                                'Curricular units 2nd sem (approved)',
                                'Curricular units 2nd sem (without evaluations)', 'Total Units Enrolled',
                                'Total Units Approved', 'Average curricular units', 'Approval Rate', 'Improvement in Grades',
                                'Economic Hardship', 'Total Units without Evaluations', 'Unemployment rate', 'Inflation rate', 
                                'GDP']
        # Fit the polynomial transformer on the numerical features
        self.poly_transformer.fit(X[self.numerical_features])
        return self
    
    def transform(self, X):
        # Transform the numerical features
        numerical_df = X[self.numerical_features]
        poly_features = self.poly_transformer.transform(numerical_df)
        
        # Create a DataFrame for the polynomial features
        poly_df = pd.DataFrame(poly_features, columns=self.poly_transformer.get_feature_names_out(self.numerical_features))
        
        # Drop the numerical features from the original DataFrame
        cat_df = X.drop(columns=self.numerical_features)
        
        # Concatenate the polynomial features with the categorical data
        result = pd.concat([poly_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
        return result

# Load the feature engineering pipeline from the file
with open(r'Week_6/feature_engineering.pkl', 'rb') as f:
    pipeline_eng = pickle.load(f)
    
with open(r'Week_6/grid_gdb.pkl', 'rb') as f:
    model = pickle.load(f)
    
st.set_page_config(
    page_title="Student Dropout Prediction App",  # Page title
    page_icon="📊",  # Page icon (emoji or file)
    layout="wide",  # Wide layout for the app
    initial_sidebar_state="expanded"  # Keep sidebar expanded by default
)
    
def set_background(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
         }}
         
         </style>
         """,
         unsafe_allow_html=True
     )

set_background(r'C:\Users\HP\Desktop\academic success.jpg')
    
# Helper function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    """Render SHAP plots in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

with st.sidebar:
    selected = option_menu('Menu',
                           ['Dropout Prediction',
                            'Feature Importance'],
                           icons = ['activity', 'kanban'],
                           default_index = 0)



df = pd.read_csv(r'Week_6/data_renamed.csv')
categorical_cols = [
            'Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance',
            'Previous qualification', 'Nationality', "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
            'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
        ]
numerical_cols = [
    'Previous qualification (grade)', 'Admission grade', 'Age at enrollment',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

columns = ['Marital status', 'Application mode', 'Application order', 'Course',
       'Daytime/evening attendance', 'Previous qualification',
       'Previous qualification (grade)', 'Nationality',
       "Mother's qualification", "Father's qualification",
       "Mother's occupation", "Father's occupation", 'Admission grade',
       'Displaced', 'Educational special needs', 'Debtor',
       'Tuition fees up to date', 'Gender', 'Scholarship holder',
       'Age at enrollment', 'International',
       'Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
       'Inflation rate', 'GDP']




# Prediction function
def predict(input_data):
    # Transform the input using the feature engineering pipeline
    transformed_data = pipeline_eng.transform(input_data)
    
    
    # Optionally, get prediction probabilities
    prediction_proba = model.predict_proba(transformed_data)
    
    return prediction_proba
    
if selected == 'Dropout Prediction':
    st.markdown("<h1 style='color: white; '>Student Dropout Prediction</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns((2))

    with st.form(key='prediction_form'):
        st.write("Enter details for prediction:")

        # Temporary user input dictionary
        user_input = {}

        # Collect user inputs
        with col1:
            for col in categorical_cols:
                user_input[col] = st.selectbox(f"{col}", df[col].unique())

        with col2:
            for col in numerical_cols:
                if col == 'Previous qualification (grade)':
                    user_input[col] = st.number_input(f"{col}", min_value=df[col].min(), value=df[col].min())
                elif col == 'Admission grade':
                    user_input[col] = st.number_input(f"{col}", min_value=df[col].min(), value=df[col].min())
                elif col == 'Age at enrollment':
                    user_input[col] = st.number_input(f"{col}", step=1, min_value=df[col].min(), value=df[col].min())
                else:
                    user_input[col] = st.number_input(f"{col}")

        submit_button = st.form_submit_button("Predict")

        if submit_button:
            # Store user input in session state
            st.session_state['user_input'] = user_input

            # Convert input data into a DataFrame
            input_data = pd.DataFrame([user_input], columns=columns)

            # Transform input data and predict
            prediction_proba = predict(input_data)

            # Display the results
            if prediction_proba[0][1] >= 0.5:
                st.markdown(f"<span style='color:red'>**Dropout Risk: {round(prediction_proba[0][1] * 100, 2)}%**</span>", unsafe_allow_html=True)
                st.error('High Dropout Risk')
            else:
                st.markdown(f"<span style='color:green'>**Dropout Risk: {round(prediction_proba[0][1] * 100, 2)}%**</span>", unsafe_allow_html=True)
                st.success('Low Dropout Risk')
                st.balloons()

if selected == 'Feature Importance':
    train = pd.read_csv(r'Week_6/train_dataset.csv')

    # Check if user_input is available in session state
    if 'user_input' in st.session_state:
        user_input = st.session_state['user_input']

        # SHAP Explanation
        input_data = pd.DataFrame([user_input], columns=columns)
        explainer = shap.Explainer(model.named_steps['classifier'], model.named_steps['scaler'].transform(train))
        shap_values = explainer(model.named_steps['scaler'].transform(pipeline_eng.transform(input_data)))

        # SHAP summary plot (global explanation)
        st.subheader("Feature Importance (Bar Plot)")
        fig, ax = plt.subplots(figsize=(2, 4))
        shap.summary_plot(shap_values, model.named_steps['scaler'].transform(pipeline_eng.transform(input_data)), feature_names= train.columns, plot_type='bar', show = False)
        st.pyplot(fig)

        # SHAP force plot (local explanation for individual prediction)
        st.subheader("Prediction Explanation (Force Plot)")
        shap.initjs()
        force_plot = shap.force_plot(shap_values[0].base_values, shap_values[0].values, pipeline_eng.transform(input_data))
        st_shap(force_plot)
    else:
        st.error("No user input available for SHAP explanation. Please make a prediction first.")



