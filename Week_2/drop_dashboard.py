# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:37:17 2024

@author: Anyanwu Justice
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Initialize an empty list to store column names that are converted to categorical
cat_cols = []

# Define a function that checks the number of unique values in a given column of a DataFrame
def check_unique_values(df, column):
    # Get the count of unique values in the specified column
    unique_count = df[column].nunique()
    
    # If the number of unique values is less than or equal to 6
    if unique_count <= 6:
        
        # Convert the column to the 'category' dtype to save memory
        df[column] = df[column].astype('category')
        
        # Append the column name to the 'cat_cols' list for reference
        cat_cols.append(column)


st.set_page_config(page_title = 'Student Performance Dashboard',
                   page_icon = ':chart_with_upwards_trend:', layout = 'wide')

#C:\Users\HP\Desktop\DS\Sales Analysis\superstore_train.csv
st.title(':chart_with_upwards_trend: Student Performance Dashboard')

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',
            unsafe_allow_html= True)

fl = st.file_uploader('Upload a File',
                      type = (['csv', 'txt', 'xlsx', 'xls']))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename)
else:
    df = pd.read_csv(r'data_renamed.csv', delimiter = ';')

# Iterate through all columns in the DataFrame
for cols in df.columns:
    # Apply the check_unique_values function to each column
    check_unique_values(df, cols)

st.sidebar.header('Choose your filters: ')


# Creating filter for stu_cat
stu_cat = st.sidebar.multiselect('Pick a Category',
                                df['Target'].unique())
if not stu_cat:
    df2 = df.copy()
    
else:
    df2 = df[df['Target'].isin(stu_cat)]
    
# Creating filter for national
national = st.sidebar.multiselect('Pick a Nationality',
                                df2.Nacionality.unique())

if not national:
    df3 = df2.copy()
    
else:
    df3 = df2[df2.Nacionality.isin(national)] 
    
    
# Creating filter for course
course = st.sidebar.multiselect('Pick a Course',
                                df2['Course'].unique())

if not stu_cat and not national and not course:
    
    filtered_df = df
    
elif not national and not course:
    
    filtered_df = df[df['Target'].isin(stu_cat)]
    
elif not stu_cat and not course:
    
    filtered_df = df[df.Nacionality.isin(national)]

elif national and course:
    filtered_df = df3[df.Nacionality.isin(national) & df3['Course'].isin(course)]
    
elif stu_cat and course:
    filtered_df = df3[df.Nacionality.isin(stu_cat) & df3['Course'].isin(course)]
    
elif stu_cat and national:
    filtered_df = df3[df['Target'].isin(stu_cat) & df3.Nacionality.isin(national)]
    
elif course:
    filtered_df = df3[df3['Course'].isin(course)]
    
else:
    filtered_df = df3[df3['Target'].isin(stu_cat) & df3.Nacionality.isin(national) & df3['Course'].isin(course)]
    
gender_df = filtered_df.groupby('Gender').agg({'Admission grade': 'mean', 'Previous qualification (grade)': 'mean', 'Curricular units 1st sem (grade)':'mean', 'Curricular units 2nd sem (grade)':'mean', 'GDP': 'mean', 'Unemployment rate': 'mean', 'Inflation rate': 'mean'}).reset_index()

marital_df = filtered_df.groupby('Marital status').agg({'Admission grade': 'mean', 'Previous qualification (grade)': 'mean', 'Curricular units 1st sem (grade)':'mean', 'Curricular units 2nd sem (grade)':'mean', 'GDP': 'mean', 'Unemployment rate': 'mean', 'Inflation rate': 'mean'}).reset_index()    


col1, col2, col3 = st.columns((3))

with col1:

        st.subheader('Gender Analysis')
        gender_metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'GDP', 'Unemployment rate', 'Inflation rate'],
                                  help = 'Click here to select an Select an Analyzing Feature to summarize the dashboard',
                                  key = 'gendermetrics')  
        fig = px.bar(gender_df, x = 'Gender', y = gender_metric, color = 'Gender',
                     #text = ['{:,.0f} units'.format(x) for x in category_df[cat_metric]],
                     color_discrete_sequence = [px.colors.qualitative.Dark2[0], px.colors.qualitative.Set1[1]],
                     template = 'plotly_dark', title = 'Average {} by Gender'.format(gender_metric))
        st.plotly_chart(fig, use_container_width = True, height = 200)


            
###########################################################################################################################
with col2:
    
    st.subheader('Debt Status Analysis')
    
    debt_metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate'],
                              help = 'Click here to select an Select an Analyzing Feature to summarize the dashboard', key = 'regmetric') 
    if debt_metric:
        debt_df = filtered_df.groupby('Debtor')[debt_metric].mean().reset_index(name = debt_metric)
        fig = px.pie(debt_df, values = debt_metric, names = 'Debtor', color_discrete_sequence = px.colors.qualitative.Prism,
                     hole = 0.4, title = 'Average {} by Debtor Status'.format(debt_metric),
                     template = 'plotly_dark',)
        
        fig.update_traces(text = debt_df['Debtor'], textposition = 'outside')
        
        st.plotly_chart(fig, use_container_width = True, height = 200)
    
#####################################################################################################################################################
with col3:

        st.subheader('Marital Status Analysis')
        marital_metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'GDP', 'Unemployment rate', 'Inflation rate'],
                                  help = 'Click here to select an Select an Analyzing Feature to summarize the dashboard',
                                  key = 'mar_metrics')  
        fig = px.bar(marital_df, x = 'Marital status', y = marital_metric, color = 'Marital status',
                     #text = ['{:,.0f} units'.format(x) for x in category_df[cat_metric]],
        color_discrete_sequence = [px.colors.qualitative.Set1[1], px.colors.qualitative.D3[9],
                                     px.colors.qualitative.Prism[0], px.colors.qualitative.Set1[3], px.colors.qualitative.Vivid[3], px.colors.qualitative.Dark2[0]],
                                     template = 'plotly_dark', title = 'Average {} by Marital Status'.format(marital_metric))
        st.plotly_chart(fig, use_container_width = True, height = 200)

  
#######################################################################################################################################################

cols1, cols2, cols3 = st.columns((3))

with cols1:
    
    with st.expander('Gender_DataView'):
        
        st.write(gender_df[['Gender', gender_metric]].style.background_gradient(cmap = 'Greens'))
        
        csv  = gender_df[['Gender', gender_metric]].to_csv(index = False).encode('utf-8')
        
        st.download_button('Download Data', data = csv, file_name = 'Gender.csv', mime = 'text/csv',
                           
                           help = 'Click here to download as a CSV file')
        
    
with cols2:
    
    with st.expander('Debt_DataView'):
        
        #debt_df = filtered_df.groupby('Debtor')[debt_metric].mean().reset_index(name = reg_metric)
        
        st.write(debt_df.style.background_gradient(cmap = 'Blues'))
        
        csv  = debt_df.to_csv(index = False).encode('utf-8')
        
        st.download_button('Download Data', data = csv, file_name = 'debt.csv', mime = 'text/csv',
                           
                           help = 'Click here to download as a CSV file')  
        
with cols3:
    
    with st.expander('Marital_Status_DataView'):
        
        st.write(marital_df[['Marital status', marital_metric]].style.background_gradient(cmap = 'Oranges'))
        
        csv  = marital_df[['Marital status', marital_metric]].to_csv(index = False).encode('utf-8')
        
        st.download_button('Download Data', data = csv, file_name = 'marital.csv', mime = 'text/csv',
                           
                           help = 'Click here to download as a CSV file')
        
#######################################################################################################################################
figure = px.scatter(filtered_df, x = 'Curricular units 1st sem (grade)', y = 'Curricular units 2nd sem (grade)', size = 'Age at enrollment')

figure.update_layout(title = 'Relationship Between First semester grades and Second semester grades', titlefont = dict(size = 25))

st.plotly_chart(figure, use_container_width = True, height = 200)

######################################################################################################################################################


int_df = filtered_df.groupby('International').agg({'Admission grade': 'mean', 'Previous qualification (grade)': 'mean', 'Curricular units 1st sem (grade)':'mean', 'Curricular units 2nd sem (grade)':'mean', 'GDP': 'mean', 'Unemployment rate': 'mean', 'Inflation rate': 'mean'}).reset_index()


col_1, col_2, col_3 = st.columns((3))

with col_1:

        st.subheader('Tuition Fees Analysis')
        tuition_metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate'],
                                  help = 'Click here to select a grade metric to summarize the dashboard',
                                  key = 'tuitionmetrics')  
        if tuition_metric:
            tuition_df = filtered_df.groupby('Tuition fees up to date')[tuition_metric].mean().reset_index(name = tuition_metric)
            fig = px.pie(tuition_df, values = tuition_metric, names = 'Tuition fees up to date', color_discrete_sequence = [px.colors.qualitative.Vivid[3], px.colors.qualitative.D3[9]],
                         hole = 0.4, title = 'Average {} by Fees Payment Status'.format(tuition_metric),
                         template = 'plotly_dark')
            # text = ['{:,.0f} units'.format(x) for x in tuition_df[tuition_metric]],
            
            fig.update_traces(text = tuition_df['Tuition fees up to date'], textposition = 'outside')
            
            st.plotly_chart(fig, use_container_width = True, height = 200)
            
###########################################################################################################################
with col_2:

        st.subheader('International Status Analysis')
        int_metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'GDP', 'Unemployment rate', 'Inflation rate'],
                                  help = 'Click here to select a grade metric to summarize the dashboard',
                                  key = 'int_metrics')  
        fig = px.bar(int_df, x = 'International', y = int_metric, color = 'International',
                     
                     color_discrete_sequence = [px.colors.qualitative.Dark2[0], px.colors.qualitative.Set1[3]],
                     template = 'plotly_dark', title = 'Average {} by International Status'.format(marital_metric))
        st.plotly_chart(fig, use_container_width = True, height = 200)


#####################################################################################################################################################
with col_3:

        st.subheader('Scholarship Status Analysis')
        sch_metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate'],
                                  help = 'Click here to select a grade metric to summarize the dashboard',
                                  key = 'schmetrics')  
        if sch_metric:
            sch_df = filtered_df.groupby('Scholarship holder')[sch_metric].mean().reset_index(name = sch_metric)
            fig = px.pie(sch_df, values = sch_metric, names = 'Scholarship holder', color_discrete_sequence = [px.colors.qualitative.Prism[0], px.colors.qualitative.Dark2[0]],
                         hole = 0.4, title = 'Average {} by Scholarship Status'.format(sch_metric),
                         template = 'plotly_dark')
            
            fig.update_traces(text = sch_df['Scholarship holder'], textposition = 'outside')
            
            st.plotly_chart(fig, use_container_width = True, height = 200)  
#######################################################################################################################################################
col_1, col_2, col_3 = st.columns((3))

with col_1:
    
    with st.expander('Tuition_Status_DataView'):
        
        st.write(tuition_df[['Tuition fees up to date', tuition_metric]].style.background_gradient(cmap = 'Greens'))
        
        csv  = tuition_df[['Tuition fees up to date', tuition_metric]].to_csv(index = False).encode('utf-8')
        
        st.download_button('Download Data', data = csv, file_name = 'Tuition.csv', mime = 'text/csv',
                           
                           help = 'Click here to download as a CSV file')
        
############################################################################################################################################################

with col_2:
    
    with st.expander('International_Status_DataView'):
        
        st.write(int_df[['International', int_metric]].style.background_gradient(cmap = 'Blues'))
        
        csv  = int_df[['International', int_metric]].to_csv(index = False).encode('utf-8')
        
        st.download_button('Download Data', data = csv, file_name = 'International.csv', mime = 'text/csv',
                           
                           help = 'Click here to download as a CSV file')
        
############################################################################################################################################################

with col_3:
    
    with st.expander('Scholarship_Status_DataView'):
        
        st.write(sch_df[['Scholarship holder', sch_metric]].style.background_gradient(cmap = 'Blues'))
        
        csv  = sch_df[['Scholarship holder', sch_metric]].to_csv(index = False).encode('utf-8')
        
        st.download_button('Download Data', data = csv, file_name = 'Scholarship.csv', mime = 'text/csv',
                           
                           help = 'Click here to download as a CSV file')
        
#############################################################################################################################################################
st.subheader(':point_right: Parents Analysis')

cols1, cols2= st.columns((2))
with cols1:
    qualif_metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'GDP', 'Unemployment rate', 'Inflation rate'],
                              help = 'Click here to select a grade metric to summarize the dashboard',
                              key = 'qualif_metrics') 
    summ_metric = st.selectbox('Summarizing Metric: ', ['mean', 'count'], help = 'Click here to select a summarizing metric', key = 'summ_metric')
    sub_cat = pd.pivot_table(data= filtered_df, index = "Mother's qualification",
                             
                             columns = "Father's qualification", values = qualif_metric, aggfunc = summ_metric).fillna(0)
    st.write(sub_cat.style.background_gradient(cmap = 'Blues'))
    
with cols2:
    occupat_metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'GDP', 'Unemployment rate', 'Inflation rate'],
                              help = 'Click here to select a grade metric to summarize the dashboard',
                              key = 'occupa_metrics') 
    summ1_metric = st.selectbox('Summarizing Metric: ', ['mean', 'count'], help = 'Click here to select a summarizing metric', key = 'summ1_metric')
    sub_cat = pd.pivot_table(data= filtered_df, index = "Mother's occupation",
                             
                             columns = "Father's occupation", values = occupat_metric, aggfunc = summ1_metric).fillna(0)
    st.write(sub_cat.style.background_gradient(cmap = 'Greens'))
    
###################################################################################################################################################################

st.subheader(':point_right: Enrolled Courses Analysis')

tab1, tab2 = st.columns((2))

with tab1:
    
    metric2 = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'GDP', 'Unemployment rate', 'Inflation rate'],
                              help = 'Click here to select an Select an Analyzing Feature to summarize the dashboard',
                              key = 'metrics2')
    
    linechart = filtered_df.groupby('Curricular units 1st sem (enrolled)')[metric2].mean().reset_index()
    
    fig1 = px.line(linechart, x = 'Curricular units 1st sem (enrolled)', y = metric2, height= 500, width=1000, template= 'gridon')
    
    st.plotly_chart(fig1, use_container_width = True)
            
###########################################################################################################################



with tab2:
    
    metric = st.selectbox('Select an Analyzing Feature: ', ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'GDP', 'Unemployment rate', 'Inflation rate'],
                              help = 'Click here to select an Select an Analyzing Feature to summarize the dashboard',
                              key = 'metrics')
    
    linechart = filtered_df.groupby('Curricular units 2nd sem (enrolled)')[metric].mean().reset_index()
    
    fig2 = px.line(linechart, x = 'Curricular units 2nd sem (enrolled)', y = metric, labels= {'Sales': 'Amount'}, height= 500, width=1000, template= 'gridon')
    
    st.plotly_chart(fig2, use_container_width = True)

  
