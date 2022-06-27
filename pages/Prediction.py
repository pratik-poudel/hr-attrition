import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report


use_color = ['#EF553B','#636EFA']

def load_data_final():
    df = pd.read_csv('attrition_final.csv')
    return df

df = load_data_final()
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)
df["Attrition"] = df['Attrition'].map({"Yes":1, "No":0})

categorical_cols = ['BusinessTravel', 'Department', 'Education', 'EducationField',
'Gender', 'JobRole', 'MaritalStatus','OverTime']
ordinal_cols = ['JobInvolvement','JobLevel','JobSatisfaction','PerformanceRating','RelationshipSatisfaction',
'WorkLifeBalance', 'EnvironmentSatisfaction']
numerical_cols = ['Age', 
# 'DailyRate',
'DistanceFromHome', 
# 'HourlyRate',
'MonthlyIncome', 
# 'MonthlyRate',
 'NumCompaniesWorked','StockOptionLevel','TotalWorkingYears', 
'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager','PercentSalaryHike']

ohe = OneHotEncoder(drop_last=False)
ordinal_enc = OrdinalEncoder()

cat_data = ohe.fit_transform(df[categorical_cols])
ord_data = ordinal_enc.fit_transform(df[ordinal_cols], df['Attrition'])

scaler = StandardScaler()
num_data = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

data = pd.concat([cat_data, ord_data, num_data, df['Attrition']], axis=1)


X = data.drop('Attrition', axis=1)
y = data.Attrition

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
                                                    stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, 
random_state=2024, stratify=y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=2024)
model.fit(X_train, y_train)
preds= model.predict(X_val)
# st.write(roc_auc_score(y_val, preds))
# st.dataframe(pd.DataFrame(classification_report(y_val, preds, output_dict=True)))
# st.dataframe(confusion_matrix(y_val, preds))

predictions = model.predict(X_test)
# st.write(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)))
# st.dataframe(confusion_matrix(y_test, predictions))

imp_df = pd.DataFrame(model.coef_[0], index=X_train.columns, columns=['Score'])
imp_df.sort_values(by='Score',ascending=False, inplace=True)
st.markdown("## Feature Importance (Related To Attrition)")
fig = px.bar(imp_df.head(20), x = 'Score')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig)


def user_input_features():
    with st.form(key='my_form'):
        col1, col2 , col3 = st.columns(3)
        with col1:
            BusinessTravel = st.selectbox('BusinessTravel', df['BusinessTravel'].unique())
            Department = st.selectbox('Department', df['Department'].unique())
            Education = st.selectbox('Education', df['Education'].unique())
            EducationField = st.selectbox('EducationField', df['EducationField'].unique())
            EnvironmentSatisfaction = st.selectbox('EnvironmentSatisfaction', df['EnvironmentSatisfaction'].unique())
            Gender = st.selectbox('Gender', df['Gender'].unique())
            JobInvolvement = st.selectbox('JobInvolvement', df['JobInvolvement'].unique())
            JobRole = st.selectbox('JobRole', df['JobRole'].unique())
            JobSatisfaction = st.selectbox('JobSatisfaction', df['JobSatisfaction'].unique())
            
            
        with col2:
            MaritalStatus = st.selectbox('MaritalStatus', df['MaritalStatus'].unique())
            OverTime = st.selectbox('OverTime', df['OverTime'].unique())
            PerformanceRating = st.selectbox('PerformanceRating', df['PerformanceRating'].unique())
            RelationshipSatisfaction = st.selectbox('RelationshipSatisfaction', df['RelationshipSatisfaction'].unique())
            WorkLifeBalance = st.selectbox('WorkLifeBalance', df['WorkLifeBalance'].unique())
            Age = st.number_input('Age', min_value= 0, max_value=80, value=18)
            # DailyRate = st.number_input('DailyRate', min_value= 0, max_value=80, value=20)
            DistanceFromHome = st.number_input('DistanceFromHome', min_value= 0, max_value=500, value=10)
            # HourlyRate = st.number_input('HourlyRate', min_value= 0, max_value=100000, value=50)
            JobLevel = st.selectbox('JobLevel', df['StockOptionLevel'].unique())
            MonthlyIncome = st.number_input('MonthlyIncome', min_value= 0, max_value=500000, value=4500)
            # MonthlyRate = st.number_input('MonthlyRate', min_value= 0, max_value=1000000, value=5000)
            
            
        with col3:
            TotalWorkingYears = st.number_input('TotalWorkingYears', min_value= 0, max_value=80, value=1)
            TrainingTimesLastYear = st.number_input('TrainingTimesLastYear', min_value= 0, max_value=80, value=1), 
            YearsAtCompany = st.number_input('YearsAtCompany', min_value= 0, max_value=80, value=1)
            YearsInCurrentRole = st.number_input('YearsInCurrentRole', min_value= 0, max_value=80, value=1)
            YearsSinceLastPromotion = st.number_input('YearsSinceLastPromotion', min_value= 0, max_value=80, value=1)
            YearsWithCurrManager = st.number_input('YearsWithCurrManager', min_value= 0, max_value=80, value=1)
            NumCompaniesWorked = st.number_input('NumCompaniesWorked', min_value= 0, max_value=80, value=1)
            PercentSalaryHike = st.number_input('PercentSalaryHike', min_value= 0, max_value=80, value=10)
            StockOptionLevel = st.selectbox('StockOptionLevel', df['StockOptionLevel'].unique())
            data = {
                'BusinessTravel': BusinessTravel ,
                'Department': Department ,
                'Education': Education ,
                'EducationField': EducationField ,
                'EnvironmentSatisfaction': EnvironmentSatisfaction ,
                'Gender': Gender ,
                'JobInvolvement': JobInvolvement ,
                'JobRole': JobRole ,
                'JobSatisfaction': JobSatisfaction ,
                'MaritalStatus': MaritalStatus ,
                'OverTime': OverTime ,
                'PerformanceRating': PerformanceRating ,
                'RelationshipSatisfaction': RelationshipSatisfaction ,
                'WorkLifeBalance': WorkLifeBalance ,
                'Age': Age ,
                # 'DailyRate': DailyRate ,
                'DistanceFromHome': DistanceFromHome ,
                # 'HourlyRate': HourlyRate ,
                'JobLevel': JobLevel ,
                'MonthlyIncome': MonthlyIncome ,
                # 'MonthlyRate': MonthlyRate ,
                'NumCompaniesWorked': NumCompaniesWorked ,
                'PercentSalaryHike': PercentSalaryHike ,
                'StockOptionLevel': StockOptionLevel ,
                'TotalWorkingYears': TotalWorkingYears ,
                'TrainingTimesLastYear': TrainingTimesLastYear ,
                'YearsAtCompany': YearsAtCompany ,
                'YearsInCurrentRole': YearsInCurrentRole ,
                'YearsSinceLastPromotion': YearsSinceLastPromotion ,
                'YearsWithCurrManager': YearsWithCurrManager ,
            }

        features = pd.DataFrame(data, index=[0])
        submit_button = st.form_submit_button(label='Submit to get predictions')
        

    return features


inp_df = user_input_features()
st.markdown('### User Input parameters')
st.dataframe(inp_df)
st.write('---')


cat_data = ohe.transform(inp_df[categorical_cols])
ord_data = ordinal_enc.transform(inp_df[ordinal_cols])

num_data = pd.DataFrame(scaler.transform(inp_df[numerical_cols]), columns=numerical_cols)

data = pd.concat([cat_data, ord_data, num_data], axis=1)

prediction = model.predict_proba(data)[:, 1][0]
prediction = prediction * 100

st.markdown("## Attrition %")
st.markdown(f"<h1 style='text-align: center; color: red;'>{np.round(prediction, 2)} %</h1>", unsafe_allow_html=True)
