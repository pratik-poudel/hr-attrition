from attr import attr
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from utils import load_data
use_color = ['#EF553B','#636EFA']


df = load_data.data

categorical_columns = df.select_dtypes(include='object').columns
categorical_columns = [f for f in categorical_columns if not 'Attrition' in f and  not 'Over18' in f]
numerical_columns = df.select_dtypes(include='int').columns
numerical_columns= [f for f in numerical_columns if not "EmployeeNumber" in f and
not "DailyRate" in f and not "EmployeeCount"  in f and not "MonthlyIncome" in f and
not "MonthlyRate" in f and not "StandardHours" in f]


def stacked_attrition(col, data):
    fig = px.histogram(data, col, color='Attrition',color_discrete_sequence=['#EF553B','#636EFA'],
    title=f"Attrition Grouped by {col}").update_xaxes(categoryorder = 'total descending')
    fig.update_xaxes(type='category')
    agg = data.groupby([col, "Attrition"])['Attrition'].size().unstack(fill_value=0)
    return st.plotly_chart(fig, use_container_width=True) 


def overall_attrition(data):
    att_ = data['Attrition'].value_counts(normalize=True)
    att = att_.to_numpy()[1] * 100
    att = np.round(att, 2)
    return att


def attrition_multi(col1, col2, data):
    grouped = data.groupby([col1, col2])['Attrition'].size().unstack(fill_value=0)
    grouped['total'] = grouped.sum(axis=1)
    grouped['attrition'] = np.round((grouped['Yes'] / grouped['total']) * 100, 2)
    fig = px.bar(grouped, y=['No', "Yes"], barmode='group')
    fig.add_traces(go.Scatter(x= grouped.index,y=grouped['attrition'], name="Attr %" ,
    mode = 'lines+markers', marker=dict(color='darkmagenta',size=10)))
    fig.add_hline(y=attrition, line_width=3, line_dash="dash", line_color="black")    
    return st.plotly_chart(fig, use_container_width=True)

def attrition_multi_data(col1, col2, data):
    grouped = data.groupby([col1, col2])['Attrition'].size().unstack(fill_value=0)
    grouped['total'] = grouped.sum(axis=1)
    grouped['attrition'] = np.round((grouped['Yes'] / grouped['total']) * 100, 2)
    
    return grouped


attrition = overall_attrition(df)

gen_att = attrition_multi_data("Gender", "Attrition", df)
gen_att.reset_index(inplace=True)
gen_att.columns = gen_att.columns.str.title()

att_by_dept = attrition_multi_data("Department", "Attrition", df).reset_index()
att_by_dept.columns = att_by_dept.columns.str.title()


# st.markdown("## KPI First Row")
kpi1, kpi2, kpi3 = st.columns(3)


with kpi1:
    st.markdown("###  Total Attrition") 
    st.markdown(f"<h1 style='text-align: center; color: red;'>{attrition} %</h1>", unsafe_allow_html=True)
    # st.dataframe(gen_att[["Gender", "Attrition"]].round(2).astype(str).style.highlight_max(axis=0, color="#e88e9b") )
    # st.markdown("---")


with kpi2:
    st.markdown("### Attrition By Gender")
    # st.markdown(f"<h1 style='text-align: center; color: red;'>{11} %</h1>", unsafe_allow_html=True)
    st.dataframe(gen_att[["Gender", "Attrition"]].round(2).astype(str).style.highlight_max(axis=0, color="#e88e9b") )

with kpi3:
    st.markdown("### Attrition by Department")
    st.dataframe(att_by_dept[["Department", "Attrition"]].round(2).astype(str).style.highlight_max(axis=0, color="#e88e9b") )



with st.container():
    c1, c2, c3  = st.columns([1, 1, 1])
    with c1:
        fig = px.histogram(df, x='Department',y='MonthlyIncome', color='Attrition',histfunc='avg', 
        color_discrete_sequence=use_color, title="""
        Average Monthly Income of Employees <br> Leaving the Company By Department""", )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, nbins=20, x='MonthlyIncome',y='Age', color='Attrition',histfunc='avg', 
        color_discrete_sequence=use_color, title="""
        Average Age Employees Leaving the Company <br> By Monthly Income""",)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.histogram(df, x='Education',y='Age', color='Attrition',histfunc='avg', 
        color_discrete_sequence=use_color, title="""
        Average Age  of Employees Leaving the Company <br> By Education""",)
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Data Distribution in Categorical Variables")
option = st.selectbox('Select Columns', categorical_columns)
c1, c2 = st.columns([2, 1])
with c1:
    fig = px.pie(df, option)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.dataframe(df[option].value_counts())

st.subheader('Categorical Columns Visualization')
option = st.selectbox("Select the columns you want to visualize:", categorical_columns)
c1, c2 = st.columns([1, 2])
with c1:
    stacked_attrition(option, df)
with c2:
    attrition_multi(option, 'Attrition', df)


with st.container():
    st.subheader('Numerical Columns Visualization')
    option = st.selectbox("Select the columns you want to visualize:", numerical_columns)
    c1, c2 = st.columns([1, 2])
    with c1:
        stacked_attrition(option, df)
        
    with c2:
        attrition_multi(option, 'Attrition', df)

df.to_csv('attrition_final.csv', index=False)
