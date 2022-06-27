from json import load
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

@st.cache()
def read():
    df = pd.read_csv('emp_attrition.csv')
    education_dict = {1: "Below College", 2:"College", 3: "Bachelor", 4:"Master", 5:"Doctor" }
    same = {1: "Low", 2:"Medium", 3: "High", 4:"Very High"}
    work = {1: "Bad", 2:"Good", 3: "Better", 4:"Best"}

    df['Education'] = df['Education'].map(education_dict)
    df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].map(same)
    df['JobInvolvement'] = df['JobInvolvement'].map(same)
    df['JobSatisfaction'] = df['JobSatisfaction'].map(same)
    df['PerformanceRating'] = df['PerformanceRating'].map(same)
    df['RelationshipSatisfaction'] = df['RelationshipSatisfaction'].map(same)
    df['WorkLifeBalance'] = df['WorkLifeBalance'].map(work)

    return df
data = read()