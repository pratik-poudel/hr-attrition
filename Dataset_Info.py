import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd


st.markdown("""
# HR Metrics
## Efficiency Metrics
    Number of open reqs	
    Positions filled/month	
    Time to fill positions : time in days (Recruiting)
    Salary associated with positions 
    Cost to hire new resource : Salary + admin costs
## Effectiveness Metrics
    Performance ratings at 90 days & 365 days or whatever the company wishes to choose : 
    Assessment results	: competency assessment
    Speed to competency	: Time in days for new hire to demonstrate competency (Hire's Manager)
    Exit Survey Results : Numeric value ranging from 1 to 5 (Survey set by HR for employees leaving the company)
## Outcomes
    Engagement survey results : Numeric value ranging from 1 to 5
    Productivity: hours of work performed for clients as percentage of total time worked
    Retention/turnover at 90 & 365 days	: Binary 
    Profitability : estimated profitability per person ( Determined by company)
""")

st.markdown("""

# General Metrics in HR Analytics
## Compensation 
    Compa-ratio : percent difference between actual salary and salary midpoint of the pay range
    Labor Cost per Full-Time Equivalent (FTE)
## HR Function Efficiency
    HR Cost per FTE (Full Time Employee)
    HR Expense Percent
    HR FTE Ratio
## Productivity
    Revenue per FTE 
    Human Capital ROI 
    Absenteeism Rate 
    Overtime per individual(Based on Head Count)
## Retention
    Turnover (Voluntary or Involuntary)
    Cost of Voluntary Turnover 
    First Year Turnover Rate 
    Greviances 
    Greivances time to first contact

## Talent Acquisition
    Cost per External Hire
    External Time to Fill
    External Hire Rate (Based on Department)
    Resignation Rate (Based on Year ,months)
## Workforce Demographics
    Promotion Rate
    Diversity Percentage (Age Group, Gender, Race )
""" )
st.markdown("# & Many More can be developed." )
