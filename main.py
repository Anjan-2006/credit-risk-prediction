import streamlit as st
from prediction_helper import predict

st.title("Credit Risk Predictor")


col1,col2,col3=st.columns(3)

with col1:
    age=st.number_input("Age",18,100,step=1)
with col2:
    income=st.number_input("Income",min_value=0,value=500000,step=1000)
with col3:
    loan_amount=st.number_input("Loan Amount",min_value=0,value=1000000,step=1000)


loan_to_income_ratio = loan_amount / income if income > 0 else 0
col4,col5,col6=st.columns(3)

with col4:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")
with col5:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with col6:
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, max_value=10, value=5)

col7,col8,col9=st.columns(3)

with col7:
    delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=30)

with col8:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)

with col9:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)

col10,col11,col12=st.columns(3)

with col10:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with col11:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with col12:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])



if st.button("Calculate Risk"):
    probability,credit_score,rating=predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    
    