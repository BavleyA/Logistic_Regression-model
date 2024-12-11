import joblib
import pandas as pd
import numpy as np
import streamlit as st

Model = joblib.load(r"E:\FCI\4th year\1\Machine Learning\Machine Learning\Machine Projects\logistic regression with grid search\task 2\logistic.pkl")
Inputs = joblib.load(r"E:\FCI\4th year\1\Machine Learning\Machine Learning\Machine Projects\logistic regression with grid search\task 2\Inputs.pkl")

def prediction(person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file):
    df = pd.DataFrame(columns=Inputs)
    df.at[0,"person_age"] = person_age
    df.at[0,"person_gender"] = person_gender
    df.at[0,"person_education"] = person_education
    df.at[0,"person_income"] = person_income
    df.at[0,"person_emp_exp"] = person_emp_exp
    df.at[0,"person_home_ownership"] = person_home_ownership
    df.at[0,"loan_amnt"] = loan_amnt
    df.at[0,"loan_intent"] = loan_intent
    df.at[0,"loan_int_rate"] = loan_int_rate
    df.at[0,"loan_percent_income"] = loan_percent_income
    df.at[0,"cb_person_cred_hist_length"] = cb_person_cred_hist_length
    df.at[0,"credit_score"] = credit_score
    df.at[0,"previous_loan_defaults_on_file"] = previous_loan_defaults_on_file
    result = Model.predict(df)[0]
    return result

def Main():
    st.title("Give a Person a Loan or Not")
    person_age = st.slider("person_age",min_value=20.0 , max_value=60.0 , step=1.0,value = 20.0)
    person_gender = st.selectbox("person_gender",['male', 'female'])
    person_education = st.selectbox("person_education",['Bachelor', 'Masters' , 'High School' , 'Associate'])
    person_income = st.slider("person_income",min_value=8000.5 , max_value=170000.0 , step=1.0,value = 8000.5)
    person_emp_exp = st.slider("person_emp_exp",min_value=0.0 , max_value=30.0 , step=1.0,value = 0.0)
    person_home_ownership = st.selectbox("person_home_ownership",['RENT','OWN','MORTGAGE'])
    loan_amnt = st.slider("loan_amount",min_value=500.0 , max_value=35000.0 , step=1.0,value = 500.0)
    loan_intent = st.selectbox("loan_intent",[ 'DEBTCONSOLIDATION',  'EDUCATION',  'HOMEIMPROVEMENT' ,  'MEDICAL',  'PERSONAL',  'VENTURE'])
    loan_int_rate = st.slider("loan_int_rate",min_value=5.0 , max_value=20.0 , step=0.1,value = 5.0)
    loan_percent_income = st.slider("loan_percent_income",min_value=0.0 , max_value=1.0 , step=0.01,value = 0.0)
    cb_person_cred_hist_length = st.slider("cb_person_cred_hist_length",min_value=1.0 , max_value=30.0 , step=1.0,value = 1.0)
    credit_score = st.slider("credit_score",min_value=350.0 , max_value=900.0 , step=1.0,value = 350.0)
    previous_loan_defaults_on_file = st.selectbox("previous_loan_defaults_on_file",['Yes','No'])

    
    if st.button("Predict"):
        result = prediction(person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file)
        list_result = ["NO" , "Yes"]
        st.text(f"Loan Status for this person is {list_result[result]}")
Main()