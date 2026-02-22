import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'artifacts/model_data.joblib'
model_sv=joblib.load(MODEL_PATH)
model=model_sv['model']
features=model_sv['features']
scaler=model_sv['scaler']
cols_to_scale=model_sv['cols_to_scale']



def prepare_df(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type):
    
    input_dict={
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquent_to_loan': delinquency_ratio,
        'avg_dpd_to_deliquency': avg_dpd_per_delinquency,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        # additional features for scaling purpose
        'number_of_dependants': 1,  # Dummy value
        'years_at_current_address': 1,  # Dummy value
        'zipcode': 1,  # Dummy value
        'sanction_amount': 1,  # Dummy value
        'processing_fee': 1,  # Dummy value
        'gst': 1,  # Dummy value
        'net_disbursement': 1,  # Computed dummy value
        'principal_outstanding': 1,  # Dummy value
        'bank_balance_at_application': 1,  # Dummy value
        'number_of_closed_accounts': 1,  # Dummy value
        'enquiry_count': 1  # Dummy value
    }
    
    #create a DataFrame
    df=pd.DataFrame([input_dict])

    #scaler
    scaled = scaler.transform(df[cols_to_scale])
    # Clip to [0, 1] to prevent out-of-range extrapolation on the MinMaxScaler
    import numpy as np
    scaled = np.clip(scaled, 0, 1)
    df[cols_to_scale] = scaled
    
    df=df[features]

    return df

def calculate_credit_score(input_df,base_score=300,scale_length=600):
    x=np.dot(input_df.values,model.coef_.T)+model.intercept_

    default_probability=1/(1+np.exp(-x))
    non_default_probability=1-default_probability
      
    credit_score=base_score+(non_default_probability.flatten()[0])*scale_length
     
    if 300 <= credit_score < 500:
        rating = "Poor"
    elif 500 <= credit_score < 650:
        rating = "Average"
    elif 650<= credit_score < 750:
        rating = "Good"
    elif 750 <= credit_score <= 900:
        rating = "Excellent"
    else:
        rating = "Invalid Score"

   
    print(credit_score,rating)

    return default_probability.flatten()[0],int(credit_score),rating


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type):
    
    input_df=prepare_df(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type) 
    
    probabilty,credit_score,rating=calculate_credit_score(input_df)

    print(probabilty,credit_score,rating)

    return probabilty,credit_score,rating