from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
from MBS_prepayrisk import  LabelEncoderTransformer

app = Flask(__name__)

# Load the saved pipeline
pipeline = joblib.load('MBS_combined_pipeline.pkl')
pipeline_risk = joblib.load('MBS_prepayrisk_pipeline.pkl')
model= joblib.load('MBS_affordability_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/market')
def market():
    return render_template('market.html')

@app.route('/predict', methods=["POST","GET"])
def predict():

    prediction_made = False 

    if request.method == "POST":
        # Process the form data for prediction
        data = {key: request.form[key] for key in request.form}
        df = pd.DataFrame([data])

        # Extract year from date fields
        if 'FirstPaymentDate' in df.columns:
            df['FirstPaymentYear'] = pd.to_datetime(df['FirstPaymentDate']).dt.year
        else:
            df['FirstPaymentYear'] = None

        if 'MaturityPaymentDate' in df.columns:
            df['MaturityYear'] = pd.to_datetime(df['MaturityPaymentDate']).dt.year
        else:
            df['MaturityYear'] = None

        # Drop original date columns
        df.drop(columns=['FirstPaymentDate', 'MaturityPaymentDate'], errors='ignore', inplace=True)

        # Handle CreditRange and RePayRange
        if 'CreditRange' in df.columns:
            df['CreditRange'] = df['CreditRange'].astype(int)
        if 'RePayRange' in df.columns:
            df['RePayRange'] = df['RePayRange'].astype(int)

        # Handle Occupancy
        if 'Occupancy' in df.columns:
            df['Occupancy_O'] = df['Occupancy'].apply(lambda x: 1 if x == 'O' else 0)
        else:
            df['Occupancy_O'] = 0  # Default value if not provided

        # Handle Loan Purpose
        if 'LoanPurpose' in df.columns:
            df['LoanPurpose_N'] = df['LoanPurpose'].apply(lambda x: 1 if x == 'N' else 0)
        else:
            df['LoanPurpose_N'] = 0  # Default value if not provided

        # Drop columns that are not needed
        df.drop(columns=['Occupancy', 'LoanPurpose'], errors='ignore', inplace=True)

        try:
            # Make predictions
            y_class_pred, y_reg_pred = pipeline.predict(df)
            
            # Prepare responses based on predictions
            y_class_pred_val = int(y_class_pred[0])
            y_reg_pred_val =  round(float(y_reg_pred[0]), 3)  if not np.isnan(y_reg_pred[0]) else 'NaN'
            prediction_made = True
            return render_template('result.html', y_class_pred=y_class_pred_val, y_reg_pred=y_reg_pred_val,prediction_made=prediction_made)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template('result.html')


@app.route('/prepayrisk', methods=["POST","GET"])
def prepayrisk():

    if request.method == "POST":
        input_data = {key: request.form[key] for key in request.form}
        input_df = pd.DataFrame([input_data])
        
        # Ensure the custom transformer is available and properly set up
        try:
            # Make prediction
            prediction = pipeline_risk.predict(input_df)
            risk_message = "Prepayment risk is high." if prediction[0] == 1 else "Prepayment risk is low."
            return render_template('prepayrisk.html', prediction=risk_message)
           
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return render_template('prepayrisk.html')  


@app.route('/affordability', methods=["POST","GET"])
def affordability():

    if request.method == "POST":
        model_data={key: request.form[key] for key in request.form}
        model_df = pd.DataFrame([model_data])
        
        # Ensure the custom transformer is available and properly set up
        try:
            # Make prediction
            prediction = model.predict(model_df)
            result_model= "Monthly insatllement affordability risk is high." if prediction[0] == 1 else "Monthly insatllement affordability risk is low."
            return render_template('affordability.html', prediction=result_model)
           
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('affordability.html')

if __name__ == '__main__':
    app.run(debug=True)
