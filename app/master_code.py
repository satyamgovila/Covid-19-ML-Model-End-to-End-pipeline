import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import io
import requests
import pickle
from apscheduler.schedulers.background import BackgroundScheduler
 
import model #model.py
import model_state #model_state.py
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template,send_file


def task_1():
	model.country_class.country_model()


def task_2():
	model_state.state_class.state_model()


app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "COVID PREDICTION ML API"


@app.route('/predict_state')
def predict_state():
    #http://0.0.0.0:5000/predict_state?state=TN&days=6
    past_values_states = pickle.load(open('past_values_states.pkl','rb'))
    conf_state_rec_model = pickle.load(open('confirmed_state_records.pkl' , 'rb'))

    forecast_list_state = []
    forecast_list_state.clear()

    input_state = request.args['state']
    input_days = request.args['days']
    input_days_state = int(input_days)
    model_state = ARIMA(past_values_states[input_state], order=(2,1,0))
    results_AR_state = model_state.fit(disp=-1)

    r1_state = results_AR_state.forecast(steps=input_days_state)
    converted_results_state = [(np.exp(x)) for x in [i for i in r1_state]]
    final_predict_state = list(converted_results_state[0])
    final_predict_state=[int(i)-100 for i in final_predict_state]

    for i in conf_state_rec_model[input_state][-input_days_state:]:
        forecast_list_state.append(i)
    forecast_list_state.extend(final_predict_state)


    output_state_data = {
          "past_future_values" : forecast_list_state,
          "predicted_future_values" : final_predict_state
             }

    return output_state_data



@app.route('/predict_country',methods=['GET'])
#http://0.0.0.0:5000/predict_country?days=6
def predict_json():
    model = pickle.load(open('model.pkl','rb'))
    model3 = pickle.load(open('past_values.pkl','rb'))

    forecast_list = []
    forecast_list.clear()
    days = request.args['days']
    days_int = int(days)

    r1 = model.forecast(steps=days_int)
    converted_results = [(np.exp(x)) for x in [i for i in r1]]
    final_predict = list(converted_results[0])
    predicted_future_values = [int(i) for i in final_predict]
    forecast_list = []
    for i in model3[-days_int:]:
        forecast_list.append(i)
    forecast_list.extend(predicted_future_values)
    output = {
          "past_future_values" : forecast_list,
          "predicted_future_values" : predicted_future_values
             }
            
    return output


@app.route('/predict_case',methods=['GET'])
#http://0.0.0.0:5000/predict_case?state=WB

def predict_state_json():
    model4 = pickle.load(open('original_state_records.pkl' , 'rb'))
    input_state = request.args['state']
    covid_state = model4[['Status' , input_state]]

    case_output = {
        "death_cases" : covid_state.loc[model4['Status'] == "Deceased"].sum()[1],
        "recovered_cases" : covid_state.loc[model4['Status'] == "Recovered"].sum()[1] ,
        "confirmed_cases" : covid_state.loc[model4['Status'] == "Confirmed"].sum()[1]
    } 
    return case_output



if __name__ == "__main__":
    scheduler= BackgroundScheduler()
    d1 = timedelta(seconds=5)
    d2 = timedelta(seconds=30)
    scheduler.add_job(task_1, 'interval', hours=8,start_date=datetime.now()+d1)
    scheduler.add_job(task_2, 'interval', hours=7,start_date=datetime.now()+d2)
    scheduler.start()
    app.run(debug=True, host='0.0.0.0',port=80)
