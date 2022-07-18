
import pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

import io

class state_class:
	def state_model():
		url="https://api.covid19india.org/csv/latest/state_wise_daily.csv"
		s=requests.get(url).content
		covid=pd.read_csv(io.StringIO(s.decode('utf-8')))
		covid = covid.fillna(0)

		pickle.dump(covid, open('original_state_records.pkl','wb'))



		covid_data  =  covid.loc[covid['Status'] == "Confirmed"]
		pickle.dump(covid_data, open('confirmed_state_records.pkl','wb'))

		covid_data = covid_data.drop(['Status'], axis=1)
		covid_data['date_format'] = pd.date_range(start='03/14/2020', periods=len(covid_data), freq='D')
		covid_data['date_format'] = pd.to_datetime(covid_data['date_format'], infer_datetime_format=True)
		covid_data = covid_data.drop(['Date'], axis=1)

		indexedDataset = covid_data.set_index(['date_format'])

		def compute(value_log):
		    value_log += 100
		    indexedDataset_logScale = np.log(value_log)
		    return indexedDataset_logScale	

		indexedDataset_log = indexedDataset.apply(compute)
		pickle.dump(indexedDataset_log, open('past_values_states.pkl','wb'))


		days_predicted = indexedDataset_log.shape[0]
		pickle.dump(days_predicted, open('days_predicted_states.pkl','wb'))

