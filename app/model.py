import pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import io

class country_class:
	def country_model():
		url="https://api.covid19india.org/csv/latest/case_time_series.csv"
		s=requests.get(url).content
		covid=pd.read_csv(io.StringIO(s.decode('utf-8')))

		covid_data = covid[['Date' , 'Total Confirmed']]
		covid_data['date_format'] = pd.date_range(start='03/1/2020', periods=len(covid_data), freq='D')
		covid_data['date_format'] = pd.to_datetime(covid_data['date_format'], infer_datetime_format=True)

		covid_data = covid_data[['date_format' , 'Total Confirmed']]
		indexedDataset = covid_data.set_index(['date_format'])

		indexedDataset_logScale = np.log(indexedDataset)
		days_predicted = indexedDataset_logScale.shape[0]

		pickle.dump(covid_data['Total Confirmed'], open('past_values.pkl','wb'))
		# pickle.dump(days_predicted, open('days_predicted.pkl','wb'))
		model = ARIMA(indexedDataset_logScale, order=(2,1,0))
		results_AR = model.fit(disp=-1)

		# Saving model to disk
		pickle.dump(results_AR, open('model.pkl','wb'))




