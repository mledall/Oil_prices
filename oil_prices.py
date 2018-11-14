'''
Scotiabank data challenge:
What will the price of oil be on Friday, November 16th, 2018?

By Matthias Le Dall

----

I use a dataset containing 4 years of the West Texas Intermediate (WTI) prices to train a Long-Short-Term Memory (LSTM) neural net to predict the price of WTI on Friday Nov. 16th 2018.

The data set can be found here, https://fred.stlouisfed.org/series/DCOILWTICO. I used it because:
- It contains daily prices
- It is widely used as a benchmark for oil prices

My training routing consists of a walk-forward LSTM algorithm:
- Use a window of 8 days to predict the following day
- Slide the window forward by one day
- Repeat the previous two steps until the entire dataset is covered

This algorithm is effectively equivalent to a supervised learning routine, where the input is the past sequence and the label is the future sequence.

My prediction routine then consists of,
- Use the last 8 days of the dataset to predict the following day
- Slide the window forward
- Repeat the previous two steps until the desired date is predicted

I last updated my dataset on Nov. 13th, 2018.

'''

import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers

sns.set_style("darkgrid")

def data_import():
	data = pd.read_csv('DCOILWTICO.csv', sep = ',')
	data = data.rename({'DCOILWTICO':'WTI'}, axis=1)
	return data

def data_imputing(df):
	'''
	Imputes the data:
	- replaces the '.' by the average of the preceding and succeeding values
	- converts all entries from strings to floats
	'''
	# Replaces dots by 0's
	dot_indices = df[df['WTI'] == '.'].index
	for idx in dot_indices:
		df.loc[idx, 'WTI'] = 0
	# Replaces all strings to floats
	df['WTI'] = pd.to_numeric(df['WTI'])
	zero_indices = df[df['WTI'] == 0].index
	# Replaces all 0's by averages
	for idx in zero_indices:
		df.loc[idx, 'WTI'] = (df.loc[idx-1, 'WTI']+df.loc[idx+1, 'WTI'])/2
	data = df
	return data

def data_plot():
	data = data_import()
	df = data_imputing(data)
	last_date = df['DATE'].values[-1]
	first_date = df['DATE'].values[0]
	length = pd.to_datetime(last_date) - pd.to_datetime(first_date)
	length = int(str(length).split(' ')[0])
	length = length/365.25
	length = int(length)
	fig = plt.figure()
	df['WTI'].plot()
	tick_loc = list(df.index.values[::500])
	tick_label = [df.loc[i,'DATE'] for i in tick_loc]
	plt.xticks(tick_loc, tick_label)
	plt.xticks(rotation=45)
	plt.ylabel('WTI price')
	plt.title('Price of WTI oil for the past {} years'.format(length))
	fig.savefig('WTI_oil_prices.png', bbox_inches='tight', dpi=500)
	plt.show()


def train_predict_split(df, n_past = 8, n_future = 1):
	'''
	Keeps the last 8 days as the predict dataset used that will be used to predict the price on Nov.16th
	The rest will be used for training
	'''
	df_train = df[:-n_past]
	df_predict = df[-n_past:]
	df_predict['original?'] = ['Observed' for i in range(len(df_predict))]
	return df_train, df_predict

def data_scaler(df):
	X_data = df.values[:, 1:]
	scaler = MinMaxScaler(feature_range=(0, 1))
	X_scaled = scaler.fit_transform(X_data)
	# Creates the column 't' with the scaled prices
	df['t'] = X_scaled
	return scaler

def timeseries_to_supervised(df, n_past, n_future):
	'''
	Creates the features that the LSTM will learn from:
	- a window of 7 days in the past plus today
	- a window of 1 day in the future
	'''
	for i in range(1, n_past):
		label = 't-'+str(i)
		df[label] = df.shift(i)['t']
	for i in range(1, n_future+1):
		label = 't+'+str(i)
		df[label] = df.shift(-i)['t']
	df.dropna(inplace = True)
	df.index = range(len(df))


def training_validation_split(df, n_past, n_future):
	'''
	We split the training dataset into a "training" set and a validation set. The training set contains 90% of the whole dataset. 
	'''
	train_size = int(0.9*len(df))
	past = ['t']+['t-'+str(i) for i in range(1,n_past)]
	future = ['t+'+str(i) for i in range(1,n_future+1)]
	X_train, y_train = df[past][:train_size], df[future][:train_size]
	X_valid, y_valid = df[past][train_size:], df[future][train_size:]
	return X_train.values, y_train.values, X_valid.values, y_valid.values

def input_reshaping(xtrain, xvalid):
	'''
	Shapes the training and validation inputs compatible with the LSTM
	'''
	xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
	xvalid = xvalid.reshape(xvalid.shape[0], xvalid.shape[1], 1)
	return xtrain, xvalid

def model_trainer(df_train, n_past, n_future, model_name):
	'''
	Trains an LSTM Netword with 50 neurons, and a dense output layer with one neuron (because we only predict the price on one day)
	'''
	X_train, y_train, X_valid, y_valid = training_validation_split(df_train, n_past, n_future)
	X_train, X_valid = input_reshaping(X_train, X_valid)
	model = tf.keras.Sequential()
	model.add(layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
	model.add(layers.Dense(1))#y_train.shape[1]
	model.compile(loss='mae', optimizer='adam')
	history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_valid, y_valid), verbose=2, shuffle=False)
	tf.keras.models.save_model(model, model_name, overwrite=True, include_optimizer=True)
	fig = plt.figure()
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='validation')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Absolute Error')
	plt.legend()
	fig.savefig('LSTM_model_performance.png', bbox_inches='tight', dpi=500)
	plt.show()

def make_one_prediction(x_input, model):
	'''
	Makes a new prediction given an input of an 8-day price sequence
	'''
	x_input = x_input.reshape(1, x_input.shape[0],1)
	new_prediction = model.predict(x_input)[0][0]
	return new_prediction

def make_many_predictions(df_train, model, past, n_future):
	'''
	Iterates over the training set to predict all of the dataset. This will be used for validation of the network
	'''
	predict_prices = []
	for i in range(len(df_train[past])):
		x_input = df_train.loc[i, past].values
		predict_prices.append(make_one_prediction(x_input, model))
	predic_titles = ['predict t+'+str(i) for i in range(1,n_future+1)]
	predictions_df = pd.DataFrame(predict_prices, columns = predic_titles)
	return pd.concat([df_train, predictions_df], axis = 1)

def real_price_prediction(df_train, scaler):
	'''
	Takes the predicted scaled prices and inverts them back to real prices
	'''
	X_inverted = df_train.values[:, -1:]
	X_inverted = scaler.inverse_transform(X_inverted)
	df_inverted = pd.DataFrame(X_inverted, columns = ['predict WTI'])
	return pd.concat([df_train, df_inverted], axis = 1)

def plot_real_prediction(df_train):
	'''
	Plots the comparison of the observed prices to the predicted prices
	'''
	fig = plt.figure()
	df_train['WTI'].plot(label = 'Observed price')
	df_train['predict WTI'].shift(1).plot(label = 'Predicted price')
	tick_loc = list(df_train.index.values[::500])
	tick_label = [df_train.loc[i,'DATE'] for i in tick_loc]
	plt.xticks(tick_loc, tick_label)
	plt.xticks(rotation=45)
	plt.ylabel('WTI price')
	plt.legend()
	plt.title('Comparison between observed and predicted data')
	fig.savefig('WTI_oil_price_comparison.png', bbox_inches='tight', dpi=500)
	plt.show()

def prediction_from_predict_set(x_input, scaler, model):
	'''
	Makes a new prediction given a 8-day price sequence taken from the predict dataset
	'''
	x_input = x_input.reshape(x_input.shape[0],1)
	x_scaled = scaler.fit_transform(x_input)
	x_scaled = x_scaled.reshape(x_scaled.shape[0])
	predict = make_one_prediction(x_scaled, model)
	x_invert = scaler.inverse_transform(predict)
	return x_invert[0][0]

def one_more_day(dates):
	'''
	Generates the date of the next day given the array of dates in the dataset.
	'''
	date_str = dates.values[-1:][0]
	date_stamp = pd.to_datetime(date_str)
	new_date = pd.DatetimeIndex([date_stamp]) + pd.DateOffset(1)
	new_date_str = str(new_date[0]).split(' ')[0]
	return new_date_str

def prediction_walk_forward(df_predict, scaler, model):
	'''
	Given the 8 last days of the predicting set, this predicts the next day
	'''
	input_data = df_predict[-8:]
	last_index = input_data.index.values[-1]
	next_day = one_more_day(input_data['DATE'])
	x_input = input_data['WTI'].values
	predict_forward = prediction_from_predict_set(x_input, scaler, model)
	return next_day, predict_forward, last_index

def prediction_run_forward(df_predict, target_date, scaler, model):
	'''
	Iterates over the 'prediction_walk_forward' until the target date is reached
	'''
	last_date = df_predict['DATE'].values[-1]
	length = pd.to_datetime(target_date) - pd.to_datetime(last_date)
	length = int(str(length).split(' ')[0])
	for i in range(length):
		next_day, predict_forward, last_index = prediction_walk_forward(df_predict, scaler, model)
		df_predict.loc[last_index+1, ['DATE', 'WTI', 'original?']] = [next_day, predict_forward, 'Predicted']

def plot_prediction(df_predict, target_WTI_price, target_date):
	'''
	Plots the predicted prices until the target date
	'''
	fig = plt.figure()
	sns.lineplot(x="DATE", y="WTI", data = df_predict, hue = 'original?')
	plt.xticks(rotation=45)
	plt.ylabel('WTI price')
	plt.title('Price of oil on {} will be ${}'.format(target_date, target_WTI_price))
	fig.savefig('WTI_oil_price_predictions.png', bbox_inches='tight', dpi=500)
	plt.show()



