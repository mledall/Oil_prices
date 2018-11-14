from oil_prices import *


with_without = 'without training'
show_plot = 'yes'

print('START')

# Defining the past and future sequences for the LSTM training
n_past = 8
n_future = 1
target_date = '2018-11-16'
past = ['t']+['t-'+str(i) for i in range(1,n_past)]
future = ['t+'+str(i) for i in range(1,n_future+1)]

# Importing and feature engineering data
print(' - Imports data and formats the data')
data = data_import()
df = data_imputing(data)
df_train, df_predict = train_predict_split(df, n_past, n_future)
scaler = data_scaler(df_train)
timeseries_to_supervised(df_train, n_past, n_future)

# Training the model anew if needed, otherwise, just loaded a pre-trained model
model_name = 'WTI_oil_price.mdl'
if with_without == 'with training':
	print(' - Training the LSTM model')
	model_trainer(df_train, n_past, n_future, model_name)
print(' - Loading the LSTM model')
model = tf.keras.models.load_model(model_name, custom_objects=None, compile=True)

# Validating the neural net by predicting all of the set and comparing with the observed data
df_train = make_many_predictions(df_train, model, past, n_future)
df_train = real_price_prediction(df_train, scaler)


# Predicting the oil price on Friday, November 16th, 2018.
prediction_run_forward(df_predict, target_date, scaler, model)
target_WTI_price = df_predict[df_predict['DATE'] == target_date]['WTI'].values[0]
print('Price of WTI oil on {}: $ {}'.format(target_date, target_WTI_price))

if show_plot == 'yes':
	data_plot()
	plot_real_prediction(df_train)
	plot_prediction(df_predict, target_WTI_price, target_date)

print('END')


