"""
ADG Efficiency
2017-01-31

This model runs experiments on model parameters on a neural network.
The neural network is built in Keras and uses TensorFlow as the backend.

To setup a model run you need to look at
- mdls_D
- mdls_L
- machine learning parameters
- pred_net_D

This script is setup for an experiment on the types
of data used as model features.

This model is setup to run an experiment on the sources of data.
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import pickle
import datetime

# importing some useful scikit-learn stuff
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# importing Keras & Tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam, RMSprop, SGD
from keras.metrics import mean_absolute_error
import keras.backend.tensorflow_backend as KTF # https://groups.google.com/forum/#!topic/keras-users/MFUEY9P1sc8

import tensorflow as tf

# importing plotly
import plotly as py
import plotly.graph_objs as go  
import plotly.tools as tls  
tls.set_credentials_file(username="YOUR USERNAME", api_key="YOUR API KEY")

# imported from another script 
from metrics import MASE 

# workaround found on stack overflow
# http://stackoverflow.com/questions/40046619/keras-tensorflow-gives-the-error-no-attribute-control-flow-ops
tf.python.control_flow_ops = tf 

# grabbing the TF session from Keras backend
sess = KTF.get_session() 

# function that inputs the database path 
def get_data(db_path,table):
    conn = sqlite3.connect(db_path)
    data = pd.read_sql(sql='SELECT * from ' + str(table),con = conn)
    conn.close()
    print('got sql data')
    return data # returns a df with all table data

# setting base directory
base_path = os.path.dirname(os.path.abspath(__file__))

# location of SQL database 
sql_path = os.path.join(base_path,'data','ELEXON DATA.sqlite')
    
# saving run time for use in creating folder for this run of experiments
run_name = str(datetime.datetime.now())
run_name = run_name.replace(':','-')
run_path = str('Experiment Results//' + run_name)
results_csv_path = os.path.join(base_path, run_path, 'results.csv')
history_csv_path = os.path.join(base_path, run_path, 'histories.csv')

# dictionary containing all of the infomation for each report
data_price_D = {'Imbalance Price':{'Report name':'B1770','Data name':'imbalancePriceAmountGBP'}}

data_price_vol_D = {'Imbalance Price':{'Report name':'B1770','Data name':'imbalancePriceAmountGBP'},
          'Imbalance Volume':{'Report name':'B1780','Data name':'imbalanceQuantityMAW'}}

# list of model names - done manually to control order & which models run
mdls_L = ['price, vol & sparse'] # 'price only', 'price & volume', 'price & sparse', 

# dictionary of mdls with parameters
# FL = first lag.  LL = last lag.  SL = step size, 
# Sparse? = to include sparse data for trend/seasonality or not, 
# Data dict = which dictionary of data to use for the lagged time series
mdls_D = {'price only':{'date_start':'01/01/2016 00:00','date_end':'31/12/2016 00:00','FL':48,'LL':48*2,'SL':48,'Sparse?':False,'Data dict' : data_price_D},
'price & volume':{'date_start':'01/01/2016 00:00','date_end':'31/12/2016 00:00','FL':48,'LL':48*2,'SL':48,'Sparse?':False,'Data dict' : data_price_vol_D},
'price & sparse':{'date_start':'01/01/2016 00:00','date_end':'31/12/2016 00:00','FL':48,'LL':48*2,'SL':48,'Sparse?':True,'Data dict' : data_price_D},
'price, vol & sparse':{'date_start':'01/01/2016 00:00','date_end':'31/12/2016 00:00','FL':48,'LL':48*2,'SL':48,'Sparse?':True,'Data dict' : data_price_vol_D}}

# machine learning parameters
n_folds = 10 # number of folds used in K-fold cross validation
epochs = 200 # number of epochs used in training 
random_state_split = 5
random_state_CV = 3

# dataframes for use in tracking model results
results_cols = ['Model name','CV MASE','Training MASE','Test MASE','MASE',
                'Training MAE','Test MAE','MAE','CV Loss','Training Loss',
                'Training History','Error Time Series']
results_DF = pd.DataFrame(columns = results_cols)
results_all_DF = results_DF 
history_all_DF = pd.DataFrame()

# for loop to iterate through models
for mdl_index, mdl in enumerate(mdls_L):
    mdl_params = mdls_D[mdl]
    
    # resetting dataframes for storing results from this model run
    results_DF = pd.DataFrame(index = [mdl], columns = results_cols)
    history_DF = pd.DataFrame()

    # creating folder for this run
    exp_path = str(run_path + '/' + str(mdl))
    os.makedirs(exp_path)

    # setting model parameters
    date_start = mdl_params['date_start']
    date_start = mdl_params['date_end']
    first_lag = mdl_params['FL']
    last_lag = mdl_params['LL']
    step_lag = mdl_params['SL']
    include_sparse = mdl_params['Sparse?']
    data_D = mdl_params['Data dict']
    
    # unpacking dictionary of data to be used 
    data_sources = [key for key in data_D] # ie Imbalance Price, Imbalance Volume
    
    # getting the imbalance price first
    if data_sources[0] != 'Imbalance Price':
        price_loc = data_sources.index('Imbalance Price')
        first_source = data_sources[0]
        data_sources[price_loc] = first_source
        data_sources[0] = 'Imbalance Price'
    print(data_sources)

    table_names = [data_D[item]['Report name'] for item in data_sources] 
    data_col_names = [data_D[item]['Data name'] for item in data_sources]
    
    # getting data from SQL
    data_L = [get_data(db_path = sql_path, table = data_D[data_source]['Report name']) for data_source in data_sources]

    # list of the index objects
    indexes_L = [pd.to_datetime(raw_data['index']) for raw_data in data_L]     
        
    # list of the settlement periods
    SP_L = [raw_data['settlementPeriod'] for raw_data in data_L]

    # list of the actual data objects
    data_objs_L = [raw_data[data_col_names[index]].astype(float) for index, raw_data in enumerate(data_L)]
    
    # indexing these data objects
    for i, series in enumerate(data_objs_L):
        df = pd.DataFrame(data=series.values, index=indexes_L[i],columns=[series.name])
        data_objs_L[i] = df

    # creating feature dataframe - gets reset every model run
    data_DF = pd.DataFrame()
    for data_index, data_obj in enumerate(data_objs_L):
        # creating lagged data frame (make this a function)
        for i in range(first_lag,last_lag,step_lag):
            lag_start, lag_end, lag_step = 1, i, 1
            # creating the lagged dataframe
            data_name = data_obj.columns.values[0]
            lagged_DF = pd.DataFrame(data_obj)
            
            for lag in range(lag_start,lag_end+1,lag_step):
                lagged_DF[str(data_name) + ' lag_'+str(lag)] = lagged_DF[data_name].shift(lag) 
            print(lagged_DF.columns)
            lagged_DF = lagged_DF[lag_end:] # slicing off the dataframe
            index = lagged_DF.index # saving the index   
        
        data_DF = pd.concat([data_DF,lagged_DF],axis=1) # creating df with data       
           
    SP = SP_L[0]                        
    SP = SP[(len(SP)-len(data_DF)):].astype(float) # slicing our settlement periods 
    
    # creating our sparse matricies for seasonality & trend
    date = index
    # creating label binarizer objects 
    encoder_SP = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    encoder_days = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    encoder_months = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    # creating sparse settelment period feature object
    encoder_SP.fit(SP)
    encoded_SP = encoder_SP.transform(SP)
    SP_features = pd.DataFrame(encoded_SP, index, columns = list(range(1,51)))
    # creating sparse day of the week feature object
    days = list(map(lambda x: x.weekday(), date))
    encoder_days.fit(days)
    encoded_days = encoder_days.transform(days)
    days_features = pd.DataFrame(encoded_days, index = index, 
                                 columns = ['Mo','Tu','We','Th','Fr','Sa','Su'])
    # creating sparse month feature object
    months = list(map(lambda x: x.month, date))
    encoder_months.fit(months)
    encoded_months = encoder_months.transform(months)
    months_features = pd.DataFrame(encoded_months, index = index, 
                                   columns = ['Ja','Feb','Mar','Ap','Ma','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    
    sparse_features = pd.concat([SP_features,days_features, months_features],axis=1)    
    
    print('incl sparse is ' + str(include_sparse))
    if include_sparse == True:
        print('including sparse')
        data_DF = pd.concat([data_DF,sparse_features],axis=1)
        
    # saving our feature matrix to a csv for checking
    features_path = os.path.join(base_path, exp_path, 'features.csv')
    data_DF.to_csv(features_path)    
    
    # creating our target matrix (the imbalance price)
    y = data_DF['imbalancePriceAmountGBP']
    y.reshape(1,-1)
    
    # dropping out the actual values from our data
    for data_col_name in data_col_names:
        data_DF = data_DF.drop(data_col_name,1)
        
    # setting our feature matrix    
    X = data_DF
    
    # splitting into test & train
    # keeping the split the same for different model runs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = random_state_split)
    split_D = {'X_train' : len(X_train), 'X_test' : len(X_test)}
    
    # standardizing our data        
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    X_all = X_scaler.transform(X)
    
    # reshaping
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).flatten()
    
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).flatten()
    
    X = np.asarray(X)  # not sure if I use this anywhere - also this data is not standardized 
    y = np.asarray(y).flatten()
    
    # saving our scaler objects for use later
    pickle.dump(X_scaler, open(os.path.join(base_path,'pickle', 'X_scaler - ' + mdl + '.pkl'), 'wb'), protocol=4) 
    
    # length of the inputs
    training_samples = X_train.shape[0]
    input_length = X_train.shape[1] # should probably rename this to width....
    batch_size = int(training_samples/4)

    pred_net_D = {'price only':{'Layer 1' : Dense(1000, input_dim = input_length, activation = 'relu'), 'Layer 2' : Dense(1000, activation='relu'), 'Layer 3' : Dense(1000, activation='relu'), 'Layer 4' : Dense(output_dim = 1, activation='linear'),'Dropout fraction':0.3},
    'price & volume':{'Layer 1' : Dense(1000, input_dim = input_length, activation = 'relu'), 'Layer 2' : Dense(1000, activation='relu'), 'Layer 3' : Dense(1000, activation='relu'), 'Layer 4' : Dense(output_dim = 1, activation='linear'),'Dropout fraction':0.3},
    'price & sparse':{'Layer 1' : Dense(1000, input_dim = input_length, activation = 'relu'), 'Layer 2' : Dense(1000, activation='relu'), 'Layer 3' : Dense(1000, activation='relu'), 'Layer 4' : Dense(output_dim = 1, activation='linear'),'Dropout fraction':0.3},         
    'price, vol & sparse':{'Layer 1' : Dense(1000, input_dim = input_length, activation = 'relu'), 'Layer 2' : Dense(1000, activation='relu'), 'Layer 3' : Dense(1000, activation='relu'), 'Layer 4' : Dense(output_dim = 1, activation='linear'),'Dropout fraction':0.3} 
    }
    
    # defining layers for prediction network
    pred_net_params = pred_net_D[mdl]
    layer1 = pred_net_params['Layer 1']
    layer2 = pred_net_params['Layer 2']
    layer3 = pred_net_params['Layer 2']
    layer4 = pred_net_params['Layer 2']
    layer5 = pred_net_params['Layer 2']
    layer6 = pred_net_params['Layer 2']
    layer7 = pred_net_params['Layer 2']
    layer8 = pred_net_params['Layer 2']
    layer9 = pred_net_params['Layer 2']
    layer10 = pred_net_params['Layer 4']
    dropout_fraction = pred_net_params['Dropout fraction'] 

    with KTF.tf.device('gpu:0'): # force tensorflow to train on GPU
        KTF.set_session(KTF.tf.Session(config=KTF.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
        
        def get_model(): # use this function so I can recreate new network within CV loop
            network = Sequential()            
            network.add(layer1)
            network.add(Dropout(dropout_fraction))
            network.add(layer2)
            network.add(layer3)
            network.add(layer4)
            network.add(layer5)
            network.add(layer6)
            network.add(layer10)
            return network
            
        # https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668        
        def shuffle_weights(model, weights=None):
            """Randomly permute the weights in `model`, or the given `weights`.
            This is a fast approximation of re-initializing the weights of a model.
            Assumes weights are distributed independently of the dimensions of the weight tensors
              (i.e., the weights have the same distribution along each dimension).
            :param Model model: Modify the weights of the given model.
            :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
              If `None`, permute the model's current weights.
            """
            if weights is None:
                weights = model.get_weights()
            weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
            # Faster, but less random: only permutes along the first dimension
            # weights = [np.random.permutation(w) for w in weights]
            model.set_weights(weights)
                        
        optimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        # cross validation
        print('Starting Cross Validation')
        CV_network = get_model()
        CV_network.compile(loss='mean_squared_error', optimizer=optimizer)
        initial_weights_CV = CV_network.get_weights()
        CV = KFold(n_splits=n_folds, random_state = random_state_CV)
        MASE_CV_L = []
        loss_CV_L = []

        for train, test in CV.split(X_train, y_train):
            shuffle_weights(CV_network, initial_weights_CV)
            mdl_CV = CV_network.fit(X_train[train], y_train[train], nb_epoch = epochs, batch_size = batch_size)  

            y_CV_pred = CV_network.predict(X_train[test],batch_size = batch_size, verbose=0).flatten()
            MASE_CV = MASE(y_train[test],y_CV_pred, 48)
            MASE_CV_L.append(MASE_CV)
            loss_CV = mdl_CV.history['loss'][-1]
            loss_CV_L.append(loss_CV)
    
        MASE_CV = np.average(MASE_CV_L)
        loss_CV = np.average(loss_CV_L)
        
        # training network on all training data    
        print('Training prediction network')
        network = get_model()
        network.compile(loss='mean_squared_error', optimizer=optimizer) 
        initial_weights_net = network.get_weights()
        
        shuffle_weights(network, initial_weights_net)
        mdl_net = network.fit(X_train, y_train, nb_epoch = epochs, batch_size = batch_size)
    
        y_pred_train = network.predict(X_train,batch_size = batch_size, verbose=0).flatten()
        y_pred_test = network.predict(X_test,batch_size = batch_size, verbose=0).flatten()
        y_pred = network.predict(X_all,batch_size = batch_size, verbose=0).flatten()
        
        error_train = y_pred_train - y_train
        error_test = y_pred_test - y_test        
        error = y - y_pred
        abs_error = abs(error)     
    
    MASE_train = MASE(y_train, y_pred_train, 48)
    MASE_test = MASE(y_test, y_pred_test, 48)
    MASE_all = MASE(y, y_pred, 48)
    
    MAE_train = mean_absolute_error(y_train, y_pred_train).eval(session=sess)
    MAE_test = mean_absolute_error(y_test, y_pred_test).eval(session=sess)
    MAE_all = mean_absolute_error(y, y_pred).eval(session=sess)
    
    results_DF.loc[mdl,'Model name'] = mdl
    results_DF.loc[mdl,'CV MASE'] = MASE_CV
    results_DF.loc[mdl,'Training MASE'] = MASE_train
    results_DF.loc[mdl,'Test MASE'] = MASE_test
    results_DF.loc[mdl,'MASE'] = MASE_all
    
    results_DF.loc[mdl,'Training MAE'] = MAE_train
    results_DF.loc[mdl,'Test MAE'] = MAE_test
    results_DF.loc[mdl,'MAE'] = MAE_all
    
    results_DF.loc[mdl,'CV Loss'] = loss_CV
    results_DF.loc[mdl,'Training Loss'] = mdl_net.history['loss'][-1]

    results_DF.loc[mdl,'Error Time Series'] = error
    results_DF.loc[mdl,'Training History'] = mdl_net.history['loss']

    history_DF = pd.DataFrame(data=list(mdl_net.history.values())[0], index = list(range(1,epochs+1)), columns=[mdl])
       
    # figure 1 - plotting the actual versus prediction 
    actual_imba_price_G = go.Scatter(x=index,y=y,name='Actual',line=dict(width=2))   
    predicted_imba_price_G = go.Scatter(x=index,y=y_pred,name='Predicted',line=dict(width=2, dash = 'dash'))
    fig1_data = [actual_imba_price_G, predicted_imba_price_G]
    fig1_layout = go.Layout(title='Forecast',yaxis=dict(title='Imbalance Price [£/MWh]'))
    fig1 = go.Figure(data=fig1_data,layout=fig1_layout)    
    fig1_name = os.path.join(exp_path,'Figure 1.html')
    py.offline.plot(fig1,filename = fig1_name, auto_open = False) # creating offline graph
    # py.plotly.plot(fig1, filename='Forecast', sharing='public') # creating online graph
    
    # saving results
    network_params_DF = pd.DataFrame(pred_net_params, index=[mdl])
    mdl_params_DF = pd.DataFrame(mdl_params, index=[mdl])
    results_DF = pd.concat([results_DF, network_params_DF, mdl_params_DF], axis = 1, join = 'inner')
    results_all_DF = pd.concat([results_all_DF, results_DF], axis = 0)
    history_all_DF = history_all_DF.join(history_DF, how='outer')

print(results_all_DF)    

results_all_DF.to_csv(results_csv_path)    
history_all_DF.to_csv(history_csv_path)    

# figure 2 - comparing training history of models
fig2_histories = [history_all_DF[col] for col in history_all_DF]
fig2_data = [go.Scatter(x = data.index, y = data, name=data.name) for data in fig2_histories]
fig2_layout = go.Layout(title='Training History',yaxis=dict(title='Loss'),xaxis=dict(title='Epochs'))
fig2 = go.Figure(data=fig2_data,layout=fig2_layout)
fig2_name = os.path.join(run_path,'Figure 2.html')
py.offline.plot(fig2,filename = fig2_name, auto_open = False) # creating offline graph