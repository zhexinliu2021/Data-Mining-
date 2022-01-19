#!/usr/bin/python3
import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tqdm import tqdm

def merge_data(df):
    "merge the times in to a row"
    #'{ 2014-02-17 : {call:22, mood: 44,.. }, ...          }
    item_dict= dict()
    for i in range(df.shape[0]):
    #for time in df.index:
        o_time = df.index[i]; var = str(df.iloc[i,:].variable); val = df.iloc[i,:].value
        #o_time = time; var = str(df.loc[o_time,'variable']); val = df.loc[o_time, 'value']
        time = str(o_time)
        time = time.strip().split()[0]

        item_dict.setdefault(time, []).append((var, val))
    # then merge the same variable in one day
    for time, i_list in item_dict.items():
        # [ 'time' : [ [call,val], ...]
        inter_dict = dict() # { 'call:(sum, number)
        for var, val in i_list:
            if var not in inter_dict:
                inter_dict[var]= [val,1]
            elif var in inter_dict:
                inter_dict[var][0] += val
                inter_dict[var][1] +=1
        'get mean'
        for var,tup in inter_dict.items():
            inter_dict[var] = tup[0]/tup[1]
        item_dict[time] = inter_dict

        del inter_dict
    result = pd.DataFrame(item_dict).T
    result.index = pd.to_datetime(result.index)
    return  result


def split_data(df,time_interval='1H'  ):
    """
    input: reframed dataframe.

    :param df:
    :return:
    """
    inter = float(time_interval.split('H')[0])
    'set 24 hours long as length of the test set'
    n_test = int(24/inter)
    n_instance = df.shape[0]
    n_max_test = n_instance - n_test*5
    print(n_instance)
    print(n_test)
    'let the user decide how many step length , '
    print('# of instances are: '+str(n_instance))
    step_length = int(input('how many step length you want:'))

    end_index = n_test*5
    index_list = []; index_list.append((end_index, end_index+n_test)) # the index for the frist training.
    while end_index+step_length+n_test <= (n_instance-1):
        'each iter we generate the end index for training set and test set'
        end_index += step_length
        train_index = end_index
        test_index = train_index + n_test
        #end_index += n_test
        index_list.append((train_index, test_index))

    print('maximum testing number:'+str(len(index_list)))
    n_time = int(input('number of validation :'))

    end_index = index_list[:n_time]

    print(end_index)
    return end_index

#df = np.random.randn(100,2)
#a = split_data(df)
#print(a)
def run_model(in_list, reframed, n_days, n_features,equal_train = False, validation_split=0.2, verbose = 0, scal=None):

    print('total number of testing: '+str(len(in_list)))
    error_list_pre = np.array([])
    error_list_ac = np.array([])
    error_ben = np.array([])
    scaler = scal
    error_test_arr = pd.DataFrame(columns=['actual','prediction','benchmark'])

    for i in tqdm(range(len(in_list))):
        iter_n = i+1; train_index = in_list[i][0]; test_index = in_list[i][1]

    #for iter_n, (train_index, test_index) in enumerate(in_list):

        n_test = test_index - train_index
        values = reframed.values
        # split into train and test sets
        if equal_train:
            train = values[train_index-5*n_test:train_index, :]
        else:
            train = values[:train_index, :]
        test = values[train_index:test_index, :]

        # split into input and outputs
        n_obs = n_days * n_features
        train_X, train_y = train[:, :n_obs], train[:, -n_features]
        test_X, test_y = test[:, :n_obs], test[:, -n_features]


        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_days, n_features))

        ' design the network'
        # design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        #fit the network
        history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_split=validation_split, verbose=verbose, shuffle=False)

        # make a prediction
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], n_days * n_features))

        # invert scaling for forecast

        inv_yhat = np.concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]

        #Invert scaling for training. (Bechmark)
        length_test = (test_index-train_index)
        train_X_ben = train_X.reshape(train_X.shape[0], n_days*n_features)[-length_test:,:]
        train_Y_ben = train_y.reshape(len(train_y),1)[-length_test:,:]
        inv_Y_ben = np.concatenate((train_Y_ben,train_X_ben[:,-(n_features-1):]), axis=1)
        inv_Y_ben = scaler.inverse_transform(inv_Y_ben)
        inv_Y_ben = inv_Y_ben[:,0]
        error_ben = np.append(error_ben, np.mean(inv_Y_ben))

        #concatenate mood values in pre, ac, and benchmark
        inv_y_ben_test = inv_y[1:]


        #inv_y_ben_test=np.insert(inv_y_ben_test,0,inv_Y_ben[-1] )



        #error_test_arr = pd.concat([error_test_arr,pd.DataFrame({'actual': inv_y,'prediction':inv_yhat,'benchmark':inv_Y_ben})], axis=0)
        error_test_arr = pd.concat([error_test_arr,pd.DataFrame({'actual': [np.mean(inv_y)],'prediction': [np.mean(inv_yhat)],'benchmark':[np.mean(inv_Y_ben)]})], axis=0)


        MSE = mean_squared_error(inv_y, inv_yhat)
        #print(MSE)
        #error_list = np.append(error_list,MSE)
        error_list_pre = np.append(error_list_pre, np.mean(inv_yhat))
        error_list_ac = np.append(error_list_ac, np.mean(inv_y))

        "print MSE each training "
        print('mse pre vs ac:'+ str(mean_squared_error(inv_yhat,inv_y)))
        print('mse ben vs ac:'+str(mean_squared_error(inv_y, inv_Y_ben))+'\n')
        print('mse of mean this time:'+str(mean_squared_error( [np.mean(inv_yhat)],[np.mean(inv_y)]   )))
        print('mse of mean this time:'+str(mean_squared_error( [np.mean(inv_Y_ben)],[np.mean(inv_y)]   )))
        plt.plot(inv_yhat, label='predicted')
        plt.plot(inv_y, label='actual')
        plt.legend()
        plt.show()

    print('\n\nmean MSE preVSac: '+ str(mean_squared_error(error_list_pre,error_list_ac)))
    print('MSE benVSac:' +str(mean_squared_error(error_ben,error_list_ac)))
    plt.plot(error_ben, label='benchmark')
    plt.plot(error_list_pre, label='prediction')
    plt.plot(error_list_ac, label='actual')
    plt.legend()
    plt.show()
    error_test_arr.to_csv('prediction_mean_36days.csv')


def bar(n = 10):
    for i in tqdm(range(n)):pass


