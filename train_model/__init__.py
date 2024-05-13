import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math 
from numpy import array

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.metrics import mean_squared_error

import logging

from azure.functions import HttpRequest, HttpResponse

   ## creating function for creating dataset for train and test.
    
def create_dataset(dataset, time_step = 1):
    dataX, dataY =[], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def main(req: HttpRequest) -> HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')


    account_url = f"https://storageaacount456.dfs.core.windows.net/"
    token_credential = DefaultAzureCredential()
    service_client = DataLakeServiceClient(account_url, credential=token_credential)

    filesystem_client = service_client.get_file_system_client(file_system="stocks-data")
    directory_client = filesystem_client.get_directory_client(f"data/{ticker}")
    file_client = filesystem_client.get_paths(f"data/{ticker}",recursive=False)
    date_list = []
    for p in file_client:
        f = p.name
        f = f.replace(f"data/{ticker}","").replace(f"/{ticker}_","").replace(".csv","")
        try:
            dt = datetime.strptime(f,"%Y-%m-%d")
            date_list.append(dt)
        except:
            pass
    
    max_date = max(date_list)
    max_date_str = max_date.strftime("%Y-%m-%d")
    
    ticker='AAPL'
    data = pd.read_csv(f'./data/{ticker}/{ticker}_{max_date_str}.csv')
    
    size = len(data)
    year = 365 * 5
    df = data[size - year:]
    
    data = df.reset_index()['Close']
    
    #data.to_csv(f'./data/{ticker}_5yrs.csv')
    
    #print (data)
    # plt.plot(data)
    # plt.show()
    
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(np.array(data).reshape(-1,1))
    # print(data)
    
    
    #splitting data.
    training_size = int(len(data) * 0.40)
    test_size = len(data) - training_size
    
    train_data, test_data = data[0:training_size,:], data[training_size:len(data), :1]
    #print(training_size, test_size)
    
    
    time_step =5
    
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    
    # print(X_train.shape), print(Y_train.shape)
    # print(Y_train)
    
    # making data ready for LSTM model reshaping for value to give input in model.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    #good to go for createing model.
    
    model = Sequential()
    
    model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    #model.summary()
    
    #traning the model and saving it.
    model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=150,batch_size=64,verbose=1)

    directory_client = filesystem_client.get_directory_client(f"model/{ticker}")
    file_client = directory_client.get_file_client(f"{ticker}_{current_date}.h5")
    model_bytes= BytesIO()
    
    model.save( model_bytes)

    model_bytes.seek(0)
    file_client.upload_data(model_bytes,overwrite = True)   
    
    if name:
        return HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200
        )
