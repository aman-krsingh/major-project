from io import BytesIO
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential

import tempfile
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math 
from numpy import array
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.metrics import mean_squared_error

import logging

from azure.functions import HttpRequest, HttpResponse





def main(req: HttpRequest) -> HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    
    ticker='AAPL'
    
    time_step =10
    
    model = Sequential()
    
    model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    account_url = f"https://storageaacount456.dfs.core.windows.net/"
    token_credential = DefaultAzureCredential()
    service_client = DataLakeServiceClient(account_url, credential=token_credential)
    filesystem_client = service_client.get_file_system_client(file_system="stocks-data")
    directory_client = filesystem_client.get_directory_client(f"model/{ticker}")
    file_client = filesystem_client.get_paths(f"model/{ticker}",recursive=False)
    date_list = []
    for p in file_client:
        f = p.name
        f = f.replace(f"model/{ticker}","").replace(f"/{ticker}_","").replace(".h5","")
        try:
            dt = datetime.strptime(f,"%Y-%m-%d")
            date_list.append(dt)
        except:
            pass
    
    max_date = max(date_list)
    max_date_str = max_date.strftime("%Y-%m-%d")
    file_client = directory_client.get_file_client(f"model/{ticker}_{max_date_str}.h5")
    
    # Download the file content into a BytesIO object
    model_weights_bytes = BytesIO(file_client.download_file().readall())
    
    # Reset the buffer position
    model_weights_bytes.seek(0)
    
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        temp_file.write(model_weights_bytes.getvalue())
    
    # Load the model weights from the temporary file
    model.load_weights(temp_file.name)
    
    # Clean up the temporary file
    os.unlink(temp_file.name)
    
    f_client = filesystem_client.get_file_client(f"data/{ticker}/{ticker}_{max_date_str}.csv")
    downloaded_bytes = f_client.download_file().readall()
    read_bytes = BytesIO(downloaded_bytes)
    data = pd.read_csv(read_bytes)
    
    
    size = len(data)
    year = 365 * 5
    df = data[size - year:]
    
    data = df.reset_index()['Close']
    
    scaler = MinMaxScaler(feature_range=(0,1))
    data = scaler.fit_transform(np.array(data).reshape(-1,1))
    
    #splitting data.
    training_size = int(len(data) * 0.80)
    test_size = len(data) - training_size
    
    test_data = data[training_size:len(data), :1]
    
    # time_step =10
    
    #future prediction
    
    l= len(test_data)
    length = l-time_step
    
    X_input = test_data[length:].reshape(1,-1)
    
    #print(X_input.shape)
    
    temp_input = list(X_input)
    temp_input = temp_input[0].tolist()
    
    #print(temp_input)
    
    
    days =30
    
    lst_output=[]
    n_steps=time_step
    i=0
    
    while(i<days):
        if(len(temp_input) > time_step):
            #print(temp_input)
            X_input=np.array(temp_input[1:])
            #print("{} days input{}".format(i,X_input))
            X_input=X_input.reshape(1,-1)
            X_input = X_input.reshape((1, n_steps, 1))
            #print(X_input)
            yhat = model.predict(X_input, verbose=0)
            #print("{} day output{}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            X_input=X_input.reshape((1, n_steps,1))
            yhat = model.predict(X_input, verbose=0)
            #print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            #print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    #print(lst_output)
    
    
    time_step_plus1 = time_step + 1
    
    #number from 1 to time_step + 1
    days_new=np.arange(1, time_step_plus1)
    
    
    #number from time_step +1 to 30 more (6 to 35 [30 numbers])
    days_pred =np.arange(time_step_plus1, time_step_plus1 + len(lst_output))
    
    ld=len(data)
    L = ld-time_step
    
    new_data=data.tolist()
    new_data.extend(lst_output)
    
    
    #   future 30days predicted value.
    days_pred = scaler.inverse_transform(lst_output)
    # plt.plot(days_pred, color = "red")
    # plt.show()
    
    
    new_data=data.tolist()
    new_data.extend(lst_output)
    new_data = scaler.inverse_transform(new_data).tolist()
    data = scaler.inverse_transform(data)
    
    # plt.plot(new_data, color='red')
    # plt.plot(data, color='blue')
    # plt.show()  

    
    result={
        "pred_value": days_pred,
        "hist_with_pred_value":new_data,
        "hist_value":data
    }
    return HttpResponse(result)

