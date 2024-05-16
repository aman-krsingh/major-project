import os,math,logging,tempfile
import numpy as np
import pandas as pd
from io import BytesIO
from numpy import array
from datetime import datetime
import matplotlib.pyplot as plt
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from azure.functions import HttpRequest, HttpResponse

    
def create_dataset(dataset, time_step = 1):
    dataX, dataY =[], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def main(req: HttpRequest) -> HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    account_url = f"https://storageaacount456.dfs.core.windows.net/"
    token_credential = DefaultAzureCredential()
    service_client = DataLakeServiceClient(account_url, credential=token_credential)

    ticker='TCS.NS'
   
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
    training_size = int(len(data) * 0.60)
    test_size = len(data) - training_size
    
    train_data, test_data = data[0:training_size,:], data[training_size:len(data), :1]
    
    time_step =15
    
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    
    # making data ready for LSTM model reshaping for value to give input in model.
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    #traning the model and saving it.
    model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=150,batch_size=64,verbose=1)

    directory_client = filesystem_client.get_directory_client(f"model/{ticker}")
    file_client = directory_client.get_file_client(f"{ticker}_{max_date_str}.h5")
    
    # Save the model weights to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        model.save_weights(temp_file.name)
   
    # Read the temporary file into a BytesIO object
    with open(temp_file.name, 'rb') as f:
        model_weights_bytes = BytesIO(f.read())

    # Reset the buffer position
    model_weights_bytes.seek(0)
   
    # Upload the model weights to Azure Storage
    file_client.upload_data(model_weights_bytes, overwrite=True)
   
    # Clean up the temporary file
    os.unlink(temp_file.name)  
    
    return HttpResponse("This HTTP triggered function executed successfully.", status_code=200)
