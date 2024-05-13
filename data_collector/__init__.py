from io import BytesIO
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential

import logging
from azure.functions import TimerRequest

from datetime import datetime, timezone

import yfinance as yf


def main(mytimer: TimerRequest) -> None:
    utc_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d")
    
    ticker_list = ['AAPL', 'META', 'GOOG']
    API_key = 'WSKF50ODKWY4WP1O'
    
    #storage acount address
    account_url = f"https://storageaacount456.dfs.core.windows.net/"
    token_credential = DefaultAzureCredential()

    #service clint bana dega data lake ka
    service_client = DataLakeServiceClient(account_url, credential=token_credential)
    filesystem_client = service_client.get_file_system_client(file_system="stocks-data")    

    for ticker in ticker_list:
        directory_client = filesystem_client.get_directory_client(f"data/{ticker}")
        file_client = directory_client.get_file_client(f"{ticker}_{current_date}.csv")
        prd= 12*25
        df =yf.download(f'{ticker}', period=f'{prd}mo')
        csv_bytes= BytesIO()
        df.to_csv(csv_bytes)
        csv_bytes.seek(0)
        file_client.upload_data(csv_bytes,overwrite = True)

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)
