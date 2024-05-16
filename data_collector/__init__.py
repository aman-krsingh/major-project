import os,requests,logging
import yfinance as yf
from io import BytesIO
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
from azure.functions import TimerRequest
from datetime import datetime, timezone


def main(mytimer: TimerRequest) -> None:
    utc_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d")
    
    ticker_list = ['AAPL', 'GOOG', 'META', 'PAYTM', 'TCS']

    ticker_dict = {
        'AAPL': "AAPL",
        'GOOG': "GOOG",
        'META': "META",
        'PAYTM': "PAYTM.NS",
        'TCS': "TCS.NS"
    }
    
    #storage acount address
    account_url = f"https://storageaacount456.dfs.core.windows.net/"
    token_credential = DefaultAzureCredential()

    service_client = DataLakeServiceClient(account_url, credential=token_credential)
    filesystem_client = service_client.get_file_system_client(file_system="stocks-data")    

    for ticker in ticker_list:
        directory_client = filesystem_client.get_directory_client(f"data/{ticker}")
        file_client = directory_client.get_file_client(f"{ticker}_{current_date}.csv")
        prd= 12*25
        df =yf.download(ticker_dict[ticker], period=f'{prd}mo')
        csv_bytes= BytesIO()
        df.to_csv(csv_bytes)
        csv_bytes.seek(0)
        file_client.upload_data(csv_bytes,overwrite = True)
        url = os.environ[f"train_{ticker}_url"]
        res = requests.get(url)

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)
