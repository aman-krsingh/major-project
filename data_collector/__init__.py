from io import BytesIO
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential


import logging

from azure.functions import TimerRequest
from datetime import datetime, timezone

from alpha_vantage.timeseries import TimeSeries



def main(mytimer: TimerRequest) -> None:
    utc_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    ticker_list = ['AAPL', 'META', 'GOOG']

    API_key = 'WSKF50ODKWY4WP1O'
    #storage acount address
    account_url = f"https://we.dfs.core.windows.net"
    token_credential = DefaultAzureCredential()

    #service clint bana dega data lake ka
    service_client = DataLakeServiceClient(account_url, credential=token_credential)
    filesystem_client = service_client.get_file_system_client(file_system="container-name")
    directory_client = filesystem_client.get_directory_client("dir")
    file_client = directory_client.get_file_client("")
    

    for ticker in ticker_list:
        ts = TimeSeries(key= API_key, output_format='pandas')
        res = ts.get_daily(ticker, outputsize='full')
        df=res[0]
        csv_bytes= BytesIO()
        df.to_csv(csv_bytes)
        csv_bytes.seek(0)
        file_client.upload_data(csv_bytes,overwrite = True)

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)
