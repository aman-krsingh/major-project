import logging

from azure.functions import TimerRequest
from datetime import datetime, timezone

from alpha_vantage.timeseries import TimeSeries



def main(mytimer: TimerRequest) -> None:
    utc_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    ticker_list = ['AAPL', 'META', 'GOOG']

    API_key = 'WSKF50ODKWY4WP1O'

    for ticker in ticker_list:
        ts = TimeSeries(key= API_key, output_format='pandas')
        res = ts.get_daily(ticker, outputsize='full')
        df=res[0]
        #df.to_csv(f'./data/{ticker}2.csv')

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)
