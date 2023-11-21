# latest code
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lib.utils import read_json
import warnings
import time


path = os.getcwd()
config = read_json('config/default.json')
params = config['params']
warnings.filterwarnings("ignore")

# Define the list of stock symbols
stock_symbols = config['stock_list_regime']

# Download historical data for all stocks
historical_data = {}
for symbol in stock_symbols:
    historical_data[symbol] = yf.download(symbol, period="10y")


# Function to perform regime shift analysis on a stock
def analyze_stock(symbol, date, historical_data, cluster_cache):
    try:
        # Check if we have cached clustering results
        if symbol in cluster_cache and date <= cluster_cache[symbol]['end_date']:
            scaler = cluster_cache[symbol]['scaler']
            kmeans = cluster_cache[symbol]['kmeans']
            cluster_labels = cluster_cache[symbol]['cluster_labels']
            cluster_volatility = cluster_cache[symbol]['cluster_volatility']
        else:
            end_date = date
            start_date = end_date - timedelta(days=2 * 365)
            data = historical_data[symbol].loc[start_date:end_date]

            if data.empty:
                raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")

            data["Realized_Volatility"] = data["Close"].pct_change().rolling(window=5).std()
            data["Change_in_Volatility"] = data["Realized_Volatility"].pct_change().rolling(window=5).std()
            data["ATR"] = data["High"] - data["Low"]

            dataset = data[["Realized_Volatility", "Change_in_Volatility", "ATR"]].dropna()

            if dataset.empty:
                raise ValueError(f"Not enough data to perform clustering for {symbol} on {date}")

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(dataset)

            k = 3  # Number of clusters
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(scaled_data)

            dataset['Cluster'] = kmeans.labels_
            cluster_volatility = dataset.groupby('Cluster')['Realized_Volatility'].mean().sort_values()

            sorted_clusters = cluster_volatility.index
            cluster_labels = {sorted_clusters[i]: label for i, label in enumerate(['buy', 'hold', 'sell'])}
            cluster_volatility = cluster_volatility.sort_values().values

            cluster_cache[symbol] = {
                'end_date': end_date + timedelta(days=30),
                'scaler': scaler,
                'kmeans': kmeans,
                'cluster_labels': cluster_labels,
                'cluster_volatility': cluster_volatility
            }

        # Get historical data up to the given date
        data_up_to_date = historical_data[symbol].loc[:date]

        # Calculate the necessary attributes using a rolling window
        realized_volatility = data_up_to_date["Close"].pct_change().rolling(window=5).std().iloc[-1]
        change_in_volatility = \
        data_up_to_date["Close"].pct_change().rolling(window=5).std().pct_change().rolling(window=5).std().iloc[-1]
        atr = (data_up_to_date["High"] - data_up_to_date["Low"]).rolling(window=5).mean().iloc[-1]

        # Prepare the data point for prediction
        last_features = np.array([[realized_volatility, change_in_volatility, atr]])
        last_scaled_features = scaler.transform(last_features)

        # Predict the cluster for the last data point
        last_cluster = kmeans.predict(last_scaled_features)[0]
        regime = cluster_labels[last_cluster]
        avg_volatility = cluster_volatility[last_cluster]

        return regime, avg_volatility
    except Exception as e:
        print(f"An error occurred for {symbol} on {date}: {e}")
        return 'hold', np.nan


# Define the date range for the analysis
end_date = pd.Timestamp.today()  # Set end_date to today's date
start_date = end_date - timedelta(days=7 * 365)  # Start date is 7 years before the end date

# Create a date range
date_range = pd.date_range(start=start_date, end=end_date)

# Initialize an empty DataFrame to store combined results
combined_results = pd.DataFrame(index=date_range)

# Initialize a cache for clustering results
cluster_cache = {}

total_time_start = time.time()

# Loop over each date and perform analysis for each stock
for date in date_range:
    day_time_start = time.time()
    regimes = []
    avg_vols = []
    for symbol in stock_symbols:
        regime, avg_volatility = analyze_stock(symbol, date, historical_data, cluster_cache)
        regimes.append(regime)
        avg_vols.append(avg_volatility)

    # Skip dates where there is no data for any stock
    if not any(pd.isna(avg_vols)):
        combined_results.loc[date, 'Percent_Buy'] = regimes.count('buy') / len(regimes)
        combined_results.loc[date, 'Percent_Sell'] = regimes.count('sell') / len(regimes)
        for i, symbol in enumerate(stock_symbols):
            combined_results.loc[date, f'{symbol}_Regime'] = regimes[i]
            combined_results.loc[date, f'{symbol}_Avg_Volatility'] = avg_volatility

    day_time_end = time.time()
    print(f"Processed {date}: Time taken: {day_time_end - day_time_start:.2f} seconds")

total_time_end = time.time()
print(f"Total time taken: {total_time_end - total_time_start:.2f} seconds")

# Remove dates with no data
combined_results.dropna(how='all', inplace=True)

# Reorder the columns to have the Percent_Buy and Percent_Sell after the date
column_order = ['Percent_Buy', 'Percent_Sell']
for symbol in stock_symbols:
    column_order.append(f'{symbol}_Regime')
    column_order.append(f'{symbol}_Avg_Volatility')
combined_results = combined_results[column_order]

# Save the combined dataset to a CSV file
combined_results.to_csv(os.path.join(path, 'combined_regime_analysis.csv'))


