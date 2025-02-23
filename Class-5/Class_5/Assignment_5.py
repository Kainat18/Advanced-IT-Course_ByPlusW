import os
import glob
import shutil
import pandas as pd

# Move all CSV files to a backup folder

csv_files = glob.glob("*.csv")
for file in csv_files:
 shutil.move(file, "backup_folder/")
 print(f"Moved file: {file}")

# Automating Export
def export_data(df, filename, format):
 if format == "csv":
  df.to_csv(filename, index=False)
  print(f"Data exported to {filename} in CSV format.")
 elif format == "json":
    df.to_json(filename, orient="records")
    print(f"Data exported to {filename} in JSON format.")
 else:
  print("Unsupported format.")

# Example usage:
# Creating a sample dataframe
data = {'Name': ['Alice', 'Bob', 'Charlie'],
'Age': [25, 30, 35],
'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)
# Exporting to CSV
export_data(df, "output.csv", "csv")
# Exporting to JSON
export_data(df, "output.json", "json")


# 2. Real-Time Stock Market Data Collection and Analysis Using Python and SQLite

import yfinance as yf
import sqlite3
import pandas as pd
import time

# Database setup
db_name = "stocks.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS stock_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER
)''')
conn.commit()

# Function to fetch stock data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d", interval="1m")

        if data.empty:
            print(f"No data found for {symbol}. Skipping...")
            return None  # Return None if no data is available

        latest = data.iloc[-1]  # Get the most recent price data
        return {
            "symbol": symbol,
            "open": latest["Open"],
            "high": latest["High"],
            "low": latest["Low"],
            "close": latest["Close"],
            "volume": latest["Volume"]
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Function to store data in SQLite
def store_data(symbol):
    try:
        stock_data = fetch_stock_data(symbol)
        if stock_data:  # Only store if data is available
            cursor.execute('''INSERT INTO stock_data (symbol, open, high, low, close, volume)
                              VALUES (?, ?, ?, ?, ?, ?)''',
                           (stock_data["symbol"], stock_data["open"],
                            stock_data["high"], stock_data["low"],
                            stock_data["close"], stock_data["volume"]))
            conn.commit()
            print(f"Stored data for {symbol}")
    except Exception as e:
        print(f"Error storing data for {symbol}: {e}")

# Function to analyze stock data
def analyze_stock(symbol):
    try:
        df = pd.read_sql_query(
            "SELECT * FROM stock_data WHERE symbol=? ORDER BY timestamp DESC LIMIT 100",
            conn,
            params=(symbol,)
        )
        print(df)
    except Exception as e:
        print(f"Error analyzing data for {symbol}: {e}")

# Example Usage
symbol = "NVDA"  # Apple stock

for _ in range(5):  # Fetch data 5 times with intervals
    store_data(symbol)
    analyze_stock(symbol)
    time.sleep(60)  # Wait for 1 minute before fetching again

# Close database connection safely
if conn:
    conn.close()


#Augmented Reality Transformation – Perform linear algebra operations like scaling, rotation, and translation.
import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "http://books.toscrape.com/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_books(url):
 response = requests.get(url, headers=HEADERS)
 soup = BeautifulSoup(response.text, "html.parser")
 books = soup.find_all("article", class_="product_pod")
 book_list = []
 for book in books:
  title = book.h3.a["title"]
  price = book.find("p", class_="price_color").text
  stock = book.find("p", class_="instock availability").text.strip()
  book_list.append({"Title": title, "Price": price, "Availability":
  stock})
 return book_list

books_data = get_books(URL)
df = pd.DataFrame(books_data)
df.to_csv("books.csv", index=False)
print("Data saved to books.csv")