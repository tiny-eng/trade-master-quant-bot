import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Database connection info
DATABASE_URL = "postgresql://postgres:123456789@localhost/trader_master"

engine = create_engine(DATABASE_URL)

# Load table
df = pd.read_sql("SELECT * FROM candle_a_minute;", engine)

# Convert timestamp
df['time'] = pd.to_datetime(df['ts'], unit='s')

# df = df.set_index('time')
# last_month = df.last('30D')

# # Plot close price
# plt.figure(figsize=(12, 5))
# plt.plot(last_month.index, last_month['c'])
# plt.title('Close Price - Last 30 Days')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.grid(True)
# plt.show()

# Filter last day (last 24 hours)
end = df['time'].max()
start = end - timedelta(days=1)

last_day = df[(df['time'] >= start) & (df['time'] <= end)]

# Plot close prices
plt.figure(figsize=(12, 5))
plt.plot(last_day['time'], last_day['c'])
plt.title('Close Price - Last 24 Hours (1-minute)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()