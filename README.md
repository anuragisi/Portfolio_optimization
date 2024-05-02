# Portfolio_optimization To Boost Fianancial Performance

<div>

# Aim: 
Assume we have a portfolio full of risky assets. To find optimal portfolio which gives highest return with lowest risk.
</div>
<div>

# Methodology: 
Sharpe Ratio = (Portfolio Return - Risk Free Rate)/Standard Deviation
</div>

<pre>
import numpy as np
import pandas as pd
import yfinance as yf 
from scipy.optimize import minimize
from datetime import datetime, timedelta
</pre>

# 1. Define Tickers and Time Range

<h2>Define the list of tickers</h2>
<pre>
  tickers = ['HDFCBANK.NS','SBIN.NS','FEDERALBNK.NS','ICICIBANK.NS','AXISBANK.NS']
</pre>

<h2>Set the end date to today</h2>
<pre>
  end_date = datetime.today()
</pre>

<h2>Set the start date to 5 years ago</h2>
<pre>
start_date = end_date - timedelta(days = 5 * 365)
print(start_date)
</pre>

