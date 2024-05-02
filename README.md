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
<samp>
  <img width="745" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/2a5e321a-3ec0-4a90-9ae9-48ef3efc0482">

</samp>

# 2. Download Adjusted Close Prices
<h2>Create an empty DataFrame to store the adjusted close prices</h2>
<pre>adj_close_df = pd.DataFrame()</pre>
<h2>Download the close prices for each ticker</h2>
<pre>for ticker in tickers:
  data = yf.download(ticker, start = start_date, end = end_date)
  adj_close_df[ticker] = data['Adj Close']</pre>
<samp>
  <img width="521" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/657ce7f8-aee0-4e12-afab-9e8ef4fd9e2f">

</samp>
<h2>Display the DataFrame</h2>
<pre>print(adj_close_df)</pre>
<samp>
  <img width="579" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/dcb0e231-0fe5-408c-a914-43a9aaa9a491">

</samp>

# 3. Calculate Lognormal Returns
<h2>Calculate the lognormal returns for each ticker</h2>
<pre>log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns</pre>
<samp>
  <img width="624" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/5ac30e9c-7fbb-4f1f-a143-4d185961c9f9">

</samp>

# 4. Calculate Covariance Matrix
<h2>Calculate the covariance matrix using annualized log returns</h2>
<pre>cov_matrix = log_returns.cov()*252
cov_matrix</pre>
<samp>
  <img width="628" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/5f2a1ed1-b874-454b-8afd-09c2fa250b7f">
</samp>

# 5. Define Portfolio Performance Metrics
<h2>Calculate the portfolio standard deviation</h2>
<p>(This calculates the portfolio variance, which is a measure of the risk associated with a portfolio of assets. It represents the combined volatility of the assets in the portfolio.)</p>
<pre>def standard_deviation (weights, cov_matrix):
  variance = weights.T @ cov_matrix @ weights
  return np.sqrt(variance)</pre>

<h2>Calculate the expected return</h2>
<p>(Key Assumption: Expected returns are based on historical returns)</p>
<pre>def expected_return (weights, log_returns):
  return (np.sum(log_returns.mean() * weights) * 252)</pre>

<h2>Calculate the Share Ratio</h2>
<pre>def sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
  return(expected_return (weights, log_returns) - risk_free_rate)/standard_deviation (weights, cov_matrix)</pre>
  
# 6. Portfolio Optimization
<h2>Set the risk-free rate</h2>

(We can also use FRED api)

<pre>risk_free_rate = 0.02</pre>
<h2>Define the function to minimize (negative Sharpe Ratio)</h2>
In the case of the scipy.optimize() function, there is no direct method to find the maximum value of a function.
<pre>
  def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
  return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
</pre>

<h2>Set the constraints and bounds</h2>
<div>Constraints are conditions that must be met by the solution during the optimization process. In this case, the constraints is that the sum of all portfolio weights must be equal to 1. The constraints variable is a dictionary with two keys: 'type and 'fun'. 'type' is set to 'eq' which means "equality constraint" and 'fun' is assigned the function check_sum, which checks if the sum of the portfolio weights equals 1.

Bounds are the limits placed on the variables during the optimization process. In this case, the variables are the portfolio weights, and each weight should be between 0 and 1.

We cannot do short-sell here, we can only go long.</div>
<pre>
constraints = {'type':'eq','fun': lambda weights: np.sum(weights) - 1}
#bounds = [(0, 0.5) for _ in range(len(tickers))]
bounds = [(0.1, 0.5) for _ in range(len(tickers))]
</pre>
<h2>Set the initial weights</h2>
<pre>initial_weights = np.array([1/len(tickers)]*len(tickers))
print(initial_weights)</pre>
<samp>
  <img width="448" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/6c21e821-0c1f-4587-92b8-76ca0775670a">
</samp>
<h2>Optimize the weights to maximize Sharpe Ratio</h2>
<div>'SLSQP' stands for Sequential Least Squares Quadratic Programming, which is a numerical optimization technique suitable for solving non-linear optimization</div>
<pre>optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
optimized_results</pre>
<samp>
  <img width="784" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/37f378a8-136d-4042-82d4-bff9e8ec52be">
</samp>
<h2>Get the optimal weights</h2>
<pre>optimal_weights = optimized_results.x</pre>

# 7. Analyze the Optimal Portfolio
<h2>Display analytics of the optimal portfolio</h2>
<pre>
  print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
  print(f"{ticker}:{weight}")

print()

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return}")
print(f"Expected Volatility: {optimal_portfolio_volatility}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio}")
</pre>
<samp>
<img width="678" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/6baea130-3925-4aff-a1cc-253e8735fa6b">
</samp>
<h2>Display the final portfolio in a plot</h2>
<pre>
import matplotlib.pyplot as plt

#Create a abar chart of the optimal weights
plt.figure(figsize = (10,6))
plt.bar(tickers, optimal_weights)

#Add labels and a title
plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

#Display the chart
plt.show()  
</pre>
<samp>
<img width="714" alt="image" src="https://github.com/anuragprasad95/Portfolio_optimization/assets/3609255/06aaacc6-ed06-43dd-9a20-ec67d8eb1ad8">
</samp>
