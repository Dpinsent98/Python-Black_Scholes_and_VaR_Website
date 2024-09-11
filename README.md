# Financial Dashboard 

This application is a Streamlit-based dashboard for calculating financial metrics, including:

Value at Risk (VaR) Calculator:
Calculates VaR using both Historical Simulation and Monte Carlo methods.
Black-Scholes Option Pricer:
Allows users to price European, Knock-In, and Knock-Out Barrier options using the Black-Scholes model.
Features
VaR Calculator:

* Historical VaR calculation using portfolio returns.
* Monte Carlo VaR simulation based on correlated random returns.
* Customizable parameters: tickers, portfolio weights, confidence level, initial portfolio value, start/end date, number of simulations.

Black-Scholes Option Pricer:

* Price Standard European options and Barrier options.
* Customizable parameters: strike price, time to expiry, interest rate, volatility, asset price, and dividend yield.
* Outputs Call/Put option prices along with Greeks like Delta, Gamma, Theta, Vega, and Rho.
* Heatmaps for visualizing option prices across a range of volatilities and asset prices.

Prerequisites
Ensure that the following Python libraries are installed:
streamlit yfinance numpy pandas seaborn matplotlib scipy

How to Run
Clone the repository and navigate to the project directory.
Run the following command to start the Streamlit app:
streamlit run app.py

Open the URL provided in your terminal, and the dashboard will be ready to use.
Usage

Value at Risk Calculator:

Navigate to the "VaR Calculator" from the sidebar.
Input the stock tickers, portfolio weights, and other parameters.
View the VaR results for both Historical and Monte Carlo methods along with visualizations.

Black-Scholes Pricer:

Navigate to "Black Scholes Pricer" from the sidebar.
Input the desired parameters like strike price, volatility, and time to expiry.
View the Call and Put prices, along with detailed heatmaps for different volatility and spot price scenarios.
