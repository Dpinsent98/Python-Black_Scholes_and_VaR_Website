import streamlit as st
import numpy as np
import scipy.stats as sc
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yfinance as yf
from datetime import date
import altair as alt
np.random.seed(42)

st.set_page_config(
    layout="centered")

# Side bar link and select box
linkedin_url = "https://www.linkedin.com/in/dragan-pinsent-599665280/"
st.sidebar.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Dragan Pinsent`</a>', unsafe_allow_html=True)

page = st.sidebar.radio('Page Select:', ['Black Scholes Option Pricer', 'VaR Calculator', 'Min Variance Portfolio Tool', 'Monte Carlo Options Pricer'])

prev_VaR = 0

## VAR Calculator
if page == 'VaR Calculator':
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 class="title">Value at Risk Calculator</h1>', unsafe_allow_html=True)

    # Input Params
    tickers = st.sidebar.text_input('Enter Tickers Separated by Space:', 'AAPL MSFT GOOG TSLA').split()
    weights_input = st.sidebar.text_input('Enter Portfolio Weights Separated by Space:', '0.25 0.25 0.25 0.25')

    n_days = st.sidebar.number_input("Number of Days:",value = 1) 
    confidence_level = st.sidebar.number_input('Confidence Level (%):', min_value=90.0, max_value=99.9, value=95.0) / 100  
    initial_portfolio_value = st.sidebar.number_input("Initial Portfolio Value:", value = 10000)  
    start_date = st.sidebar.date_input("Start Date:", value=date(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date:", value=date.today())
    weights = np.array([float(x) for x in weights_input.split()])
    n_assets = len(tickers) 

    st.sidebar.subheader('Monte Carlo Input:')

    # Catching errors
    if np.sum(weights) > 1:
        st.error("The sum of portfolio weights must less than or equal 1. Please adjust your input.")
    if len(weights) != len(tickers):
        st.error("Number of weights and tickers are missmatched")

    def calculate_statistics(prices):
        # Calculate percentage changes (daily returns) and remove NaN values
        returns = prices.pct_change().dropna()
        # Calculate mean and covariance matrix of returns
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        return returns, mean_returns, cov_matrix

    def historical_simulation(tickers, start_date, end_date):
        # Download adjusted close prices for the specified tickers and date range
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Calculate returns, mean returns, and covariance matrix
        returns, mean_returns, cov_matrix = calculate_statistics(prices)
        
        # Calculate portfolio returns using weights
        portfolio_returns = returns.dot(weights)
        
        # Calculate Value at Risk (VaR) at the specified confidence level
        VaR = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Calculate the Expected Shortfall (ES)
        tail_losses = portfolio_returns[portfolio_returns < VaR]
        expected_historical_shortfall = np.mean(tail_losses)
        
        beta = np.dot(cov_matrix, weights) / np.dot(weights.T, np.dot(cov_matrix,weights))
        marginal_VaR = VaR * beta/weights 

        return VaR, portfolio_returns, expected_historical_shortfall, marginal_VaR
    
    def parametric_VaR(tickers, start_date, end_date):
        # Download adjusted close prices
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Calculate statistics
        returns, mean_returns, cov_matrix = calculate_statistics(prices)
        
        # Calculate portfolio mean and standard deviation
        portfolio_mean = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
        
        # Calculate VaR using the normal distribution
        VaR = portfolio_mean - sc.norm.ppf(confidence_level) * portfolio_std
        
        # Calculate Expected Shortfall (ES) using the normal distribution
        pdf_z = sc.norm.pdf(sc.norm.ppf(confidence_level))
        expected_parametric_shortfall = portfolio_mean - (portfolio_std * pdf_z / (1 - confidence_level))

        return VaR, expected_parametric_shortfall

    def monte_carlo_simulation(tickers, start_date, end_date, n_simulations):
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

        returns, mu, cov_matrix = calculate_statistics(data)
        L = np.linalg.cholesky(cov_matrix)

        # Generate correlated asset returns
        simulated_returns = np.zeros((n_simulations, 1, n_assets))
        for i in range(n_simulations):
            Z = np.random.normal(size=(1, n_assets))
            correlated_randoms = Z @ L.T
            simulated_returns[i] = correlated_randoms * np.sqrt(1/1) + mu

        # Calculate the portfolio value for each simulation
        simulated_portfolio_values = np.zeros(n_simulations)
        initial_portfolio_value = 1  # Assume the portfolio starts with a value of 1 (normalized)
        for i in range(n_simulations):
            cumulative_returns = np.cumprod(1 + simulated_returns[i], axis=0)
            final_portfolio_value = initial_portfolio_value * np.dot(cumulative_returns[-1], weights)
            simulated_portfolio_values[i] = final_portfolio_value

        # Calculate the portfolio returns from the initial value
        simulated_portfolio_returns = simulated_portfolio_values / initial_portfolio_value - 1
        # Calculate the VaR at the given confidence level
        VaR = np.percentile(simulated_portfolio_returns, (1 - confidence_level) * 100)
        tail_loses = simulated_portfolio_returns[simulated_portfolio_returns < VaR]
        expected_shortfall = np.mean(tail_loses)

        return VaR, simulated_portfolio_returns, expected_shortfall

    def plot_VaR_figure(returns, VaR, Method):
        if Method == 'Historical':
            plt.figure(figsize=(10, 6))
            plt.hist(returns, bins=50, alpha=0.75, color='green', edgecolor='black')
            plt.axvline(x=VaR, color='red', linestyle='dashed', linewidth=2, label=f'VaR at 95% confidence level: {-VaR:.4%}')
            plt.title('Distribution of 1 Day Historical Portfolio Returns')
            plt.xlabel('Portfolio Returns')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        else:
            plt.figure(figsize=(10, 6))
            plt.hist(returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
            plt.axvline(x=VaR, color='red', linestyle='dashed', linewidth=2, label=f'VaR at 95% confidence level: {-VaR:.4%}')
            plt.title('Distribution of 1 Day Monte Carlo Simulated Portfolio Returns')
            plt.xlabel('Portfolio Returns')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)


    # Running the different VaR calculations

    # Historical VaR
    historical_VaR, portfolio_returns, historical_ES, marginal_VaR = historical_simulation(tickers, start_date, end_date)

    # Parametric VaR
    parametric_VaR, parametric_ES = parametric_VaR(tickers, start_date, end_date)

    
    plot_VaR_figure(portfolio_returns, historical_VaR, 'Historical')

    st.write(f'VaR and Expected Shortfall at {confidence_level * 100:.2f}% confidence:')

    st.write(f"Historical VaR: {-historical_VaR * np.sqrt(n_days):.4%} (£{-historical_VaR * np.sqrt(n_days)* initial_portfolio_value:.2f}), Historical Expected Shortfall: {-historical_ES* np.sqrt(n_days):.4%} (£{-historical_ES * initial_portfolio_value * np.sqrt(n_days):.2f})")
    st.write(f"Parametric VaR: {-parametric_VaR * np.sqrt(n_days):.4%} (£{-parametric_VaR * np.sqrt(n_days)* initial_portfolio_value:.2f}), Parametric Expected Shortfall: {-parametric_ES* np.sqrt(n_days):.4%} (£{-parametric_ES * initial_portfolio_value * np.sqrt(n_days):.2f})")
    st.table(np.transpose(pd.DataFrame({'Ticker': tickers, 'marginal VaR':np.round(marginal_VaR,4)})))

    # Monte Carlo Simulation
    n_simulations = st.sidebar.number_input("Number of Monte Carlo Simulations:", value=10000)
    monte_carlo_VaR, monte_carlo_returns, monte_carlo_ES = monte_carlo_simulation(tickers, start_date, end_date, n_simulations)
    # Display Monte Carlo results in two columns
    
    plot_VaR_figure(monte_carlo_returns, monte_carlo_VaR, 'Monte Carlo')

    st.write(f"Monte Carlo Simulated VaR: {-monte_carlo_VaR * np.sqrt(n_days):.4%} (£{-monte_carlo_VaR * np.sqrt(n_days)* initial_portfolio_value:.2f}), Simulated Expected Shortfall: {-monte_carlo_ES* np.sqrt(n_days):.4%} (£{-monte_carlo_ES * initial_portfolio_value * np.sqrt(n_days):.2f})")

    incremental_VaR = historical_VaR - prev_VaR
    st.sidebar.write(f'Incremental VaR: {np.round(incremental_VaR,5)}')
    prev_VaR = historical_VaR


##  Black Scholes Option Pricer     
elif page == 'Black Scholes Option Pricer':
    # Custom CSS to align the title
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit app with custom CSS class for the title
    st.markdown('<h1 class="title">Black Scholes Option Pricer</h1>', unsafe_allow_html=True)

    option_type = st.sidebar.selectbox('Select Option Type', options=['Standard European', 'Knock Out Barrier', 'Knock In Barrier'])

    # Conditionally show second box based on selection from the first dropdown
    if option_type in ['Knock Out Barrier', 'Knock In Barrier']:
        H = st.sidebar.number_input('Enter Barrier Level', value=55)
    else: 
        H = 0

    K = st.sidebar.number_input("Strike Price:", min_value = 0.001, value = 50.00, step = 0.01, format="%0.2f")
    T = st.sidebar.number_input("Time Till Expiry (Years):", min_value = 0.001, value = 2.00, step = 0.001, format="%0.3f")
    r = st.sidebar.number_input("Risk Free Interest Rate (%):", min_value = 0.00, value = 5.00, step = 0.01, format="%0.2f")/100
    Q = st.sidebar.number_input("Dividend Rate (%):", min_value = 0.00, value = 2.00, step = 0.01, format="%0.2f")/100
    V = st.sidebar.number_input("Volatility (%):", min_value = 0.00, value = 10.00, step = 0.01, format="%0.2f")/100
    S = st.sidebar.number_input("Asset Price:", min_value = 0.001, value = 50.00, step = 0.5, format="%0.2f")

    def Standard_Euro(K, T, r, Q, V, S):
        d1 = (np.log(S/K) + (r - Q + (V**2)/2)* T) / (V * np.sqrt(T))
        d2 = d1 - (V * np.sqrt(T))
        Call= S * np.exp(-Q*T) * sc.norm.cdf(d1) - K * np.exp(-r*T) * sc.norm.cdf(d2)
        Put = K * np.exp(-r*T) * sc.norm.cdf(-d2) - S * np.exp(-Q*T) * sc.norm.cdf(-d1)
        return Call, Put
    
    C, P = Standard_Euro(K, T, r, Q, V, S)
    
    if option_type == 'Knock Out Barrier' or option_type == 'Knock In Barrier':

        d1 = (np.log(S/K) + (r - Q + (V**2)/2)* T) / (V * np.sqrt(T))
        d2 = d1 - (V * np.sqrt(T))
        C = S * np.exp(-Q*T) * sc.norm.cdf(d1) - K * np.exp(-r*T) * sc.norm.cdf(d2)
        P = K * np.exp(-r*T) * sc.norm.cdf(-d2) - S * np.exp(-Q*T) * sc.norm.cdf(-d1)

        L = (r - Q + (V**2)/2)/ V**2
        y = np.log(H**2/(S*K))/ (V*np.sqrt(T)) + L*V*np.sqrt(T)
        X = np.log(S/H)/V*np.sqrt(T) + L*V*np.sqrt(T)
        Y = np.log(H/S)/V*np.sqrt(T) + L*V*np.sqrt(T)

        def Barrier_Options(K, T, r, Q, V, S, H):
            if H <= K:
                C_di = S*np.exp(-Q*T)*((H/S)**(2*L))*sc.norm.cdf(y) -K*np.exp(-r*T)*((H/S)**(2*L-2))*sc.norm.cdf(y - V*np.sqrt(T))
                C_do = C - C_di
                C_ui = C
                C_uo = 0

                P_di = -S*sc.norm.cdf(-X)*np.exp(-Q*T) + K*np.exp(-r*T)*sc.norm.cdf(-X +V*np.sqrt(T)) + S *np.exp(-Q*T)*((H/S)**(2*L))*(sc.norm.cdf(y)-sc.norm.cdf(Y)) - K *np.exp(-r*T)*(H/S)**(2*L-2)*(sc.norm.cdf(y-V*np.sqrt(T)) - sc.norm.cdf(Y-V*np.sqrt(T)))
                P_do = P - P_di
                P_uo = 0
                P_ui = P

            else:
                C_ui = S*sc.norm.cdf(X) *np.exp(-Q*T) - K*np.exp(-r*T)*sc.norm.cdf(X-V*np.sqrt(T)) - S*np.exp(-Q*T)*(H/S)**(2*L)*(sc.norm.cdf(-y) - sc.norm.cdf(-Y)) + K*np.exp(-r*T)*(H/S)**(2*L-2)*(sc.norm.cdf(-y+V*np.sqrt(T)) - sc.norm.cdf(-Y +V*np.sqrt(T)))
                C_uo = C - C_ui
                C_do = 0
                C_di = C

                P_ui = -S*sc.norm.cdf(-y)*np.exp(-Q*T)*(H/S)**(2*L) + K*np.exp(-r*T)*sc.norm.cdf(-y+V*np.sqrt(T))*(H/S)**(2*L-2) 
                P_uo = P - P_ui
                P_do = 0
                P_di = P
            return C_di, C_do, C_ui, C_uo, P_di, P_do, P_ui, P_uo
    
    

    if option_type == 'Standard European':
        Call = C
        Put = P
    elif option_type == 'Knock Out Barrier':
        C_di, C_do, C_ui, C_uo, P_di, P_do, P_ui, P_uo = Barrier_Options(K, T, r, Q, V, S, H)
        if H > K:
            Call = C_uo
            Put = P_uo
        else:
            Call = C_do
            Put = P_do
    elif option_type == 'Knock In Barrier':
        C_di, C_do, C_ui, C_uo, P_di, P_do, P_ui, P_uo = Barrier_Options(K, T, r, Q, V, S, H)
        if H > K:
            Call = C_ui
            Put = P_ui
        else:
            Call = C_di
            Put = P_di

    col1, col2 = st.columns(2)
    col1.metric(label = 'Call Price', value =  np.round(Call,3))
    col2.metric(label = 'Put Price', value =  np.round(Put,3))

    

    d1 = (np.log(S/K) + (r - Q + (V**2)/2)* T) / (V * np.sqrt(T))
    d2 = d1 - (V * np.sqrt(T))

    # Display Greeks with tooltips
    st.markdown(f"""
    <style>
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: pointer;
        border-bottom: 1px dotted black;
    }}

    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 250px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.5s;
    }}

    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    """, unsafe_allow_html=True)

    def calculate_all_greeks(S, K, T, r, Q, V):
        d1 = (np.log(S / K) + (r - Q + (V ** 2) / 2) * T) / (V * np.sqrt(T))
        d2 = d1 - V * np.sqrt(T)
        
        # Delta
        Delta_call = sc.norm.cdf(d1)
        Delta_put = sc.norm.cdf(d1) - 1

        # Gamma
        Gamma = sc.norm.pdf(d1) / (S * V * np.sqrt(T))
        
        # Vega
        Vega = S * sc.norm.pdf(d1) * np.sqrt(T)

        # Theta
        Theta_call = (-S * sc.norm.pdf(d1) * V / (2 * np.sqrt(T)) - (r - Q) * K * np.exp(-r * T) * sc.norm.cdf(d2))
        Theta_put = (-S * sc.norm.pdf(d1) * V / (2 * np.sqrt(T)) + (r - Q) * K * np.exp(-r * T) * sc.norm.cdf(-d2))
        
        # Vomma (Volga)
        Vomma = Vega * (d1 * d2 - 1) / V

        # Vanna
        Vanna = -Vega * (d1 * d2) / V

        # Rho
        Rho_call = K * T * np.exp(-r * T) * sc.norm.cdf(d2)
        Rho_put = -K * T * np.exp(-r * T) * sc.norm.cdf(-d2)

        # Charm (Delta Decay)
        Charm_call = -Gamma * (d1 * (V**2) - (r - Q) * T) / (2 * T)
        Charm_put = -Gamma * (d1 * (V**2) - (r - Q) * T) / (2 * T)

        return Delta_call, Delta_put, Gamma, Vega, Theta_call, Theta_put, Vomma, Vanna, Rho_call, Rho_put, Charm_call, Charm_put

    # Calculate all Greeks for Standard European options
    Delta_call, Delta_put, Gamma, Vega, Theta_call, Theta_put, Vomma, Vanna, Rho_call, Rho_put, Charm_call, Charm_put = calculate_all_greeks(S, K, T, r, Q, V)

    # Display Greeks with tooltips
    st.markdown(f"""
    <style>
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: pointer;
        border-bottom: 1px dotted black;
    }}

    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 250px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.5s;
    }}

    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
   
    # Delta
    col1.markdown("""
    <div class="tooltip"><b>Delta (Call/ Put):<b>
    <span class="tooltiptext">Delta represents the sensitivity of an option's price to changes in the underlying asset's price. For a call option, it ranges from 0 to 1.</span>
    </div>""", unsafe_allow_html=True)
    col1.write(f"{Delta_call:.4f}/{Delta_put:.4f}")

    # Gamma
    col3.markdown("""
    <div class="tooltip"><b>Gamma:<b>
    <span class="tooltiptext">Gamma measures the rate of change of Delta with respect to changes in the underlying asset's price. It indicates the curvature of the option's value relative to the asset's price.</span>
    </div>""", unsafe_allow_html=True)
    col3.write(f"{Gamma:.4f}")

    # Vega
    col4.markdown("""
    <div class="tooltip"><b>Vega:<b>
    <span class="tooltiptext">Vega measures the sensitivity of the option's price to changes in the volatility of the underlying asset. It shows how much the option price will change for a 1% change in volatility.</span>
    </div>""", unsafe_allow_html=True)
    col4.write(f"{Vega:.4f}")

    col2.markdown("""
    <div class="tooltip"><b>Theta (Call/ Put):<b>
    <span class="tooltiptext">Theta for a put option measures the sensitivity of the option's price to the passage of time, similar to a call option but in the context of a put option's characteristics.</span>
    </div>""", unsafe_allow_html=True)
    col2.write(f"{Theta_call:.4f}/{Theta_put:.4f}")

    # Vomma
    col3.markdown("""
    <div class="tooltip"><b>Vomma:<b>
    <span class="tooltiptext">Vomma measures the sensitivity of Vega with respect to changes in volatility. It shows how Vega changes as the volatility of the underlying asset changes.</span>
    </div>""", unsafe_allow_html=True)
    col3.write(f"{Vomma:.4f}")

    # Vanna
    col4.markdown("""
    <div class="tooltip"><b>Vanna:<b>
    <span class="tooltiptext">Vanna measures the sensitivity of Delta with respect to changes in volatility, or the sensitivity of Vega with respect to changes in the underlying price.</span>
    </div>""", unsafe_allow_html=True)
    col4.write(f"{Vanna:.4f}")

    # Rho
    col1.markdown("""
    <div class="tooltip"><b>Rho (Call/ Put):<b>
    <span class="tooltiptext">Rho measures the sensitivity of the option's price to changes in the risk-free interest rate. It represents the change in the option price for a 1% change in interest rate.</span>
    </div>""", unsafe_allow_html=True)
    col1.write(f"{Rho_call:.4f}/{Rho_put:.4f}")

    # Charm
    col2.markdown("""
    <div class="tooltip"><b>Charm (Call/ Put):<b>
    <span class="tooltiptext">Charm measures the rate of change of Delta with respect to the passage of time. It shows how Delta changes as time progresses.</span>
    </div>""", unsafe_allow_html=True)
    col2.write(f"{Charm_call:.4f}/{Charm_put:.4f}")
    
    st.sidebar.subheader('Heatmap Params')

    LV = st.sidebar.slider("Minimum Volatility:", min_value = 0.001, value = 0.05, step = 0.001, format="%0.3f")
    HV = st.sidebar.slider("Maximum Volatility:", min_value = LV, value = 0.15, step = 0.001, format="%0.3f")
    LS = st.sidebar.slider("Minimum Spot Price:", min_value = 0.001, value = 45.00, max_value= 300.00, step = 0.5, format="%0.2f")
    HS = st.sidebar.slider("Maximum Spot Price:", min_value = LS, value = 55.00, max_value = 300.00, step = 0.5, format="%0.2f")

    SH = np.linspace(start= LS,stop= HS,num = 5)
    VH = np.linspace(start= LV,stop= HV,num= 5)


    def Standard_Euro_Heatmap(K, T, r, Q, VH, SH):
        C, P = [], []

        for i in SH:
            for j in VH:

                d1 = (np.log(i/K) + (r - Q + (j**2)/2)* T) / (j * np.sqrt(T))
                d2 = d1 - (j * np.sqrt(T))

                C1= i * np.exp(-Q*T) * sc.norm.cdf(d1) - K * np.exp(-r*T) * sc.norm.cdf(d2)
                P1= K * np.exp(-r*T) * sc.norm.cdf(-d2) - i * np.exp(-Q*T) * sc.norm.cdf(-d1)

                C.append(C1)
                P.append(P1)

        C = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(C), (5,5))), columns= SH, index = VH)
        P = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(P), (5,5))), columns= SH, index = VH)
        return C, P

    C, P = Standard_Euro_Heatmap(K, T, r, Q, VH, SH)

    #
    def Barrier_Heatmap_Calculations(K, T, r, Q, VH, SH, H):
        C_di, C_do, C_ui, C_uo, P_di, P_do, P_ui, P_uo = [], [], [], [], [], [], [], []

        if H <= K:
            for i in SH:
                for j in VH:
            
                    d1 = (np.log(i/K) + (r - Q + (j**2)/2)* T) / (j * np.sqrt(T))
                    d2 = d1 - (j * np.sqrt(T))

                    C = i * np.exp(-Q*T) * sc.norm.cdf(d1) - K * np.exp(-r*T) * sc.norm.cdf(d2)
                    P = K * np.exp(-r*T) * sc.norm.cdf(-d2) - i * np.exp(-Q*T) * sc.norm.cdf(-d1)

                    L = (r - Q + (j**2)/2)/ j**2
                    y = np.log(H**2/(i*K))/ (j*np.sqrt(T)) + L*j*np.sqrt(T)

                    X = np.log(i/H)/j*np.sqrt(T) + L*j*np.sqrt(T)
                    Y = np.log(H/i)/j*np.sqrt(T) + L*j*np.sqrt(T)
                    Zero = 0

                    A1 = i*np.exp(-Q*T)*((H/i)**(2*L))*sc.norm.cdf(y) -K*np.exp(-r*T)*((H/i)**(2*L-2))*sc.norm.cdf(y - j*np.sqrt(T))
                    A2 = C - A1

                    E1 = -i*sc.norm.cdf(-X)*np.exp(-Q*T) + K*np.exp(-r*T)*sc.norm.cdf(-X +j*np.sqrt(T)) + i *np.exp(-Q*T)*((H/i)**(2*L))*(sc.norm.cdf(y)-sc.norm.cdf(Y)) - K *np.exp(-r*T)*(H/i)**(2*L-2)*(sc.norm.cdf(y-j*np.sqrt(T)) - sc.norm.cdf(Y-j*np.sqrt(T)))
                    E2 = P - E1

                    if A1 < 0:
                        A1 = 0
                    if E1 < 0:
                        E1 = 0 

                    C_ui.append(C)
                    C_uo.append(Zero)
                    C_di.append(A1)
                    C_do.append(A2)

                    P_di.append(E1)
                    P_do.append(E2)
                    P_uo.append(Zero)
                    P_ui.append(P)

        else:
            for i in SH:
                for j in VH:
            
                    d1 = (np.log(i/K) + (r - Q + (j**2)/2)* T) / (j * np.sqrt(T))
                    d2 = d1 - (j * np.sqrt(T))

                    C = i * np.exp(-Q*T) * sc.norm.cdf(d1) - K * np.exp(-r*T) * sc.norm.cdf(d2)
                    P = K * np.exp(-r*T) * sc.norm.cdf(-d2) - i * np.exp(-Q*T) * sc.norm.cdf(-d1)

                    L = (r - Q + (j**2)/2)/ j**2
                    y = np.log(H**2/(i*K))/ (j*np.sqrt(T)) + L*j*np.sqrt(T)

                    X = np.log(i/H)/j*np.sqrt(T) + L*j*np.sqrt(T)
                    Y = np.log(H/i)/j*np.sqrt(T) + L*j*np.sqrt(T)
                    Zero = 0

                    B1 = i*sc.norm.cdf(X) *np.exp(-Q*T) - K*np.exp(-r*T)*sc.norm.cdf(X-j*np.sqrt(T)) - i*np.exp(-Q*T)*(H/i)**(2*L)*(sc.norm.cdf(-y) - sc.norm.cdf(-Y)) + K*np.exp(-r*T)*(H/i)**(2*L-2)*(sc.norm.cdf(-y+j*np.sqrt(T)) - sc.norm.cdf(-Y +j*np.sqrt(T)))
                    B2 = C - B1

                    E1 = -i*sc.norm.cdf(-y)*np.exp(-Q*T)*(H/i)**(2*L) + K*np.exp(-r*T)*sc.norm.cdf(-y+j*np.sqrt(T))*(H/i)**(2*L-2) 
                    E2 = P - E1

                    if B1 < 0:
                        B1 = 0
                    if E1 < 0:
                        E1 = 0 

                    C_di.append(C)
                    C_do.append(Zero)
                    C_ui.append(B1)
                    C_uo.append(B2)

                    P_ui.append(E1)
                    P_uo.append(E2)
                    P_do.append(Zero)
                    P_di.append(P)
        
        C_di = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(C_di), (5,5))), columns= SH, index = VH)
        C_do = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(C_do), (5,5))), columns= SH, index = VH)
        C_ui = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(C_ui), (5,5))), columns= SH, index = VH)
        C_uo = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(C_uo), (5,5))), columns= SH, index = VH)
        P_di = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(P_di), (5,5))), columns= SH, index = VH)
        P_do = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(P_do), (5,5))), columns= SH, index = VH)
        P_ui = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(P_ui), (5,5))), columns= SH, index = VH)
        P_uo = pd.DataFrame(np.transpose(np.reshape(pd.DataFrame(P_uo), (5,5))), columns= SH, index = VH)
        return C_di, C_do, C_ui, C_uo, P_ui, P_uo, P_do, P_di

    C_di, C_do, C_ui, C_uo, P_ui, P_uo, P_do, P_di = Barrier_Heatmap_Calculations(K, T, r, Q, VH, SH, H)

    def options_heatmap(call_option, put_option):
        emp1, col1, emp2 =st.columns([1,5,1])
        fig1, ax1 = plt.subplots()
        sns.heatmap(call_option, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax1, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
        ax1.set_xlabel('Spot Price')
        ax1.set_ylabel('Volatility')
        ax1.set_title('Call Option Price')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
        col1.pyplot(fig1)

        # Heatmap for Put Option Price
        fig2, ax2 = plt.subplots()
        sns.heatmap(put_option, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax2, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
        ax2.set_xlabel('Spot Price')
        ax2.set_ylabel('Volatility')
        ax2.set_title('Put Option Price')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        col1.pyplot(fig2)

    if option_type == 'Standard European':
        options_heatmap(C,P)

    elif option_type == 'Knock Out Barrier':
        if H < K:
            options_heatmap(C_do,P_do)

        elif H >= K:
            options_heatmap(C_uo,P_uo)

    elif option_type == 'Knock In Barrier':
        if H < K:
            options_heatmap(C_di,P_di)

        elif H >= K:
            options_heatmap(C_ui,P_ui)

elif page == 'Min Variance Portfolio Tool':
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit app with custom CSS class for the title
    st.markdown('<h1 class="title">Minimum Portfolio Variance Tool</h1>', unsafe_allow_html=True)

    tickers = st.sidebar.text_input('Enter Tickers Separated by Space:', 'AAPL MSFT TSLA NVDA INTC QCOM AMZN AVGO AMD ADBE').split()
   
    initial_portfolio_value = st.sidebar.number_input("Initial Portfolio Value:", value = 10000)  
    start_date = st.sidebar.date_input("Start Date:", value=date(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date:", value=date.today())
    risk_free_interest = st.sidebar.number_input("Input Risk Free Interest Rate (%):", value = 3)/ 100
    n_assets = len(tickers) 

    def calculate_statistics(prices):
    # Calculate percentage changes (daily returns) and remove NaN values
        returns = prices.pct_change().dropna()
        # Calculate mean and covariance matrix of returns
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        return returns, mean_returns, cov_matrix

    prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
    # Calculate returns, mean returns, and covariance matrix
    returns, mean_returns, cov_matrix = calculate_statistics(prices)

    inv_cov_matrix = np.linalg.inv(cov_matrix)
    ones = np.ones(cov_matrix.shape[0])
    min_var_port = (np.dot(inv_cov_matrix , ones) / (np.dot(np.dot(np.transpose(ones) , inv_cov_matrix) , ones)))
    min_var_port_var = np.dot(np.dot(np.transpose(min_var_port) , cov_matrix) , min_var_port)

    min_var_port = pd.DataFrame({'Stock': tickers, 'Values': np.round(min_var_port,5)})

    st.write(min_var_port.T)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Minimum Variance:", np.round(min_var_port_var, 5))
    col2.metric("Portfolio Volatility:", np.round(np.sqrt(min_var_port_var),5))


    mvp_port_returns = np.dot(returns, min_var_port['Values'])
    col3.metric("Sharpe Ratio:", np.round((np.mean(mvp_port_returns) - risk_free_interest/365)/ min_var_port_var,5))
    col4.metric("Portfolio Returns Yearly (%):", np.round(((1 + np.mean(mvp_port_returns))**365 -1) * 100, 3))

    plt.figure(figsize=(10, 8))
    plt.hist(mvp_port_returns, bins=50, alpha=0.75, color='red', edgecolor='black')
    plt.axvline(x= np.mean(mvp_port_returns), color='black', linestyle='dashed', linewidth=2, label=f'Expected Portfolio Returns Daily: {np.mean(mvp_port_returns):.4%}')
    plt.title('Distribution of Portfolio Returns')
    plt.xlabel('Portfolio Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Correlation heatmap of the asset returns
    corr_matrix = returns.corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
    ax4.set_title('Correlation Matrix')
    st.pyplot(fig4)

elif page == 'Monte Carlo Options Pricer':
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit app with custom CSS class for the title
    st.markdown('<h1 class="title">Monte Carlo Option Pricer</h1>', unsafe_allow_html=True)

    options_type = st.sidebar.selectbox('Option Type', ['Standard European', 'Binary Option', 'Asian Floating Strike', 
                                                        'Asian Fixed Stike', '## Below Not Finished##','Knock Out Option', 'Knock In Options', 
                                                        'Double Knock Out', 'Double Knock In', 'Look Back Option', 
                                                        'Knock In and Knock Out Option', 'Gap Option', 'Choser Option'])

    variance_proceedure = st.sidebar.selectbox('Variance Reducton Proceedure', ['None','Antithetic Variates', 'Moment Matching','## Below Not Finished ##','Importance Sampling', 'Quasi Random Sequence'])

    distribution = st.sidebar.selectbox('Assumed Distribution', ['Normal', 't-distribution'])
    if distribution == 't-distribution':
        dof = st.sidebar.number_input('Degrees of Freedom', min_value= 1, value = 5)

    if options_type == 'Binary Option':
        Binary_payout = st.sidebar.number_input('Binary Payout' , min_value = 0, value = 10)


    Strike = st.sidebar.number_input("Strike Price:", min_value = 0.001, value = 50.00, step = 0.01, format="%0.2f")
    T = st.sidebar.number_input("Time Till Expiry (Years):", min_value = 0.001, value = 2.00, step = 0.001, format="%0.3f")
    r = st.sidebar.number_input("Risk Free Interest Rate (%):", min_value = 0.00, value = 5.00, step = 0.01, format="%0.2f")/100
    Vol = st.sidebar.number_input("Volatility (%):", min_value = 0.00, value = 10.00, step = 0.01, format="%0.2f")/100
    Stock = st.sidebar.number_input("Asset Price:", min_value = 0.001, value = 50.00, step = 0.5, format="%0.2f")
    NLoops= st.sidebar.number_input('Number of Loops', min_value = 100, value = 100, step = 2)
    NSteps= st.sidebar.number_input('Number of Steps', min_value = 1, value = 100) + 1

    check_box = st.sidebar.checkbox('Would You Like Graphics? (Not Recommended for Large Simulations)', value=True)

    Button = st.sidebar.button('Run Simulation')

    def standard_sim(distribution, dof):
        monte_sim = np.zeros(shape = (NLoops, NSteps))

        if distribution == 'Normal':
            for loop in range(NLoops):
                monte_sim[loop, 0] = Stock
                for step in range(1,NSteps):
                    dt = T / NSteps  # Time increment
                    random_normal = np.random.normal(0, 1)  # Generate a standard normal random variable
                    monte_sim[loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * random_normal))
            return monte_sim

        else:
            for loop in range(NLoops):
                monte_sim[loop, 0] = Stock
                for step in range(1,NSteps):
                    dt = T / NSteps  # Time increment
                    random_t = np.random.standard_t(dof)  # Generate a standard normal random variable
                    monte_sim[loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * random_t))
            return monte_sim
        
    def antithetic_method(distribition, dof):
        monte_sim = np.zeros(shape = (NLoops, NSteps))

        if distribution == 'Normal':
            for loop in range(0,NLoops,2):
                monte_sim[loop, 0] = Stock
                for step in range(1,NSteps):
                    dt = T / NSteps  # Time increment
                    random_normal = np.random.normal(0, 1)  # Generate a standard normal random variable
                    monte_sim[loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * random_normal))
                    monte_sim[1 + loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * -random_normal))
            return monte_sim

        else:
            for loop in range(0,NLoops,2):
                monte_sim[loop, 0] = Stock
                for step in range(1,NSteps):
                    dt = T / NSteps  # Time increment
                    random_t = np.random.standard_t(dof)  # Generate a standard normal random variable
                    monte_sim[loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * random_t))
                    monte_sim[1 + loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * -random_t))
            return monte_sim
        
    def moment_matching(distribution, dof):
        monte_sim = np.zeros(shape = (NLoops, NSteps))
        matched_sim = np.zeros(shape = (NLoops, NSteps))
        rand_samples = np.zeros(shape = (NLoops, NSteps))

        if distribution == 'Normal':
            for loop in range(NLoops):
                monte_sim[loop, 0] = Stock
                matched_sim[loop, 0] = Stock
                for step in range(1,NSteps):
                    dt = T / NSteps  # Time increment
                    rand_samples[loop, step] = np.random.normal(0, 1)  # Generate a standard normal random variable
                    monte_sim[loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * rand_samples[loop, step]))
            rand_mean = np.mean(rand_samples, axis = 1)
            rand_std = np.std(rand_samples, axis = 1)
            
            for loop in range(NLoops):
                for step in range(1,NSteps):
                    matched_sim[loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * (rand_samples[loop, step] - rand_mean[loop])/rand_std[loop]))
            return matched_sim

        else:
            for loop in range(NLoops):
                monte_sim[loop, 0] = Stock
                for step in range(1,NSteps):
                    dt = T / NSteps  # Time increment
                    rand_samples[loop, step] = random_t = np.random.standard_t(dof)(0, 1)  # Generate a standard normal random variable
                    monte_sim[loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * rand_samples[loop, step]))
            rand_mean = np.mean(rand_samples, axis = 1)
            rand_std = np.std(rand_samples, axis = 1)

            for loop in range(NLoops):
                for step in range(1,NSteps):
                    matched_sim[loop, step] = (monte_sim[loop, step - 1] *
                                            np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * (rand_samples[loop, step] - rand_mean[loop])/rand_std[loop]))
            return matched_sim
        
        def sobol_sim(distribution, dof):
            onte_sim = np.zeros(shape = (NLoops, NSteps))

            if distribution == 'Normal':
                for loop in range(NLoops):
                    monte_sim[loop, 0] = Stock
                    for step in range(1,NSteps):
                        dt = T / NSteps  # Time increment
                        random_normal = np.random.normal(0, 1)  # Generate a standard normal random variable
                        monte_sim[loop, step] = (monte_sim[loop, step - 1] *
                                                np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * random_normal))
                return monte_sim

            else:
                for loop in range(NLoops):
                    monte_sim[loop, 0] = Stock
                    for step in range(1,NSteps):
                        dt = T / NSteps  # Time increment
                        random_t = np.random.stmandard_t(dof)  # Generate a standard normal random variable
                        monte_sim[loop, step] = (monte_sim[loop, step - 1] *
                                                np.exp((r - (Vol ** 2) / 2) * dt + Vol * np.sqrt(dt) * random_t))
            return monte_sim

    col1, col2 = st.columns(2)

    if Button == True:

        if variance_proceedure == 'None':
            
            if distribution == 'Normal':
                monte_sim = standard_sim('Normal',0)
            else:
                monte_sim = standard_sim('t', dof)
        
        elif variance_proceedure == 'Antithetic Variates':
            if distribution == 'Normal':
                monte_sim = antithetic_method('Normal',0)
            else:
                monte_sim = antithetic_method('t', dof)

        elif variance_proceedure == 'Moment Matching':
            if distribution == 'Normal':
                monte_sim = moment_matching('Normal',0)
            else:
                monte_sim = antithetic_method('t', dof)
                
        final_price = monte_sim[:,-1]

        if options_type == 'Standard European':
            call_payoffs = np.maximum(final_price - Strike, 0)
            put_payoffs = np.maximum(Strike - final_price, 0)

            call_option_price = np.exp(-r * T) * np.mean(call_payoffs)
            put_option_price = np.exp(-r * T) * np.mean(put_payoffs)


            col1.metric('Estimated Call Option Price:', np.round(call_option_price,2))
            col2.metric('Estimated Put Option Price:', np.round(put_option_price,2))

        elif options_type == 'Binary Option':

            Call_Payout = []
            Put_Payout = []
            
            for price in final_price:
                if price > Strike:
                    Call_Payout.append(Binary_payout)
                    Put_Payout.append(0)
                else:
                    Call_Payout.append(0)
                    Put_Payout.append(Binary_payout)

            binary_call_price = np.exp(-r*T) * np.mean(Call_Payout)
            binary_put_price = np.exp(-r*T) * np.mean(Put_Payout)

            col1.metric(f'Estimated Binary Call Option with Payout {Binary_payout}', np.round(binary_call_price,2))
            col2.metric(f'Estimated Binary Put Option with Payout {Binary_payout}', np.round(binary_put_price,2))

        elif options_type == 'Asian Floating Strike':

            AsiaCallPayout = []
            AsiaPutPayout = []
            i = 0

            for price in final_price:
            
                sim_mean = monte_sim.mean(axis = 1)[i]
            
                if price > sim_mean:
                    AsiaCallPayout.append(price- sim_mean)
                    AsiaPutPayout.append(0)
                    i += 1
                else:
                    AsiaCallPayout.append(0)
                    AsiaPutPayout.append(sim_mean - price)
                    i += 1

            moving_strike_call_price = np.exp(-r *T) *np.mean(AsiaCallPayout)
            moving_strike_put_price = np.exp(-r *T) *np.mean(AsiaPutPayout)

            #moving_stock_call_price = np.exp(-r *T) * np.mean(average_stock_call)

            col1.metric('Estimated Moving Strike Call Price:', np.round(moving_strike_call_price,2))
            col2.metric('Estimated Moving Stock Put Price:', np.round(moving_strike_put_price,2))

        elif options_type == 'Asian Fixed Strike':

            AsiaCallPayout = []
            AsiaPutPayout = []

            sim_mean = monte_sim.mean(axis = 1)[i]

            fixed_strike_call_pay = np.exp(-r*T)*np.mean(np.maximum(sim_mean - Stike, 0))
            fixed_strike_put_pay = np.exp(-r*T)*np.mean(np.maximum(Strike - sim_mean, 0))

            col1.metric('Estimated Fixed Strike Call Price:', np.round(fixed_strike_call_pay,2))
            col2.metric('Estimated Fixed Stock Put Price:', np.round(fixed_strike_put_pay,2))

        # Prepare data for Altair
        time_steps = np.arange(NSteps)
        simulations_df = pd.DataFrame(monte_sim.T, columns=[f"Sim {i+1}" for i in range(NLoops)])
        simulations_df['Time'] = time_steps
        simulations_long_df = simulations_df.melt(id_vars=['Time'], var_name='Simulation', value_name='Price')

        if check_box == True:

            # Create the Altair line chart for Monte Carlo simulations
            chart = alt.Chart(simulations_long_df).mark_line(opacity=0.7).encode(
                x='Time',
                y= alt.Y('Price', scale = alt.Scale(domain = [np.min(monte_sim) - 5, np.max(monte_sim) + 5])),
            color=alt.Color('Simulation:N', legend=None)
            ).properties(
                width=800,
                height=400,
                title="Monte Carlo Simulation of Asset Prices"
            )

            # Display the Altair chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

            # Assuming final_price is your list of final prices from the Monte Carlo simulation
            final_price_df = pd.DataFrame(final_price, columns=["Final Price"])

            # Create a bar chart for the distribution of final prices
            chart = alt.Chart(final_price_df).mark_bar(color = 'red').encode(
                x=alt.X('Final Price:Q', bin=alt.Bin(maxbins=50)),  # Create bins for the final prices
                y='count()'
            ).properties(
                title="Distribution of Final Prices"
            )

            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)