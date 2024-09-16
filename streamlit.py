import streamlit as st
import numpy as np
import scipy.stats as sc
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yfinance as yf
from datetime import date
np.random.seed(42)

st.set_page_config(
    layout="centered")

# Side bar link and select box
st.sidebar.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/dragan-pinsent-599665280/"
st.sidebar.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Dragan Pinsent`</a>', unsafe_allow_html=True)

# Initialize session state if not already set
if 'black_scholes_pricer' not in st.session_state:
    st.session_state.black_scholes_pricer = False
if 'VaR_calculator' not in st.session_state:
    st.session_state.VaR_calculator = False
if 'MVP_calculator' not in st.session_state:
    st.session_state.MVP_calculator = False

# Buttons to set state
if st.sidebar.button('Black Scholes Options Pricer'):
    st.session_state.black_scholes_pricer = True
    st.session_state.VaR_calculator = False
    st.session_state.MVP_calculator = False
if st.sidebar.button('Value At Risk Calculator'):
    st.session_state.black_scholes_pricer = False
    st.session_state.VaR_calculator = True
    st.session_state.MVP_calculator = False

if st.sidebar.button('Minimum Variance Portfolio'):
    st.session_state.black_scholes_pricer = False
    st.session_state.VaR_calculator = False
    st.session_state.MVP_calculator = True

## VAR Calculator
if st.session_state.VaR_calculator:
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
    st.markdown('### Historical and Parametic VaR')

    # Input Params
    col1, col2 = st.columns(2)
    tickers = col1.text_input('Enter tickers separated by space', 'AAPL MSFT GOOG TSLA').split()
    weights_input = col2.text_input('Enter Portfolio Weights Separated by Space', '0.25 0.25 0.25 0.25')

    col1, col2, col3, col4, col5 = st.columns(5)
    n_days = col1.number_input("Number of Days:",value = 1) 
    confidence_level = col2.number_input('Confidence Level (%):', min_value=90.0, max_value=99.9, value=95.0) / 100  
    initial_portfolio_value = col3.number_input("Initial Portfolio Value", value = 10000)  
    start_date = col4.date_input("Start Date", value=date(2022, 1, 1))
    end_date = col5.date_input("End Date", value=date.today())
    weights = np.array([float(x) for x in weights_input.split()])
    n_assets = len(tickers) 

    # Catching errors
    if not np.isclose(np.sum(weights), 1):
        st.error("The sum of portfolio weights must equal 1. Please adjust your input.")
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

        return VaR, portfolio_returns, expected_historical_shortfall
    
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

    def plot_VaR_figure(returns, VaR):
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=50, alpha=0.75, color='green', edgecolor='black')
        plt.axvline(x=VaR, color='red', linestyle='dashed', linewidth=2, label=f'VaR at 95% confidence level: {-VaR:.4%}')
        plt.title('Distribution of Portfolio Returns')
        plt.xlabel('Portfolio Returns')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    # Running the different VaR calculations

    # Historical VaR
    historical_VaR, portfolio_returns, historical_ES = historical_simulation(tickers, start_date, end_date)

    # Parametric VaR
    parametric_VaR, parametric_ES = parametric_VaR(tickers, start_date, end_date)

    # Display results in two columns
    col1, col2 = st.columns(2)
    with col1:
        plot_VaR_figure(portfolio_returns, historical_VaR)

    with col2:
        st.write(f"Historical VaR at {confidence_level * 100:.2f}% confidence: {-historical_VaR:.4%}")
        st.write(f"Expected Shortfall (ES) given Historical VaR: £{-historical_ES * initial_portfolio_value:.2f}")
        st.write(f"Parametric VaR at {confidence_level * 100:.2f}% confidence: {-parametric_VaR:.4%}")
        st.write(f"Expected Shortfall (ES) given Parametric VaR: £{-parametric_ES * initial_portfolio_value:.2f}")

    # Monte Carlo Simulation
    n_simulations = st.number_input("Number of Monte Carlo Simulations:", value=10000)
    monte_carlo_VaR, monte_carlo_returns, monte_carlo_ES = monte_carlo_simulation(tickers, start_date, end_date, n_simulations)
    # Display Monte Carlo results in two columns
    col1, col2 = st.columns(2)
    with col1:
        plot_VaR_figure(monte_carlo_returns, monte_carlo_VaR)

    with col2:
        st.write(f"Monte Carlo VaR at {confidence_level * 100:.2f}% confidence: {-monte_carlo_VaR:.4%}")
        st.write(f"Expected Shortfall (ES) given Monte Carlo VaR: £{-monte_carlo_ES * initial_portfolio_value:.2f}")

##  Black Scholes Option Pricer     
elif st.session_state.black_scholes_pricer:
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

    col1, col2 = st.columns(2)
    option_type = col1.selectbox('Select Option Type', options=['Standard European', 'Knock Out Barrier', 'Knock In Barrier'])

    # Conditionally show second box based on selection from the first dropdown
    if option_type in ['Knock Out Barrier', 'Knock In Barrier']:
        H = col2.number_input('Enter Barrier Level', value=100)
    else: 
        H = 0

    col1, col2 = st.columns(2)

    K = col1.number_input("Strike Price:", min_value = 0.001, value = 50.00, step = 0.01, format="%0.2f")
    T = col1.number_input("Time Till Expiry (Years):", min_value = 0.001, value = 2.00, step = 0.001, format="%0.3f")
    r = col2.number_input("Risk Free Interest Rate:", min_value = 0.001, value = 0.05, step = 0.001, format="%0.3f")
    Q = col2.number_input("Dividend Rate:", min_value = 0.001, value = 0.02, step = 0.001, format="%0.3f")
    V = col1.number_input("Volatility:", min_value = 0.001, value = 0.10, step = 0.001, format="%0.3f")
    S = col2.number_input("Asset Price:", min_value = 0.001, value = 50.00, step = 0.5, format="%0.2f")

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

    col1, col2, col3, col4 = st.columns(4)
    col1.write('Call Price:')
    col2.write(f"# {np.round(Call,3)}")
    col3.write('Put Price:')
    col4.write(f"# {np.round(Put,3)}")

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
    col1.write('')
    col3.write('')
    col1.write('')
    col3.write('')
    col1.write('')
    col3.write('')

    # Delta
    col1.markdown("""
    <div class="tooltip">Delta (Call)
    <span class="tooltiptext">Delta represents the sensitivity of an option's price to changes in the underlying asset's price. For a call option, it ranges from 0 to 1.</span>
    </div>""", unsafe_allow_html=True)
    col1.write(f"{Delta_call:.4f}")

    col2.markdown("""
    <div class="tooltip">Delta (Put)
    <span class="tooltiptext">Delta for a put option ranges from -1 to 0 and represents the sensitivity of the put option's price to changes in the underlying asset's price.</span>
    </div>""", unsafe_allow_html=True)
    col2.write(f"{Delta_put:.4f}")

    # Gamma
    col3.markdown("""
    <div class="tooltip">Gamma
    <span class="tooltiptext">Gamma measures the rate of change of Delta with respect to changes in the underlying asset's price. It indicates the curvature of the option's value relative to the asset's price.</span>
    </div>""", unsafe_allow_html=True)
    col3.write(f"{Gamma:.4f}")

    # Vega
    col4.markdown("""
    <div class="tooltip">Vega
    <span class="tooltiptext">Vega measures the sensitivity of the option's price to changes in the volatility of the underlying asset. It shows how much the option price will change for a 1% change in volatility.</span>
    </div>""", unsafe_allow_html=True)
    col4.write(f"{Vega:.4f}")

    # Theta
    col1.markdown("""
    <div class="tooltip">Theta (Call)
    <span class="tooltiptext">Theta measures the sensitivity of the option's price to the passage of time. It represents the change in the option price as time to maturity decreases.</span>
    </div>""", unsafe_allow_html=True)
    col1.write(f"{Theta_call:.4f}")

    col2.markdown("""
    <div class="tooltip">Theta (Put)
    <span class="tooltiptext">Theta for a put option measures the sensitivity of the option's price to the passage of time, similar to a call option but in the context of a put option's characteristics.</span>
    </div>""", unsafe_allow_html=True)
    col2.write(f"{Theta_put:.4f}")

    # Vomma
    col3.markdown("""
    <div class="tooltip">Vomma (Volga)
    <span class="tooltiptext">Vomma measures the sensitivity of Vega with respect to changes in volatility. It shows how Vega changes as the volatility of the underlying asset changes.</span>
    </div>""", unsafe_allow_html=True)
    col3.write(f"{Vomma:.4f}")

    # Vanna
    col4.markdown("""
    <div class="tooltip">Vanna
    <span class="tooltiptext">Vanna measures the sensitivity of Delta with respect to changes in volatility, or the sensitivity of Vega with respect to changes in the underlying price.</span>
    </div>""", unsafe_allow_html=True)
    col4.write(f"{Vanna:.4f}")

    # Rho
    col1.markdown("""
    <div class="tooltip">Rho (Call)
    <span class="tooltiptext">Rho measures the sensitivity of the option's price to changes in the risk-free interest rate. It represents the change in the option price for a 1% change in interest rate.</span>
    </div>""", unsafe_allow_html=True)
    col1.write(f"{Rho_call:.4f}")

    col2.markdown("""
    <div class="tooltip">Rho (Put)
    <span class="tooltiptext">Rho for a put option measures the sensitivity of the option's price to changes in the risk-free interest rate, similar to a call option but in the context of a put option's characteristics.</span>
    </div>""", unsafe_allow_html=True)
    col2.write(f"{Rho_put:.4f}")

    # Charm
    col3.markdown("""
    <div class="tooltip">Charm (Call)
    <span class="tooltiptext">Charm measures the rate of change of Delta with respect to the passage of time. It shows how Delta changes as time progresses.</span>
    </div>""", unsafe_allow_html=True)
    col3.write(f"{Charm_call:.4f}")

    col4.markdown("""
    <div class="tooltip">Charm (Put)
    <span class="tooltiptext">Charm for a put option measures the rate of change of Delta with respect to the passage of time, similar to a call option but in the context of a put option's characteristics.</span>
    </div>""", unsafe_allow_html=True)
    col4.write(f"{Charm_put:.4f}")
    
    st.markdown('### Heatmap')

    col1, col2, col3, col4 = st.columns(4)
    LV = col1.slider("Minimum Volatility:", min_value = 0.001, value = 0.05, step = 0.001, format="%0.3f")
    HV = col2.slider("Maximum Volatility:", min_value = LV, value = 0.15, step = 0.001, format="%0.3f")
    LS = col3.slider("Minimum Spot Price:", min_value = 0.001, value = 45.00, max_value= 300.00, step = 0.5, format="%0.2f")
    HS = col4.slider("Maximum Spot Price:", min_value = LS, value = 55.00, max_value = 300.00, step = 0.5, format="%0.2f")

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
        col1, col2 = st.columns(2)
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
        col2.pyplot(fig2)

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

elif st.session_state.MVP_calculator:
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

    tickers = st.text_input('Enter tickers separated by space', 'AAPL MSFT TSLA NVDA INTC QCOM AMZN AVGO AMD ADBE').split()
   
    col1, col2, col3 = st.columns(3)
    initial_portfolio_value = col1.number_input("Initial Portfolio Value", value = 10000)  
    start_date = col2.date_input("Start Date", value=date(2022, 1, 1))
    end_date = col3.date_input("End Date", value=date.today())
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

    col1, col2 = st.columns(2)
    col1.write(f"Minimum Variance Portfolio Variance: {min_var_port_var:.5f}")
    col2.write(f"Portfolio Volatility: {np.sqrt(min_var_port_var):.5f}")
    risk_free_interest = col1.number_input("Input Risk Free Interest Rate", value = 3)/ 100

    mvp_port_returns = np.dot(returns, min_var_port['Values'])
    col2.write(f"Sharpe Ratio: {(np.mean(mvp_port_returns) - risk_free_interest/365)/ min_var_port_var:.3f}")
    col2.write(f"Expected Portfolio Returns Yearly {((1 + np.mean(mvp_port_returns))**365 -1) * 100:.3f}%")

    plt.figure(figsize=(10, 8))
    plt.hist(mvp_port_returns, bins=50, alpha=0.75, color='red', edgecolor='black')
    plt.axvline(x= np.mean(mvp_port_returns), color='black', linestyle='dashed', linewidth=2, label=f'Expected Portfolio Returns Daily: {np.mean(mvp_port_returns):.4%}')
    plt.title('Distribution of Portfolio Returns')
    plt.xlabel('Portfolio Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    col1.pyplot(plt)

    # Correlation heatmap of the asset returns
    corr_matrix = returns.corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
    ax4.set_title('Correlation Matrix')
    col2.pyplot(fig4)

