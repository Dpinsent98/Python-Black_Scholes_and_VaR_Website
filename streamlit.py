import streamlit as st
import numpy as np
import scipy.stats as sc
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yfinance as yf
from datetime import date

st.set_page_config(
    layout="centered")

st.sidebar.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/dragan-pinsent-599665280/"
st.sidebar.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Dragan Pinsent`</a>', unsafe_allow_html=True)
option = st.sidebar.selectbox('Page',('Black Scholes Pricer', 'VaR Calculator'))

if option == 'VaR Calculator':
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

    st.markdown('### Historical VaR')

    col1, col2 = st.columns(2)
    tickers = col1.text_input('Enter tickers separated by space', 'AAPL MSFT GOOG').split()
    weights_input = col2.text_input('Enter Portfolio Weights Separated by Space', '0.33 0.33 0.34')

    col1, col2, col3, col4, col5 = st.columns(5)
    n_days = col1.number_input("Number of Days:",value = 1) 
    confidence_level = col2.number_input('Confidence Level (%):', min_value=90.0, max_value=99.9, value=95.0) / 100  
    initial_portfolio_value = col3.number_input("Initial Portfolio Value", value = 10000)  
    start_date = col4.date_input("Start Date", value=date(2022, 1, 1))
    end_date = col5.date_input("End Date", value=date.today())
    weights = np.array([float(x) for x in weights_input.split()])
    
    # Ensure that the sum of weights is 1
    if not np.isclose(np.sum(weights), 1):
        st.error("The sum of portfolio weights must equal 1. Please adjust your input.")

    np.random.seed(42)
    n_assets = len(tickers)           # Number of assets in the portfolio  

    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    def calculate_statistics(prices):
        returns = prices.pct_change().dropna()
        mu = returns.mean().values
        cov_matrix = returns.cov().values
        return returns, mu, cov_matrix
    
    def Historical_simulation(tickers, start_date, end_date):

        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

        returns, mu, cov_matrix = calculate_statistics(data)

        port_returns = returns.dot(weights)

        VaR = np.percentile(port_returns, (1 - confidence_level) * 100)

        return VaR, port_returns
    
    His_VaR, mu = Historical_simulation(tickers, start_date, end_date)

    plt.figure(figsize=(10, 6))
    plt.hist(mu, bins=50, alpha=0.75, color='green', edgecolor='black')
    plt.axvline(x=His_VaR, color='red', linestyle='dashed', linewidth=2, label=f'VaR at 95% confidence level: {-His_VaR:.4%}')
    plt.title('Distribution of 1 Day Simulated Portfolio Returns')
    plt.xlabel('Portfolio Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.write(f"Value at Risk (VaR) at {np.round(confidence_level * 100,2)}% confidence level: {-His_VaR * np.sqrt(n_days):.4%}, Capital at Risk £{np.round(-His_VaR * initial_portfolio_value* np.sqrt(n_days),2)}.")

    st.markdown('### Monte Carlo Simulation VaR')

    n_simulations = st.number_input("Number of Monte Carlo Simulations:",value = 1000)  
    
    def Monte_simulation(tickers, start_date, end_date):

        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

        returns, mu, cov_matrix = calculate_statistics(data)
        L = np.linalg.cholesky(cov_matrix)

        # Generate correlated asset returns
        simulated_returns = np.zeros((n_simulations, n_days, n_assets))
        for i in range(n_simulations):
            Z = np.random.normal(size=(n_days, n_assets))
            correlated_randoms = Z @ L.T
            simulated_returns[i] = correlated_randoms * np.sqrt(1/n_days) + mu

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

        return VaR, simulated_portfolio_returns 

    VaR, simulated_portfolio_returns = Monte_simulation(tickers, start_date, end_date)

    plt.figure(figsize=(10, 6))
    plt.hist(simulated_portfolio_returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(x=VaR, color='red', linestyle='dashed', linewidth=2, label=f'VaR at 95% confidence level: {-VaR:.4%}')
    plt.title('Distribution of 1 Day Historical Portfolio Returns')
    plt.xlabel('Portfolio Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.write(f"Value at Risk (VaR) at {np.round(confidence_level * 100,2)}% confidence level: {-VaR * np.sqrt(n_days):.4%}, Capital at Risk £{np.round(-VaR * initial_portfolio_value* np.sqrt(n_days),2)}.")
        
else:
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

    Call, Put = Standard_Euro(K, T, r, Q, V, S)

    col1, col2, col3, col4 = st.columns(4)
    col1.write('Call Price:')
    col2.write(f"# {np.round(Call,3)}")
    col3.write('Put Price:')
    col4.write(f"# {np.round(Put,3)}")

    d1 = (np.log(S/K) + (r - Q + (V**2)/2)* T) / (V * np.sqrt(T))
    d2 = d1 - (V * np.sqrt(T))


    Delta = sc.norm.cdf(d1) 
    N1D = np.exp(-(d1**2)/2)/np.sqrt(2*np.pi)
    Gamma = N1D  / (S*V*np.sqrt(T))
    CTheta = (-(S *N1D * V)/ (2*np.sqrt(T)) - r*K*np.exp(-r*T)* sc.norm.cdf(d2))/365
    PTheta = (-(S *N1D * V)/ (2*np.sqrt(T)) + r*K*np.exp(-r*T)* sc.norm.cdf(-d2))/365
    CRho = K*T*np.exp(-r*T) * sc.norm.cdf(d2)/100
    PRho = -K*T*np.exp(-r*T) * sc.norm.cdf(-d2)/100
    Vega = S*np.sqrt(T)*N1D / 100

    col1, col2, col3, col4 = st.columns(4)
    col1.write(f'Call Delta: {np.round(Delta,4)}')
    col2.write(f'Call Theta: {np.round(CTheta,4)}')
    col1.write(f'Call Rho: {np.round(CRho,6)}')
    col2.write(f'Gamma: {np.round(Gamma,6)}')
    col1.write(f'Vega: {np.round(Vega,4)}')

    col3.write(f'Put Delta: {np.round(Delta-1,4)}')
    col4.write(f'Put Theta: {np.round(PTheta,4)}')
    col3.write(f'Put Rho: {np.round(PRho,6)}')
    col4.write(f'Gamma: {np.round(Gamma,6)}')
    col3.write(f'Vega: {np.round(Vega,4)}')
    
    

    st.markdown('### Heatmap')

    col1, col2, col3, col4 = st.columns(4)
    LV = col1.slider("Minimum Volatility:", min_value = 0.001, value = 0.05, step = 0.001, format="%0.3f")
    HV = col2.slider("Maximum Volatility:", min_value = LV, value = 0.15, step = 0.001, format="%0.3f")
    LS = col3.slider("Minimum Spot Price:", min_value = 0.001, value = 45.00, max_value= 300.00, step = 0.5, format="%0.2f")
    HS = col4.slider("Maximum Spot Price:", min_value = LS, value = 55.00, max_value = 300.00, step = 0.5, format="%0.2f")

    SH = np.linspace(start= LS,stop= HS,num = 5)
    VH = np.linspace(start= LV,stop= HV,num= 5)


    def Standard_Euro_Heatmap(K, T, r, Q, VH, SH):
        C = []
        P = []

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

    def Barrier_Heatmap(K, T, r, Q, VH, SH, H):
        C_di = []
        C_do = []
        C_ui = []
        C_uo = []

        P_di = []
        P_do = []
        P_ui = []
        P_uo = []

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



    C_di, C_do, C_ui, C_uo, P_ui, P_uo, P_do, P_di = Barrier_Heatmap(K, T, r, Q, VH, SH, H)

    if option_type == 'Standard European':
        col1, col2 = st.columns(2)
        fig1, ax1 = plt.subplots()
        sns.heatmap(C, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax1, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
        ax1.set_xlabel('Spot Price')
        ax1.set_ylabel('Volatility')
        ax1.set_title('Call Option Price')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
        col1.pyplot(fig1)

        # Heatmap for Put Option Price
        fig2, ax2 = plt.subplots()
        sns.heatmap(P, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax2, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
        ax2.set_xlabel('Spot Price')
        ax2.set_ylabel('Volatility')
        ax2.set_title('Put Option Price')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        col2.pyplot(fig2)

    elif option_type == 'Knock Out Barrier':
        if H < K:
            col1, col2 = st.columns(2)
            fig1, ax1 = plt.subplots()
            sns.heatmap(C_do, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax1, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
            ax1.set_xlabel('Spot Price')
            ax1.set_ylabel('Volatility')
            ax1.set_title('Call Option Price')
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
            col1.pyplot(fig1)

            # Heatmap for Put Option Price
            fig2, ax2 = plt.subplots()
            sns.heatmap(P_do, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax2, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
            ax2.set_xlabel('Spot Price')
            ax2.set_ylabel('Volatility')
            ax2.set_title('Put Option Price')
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
            col2.pyplot(fig2)

        elif H >= K:
            col1, col2 = st.columns(2)
            fig1, ax1 = plt.subplots()
            sns.heatmap(C_uo, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax1, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
            ax1.set_xlabel('Spot Price')
            ax1.set_ylabel('Volatility')
            ax1.set_title('Call Option Price')
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
            col1.pyplot(fig1)

            # Heatmap for Put Option Price
            fig2, ax2 = plt.subplots()
            sns.heatmap(P_uo, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax2, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
            ax2.set_xlabel('Spot Price')
            ax2.set_ylabel('Volatility')
            ax2.set_title('Put Option Price')
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
            col2.pyplot(fig2)

    elif option_type == 'Knock In Barrier':
        if H < K:
            col1, col2 = st.columns(2)
            fig1, ax1 = plt.subplots()
            sns.heatmap(C_di, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax1, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
            ax1.set_xlabel('Spot Price')
            ax1.set_ylabel('Volatility')
            ax1.set_title('Call Option Price')
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
            col1.pyplot(fig1)

            # Heatmap for Put Option Price
            fig2, ax2 = plt.subplots()
            sns.heatmap(P_di, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax2, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
            ax2.set_xlabel('Spot Price')
            ax2.set_ylabel('Volatility')
            ax2.set_title('Put Option Price')
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
            col2.pyplot(fig2)

        elif H >= K:
            col1, col2 = st.columns(2)
            fig1, ax1 = plt.subplots()
            sns.heatmap(C_ui, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax1, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
            ax1.set_xlabel('Spot Price')
            ax1.set_ylabel('Volatility')
            ax1.set_title('Call Option Price')
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
            col1.pyplot(fig1)

            # Heatmap for Put Option Price
            fig2, ax2 = plt.subplots()
            sns.heatmap(P_ui, annot=True, cmap='Spectral', cbar=False, linewidths=0.5, fmt='.2f', ax=ax2, xticklabels=np.round(SH, 2), yticklabels=np.round(VH, 3))
            ax2.set_xlabel('Spot Price')
            ax2.set_ylabel('Volatility')
            ax2.set_title('Put Option Price')
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
            col2.pyplot(fig2)

