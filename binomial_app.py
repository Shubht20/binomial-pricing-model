import streamlit as st
import numpy as np
import pandas as pd

st.title("📈 Binomial Option Pricing Calculator")

# Inputs
S = st.number_input("Spot Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T, in years)", value=1.0)
r = st.number_input("Risk-Free Interest Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
N = st.slider("Number of Steps (N)", min_value=1, max_value=100, value=3)
option_type = st.selectbox("Option Type", ("call", "put"))
style = st.selectbox("Option Style", ("european", "american"))

# Binomial Pricing Logic
def binomial_option_pricing(S, K, T, r, sigma, N, option_type='call', style='european'):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Asset Price Tree
    asset_prices = np.zeros((N + 1, N + 1))
    asset_prices[0, 0] = S
    for i in range(1, N + 1):
        asset_prices[i, 0] = asset_prices[i - 1, 0] * u
        for j in range(1, i + 1):
            asset_prices[i, j] = asset_prices[i - 1, j - 1] * d

    # Option Value Tree
    option_values = np.zeros_like(asset_prices)
    for j in range(N + 1):
        if option_type == 'call':
            option_values[N, j] = max(0, asset_prices[N, j] - K)
        else:
            option_values[N, j] = max(0, K - asset_prices[N, j])

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = discount * (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1])
            if style == 'american':
                if option_type == 'call':
                    exercise = max(0, asset_prices[i, j] - K)
                else:
                    exercise = max(0, K - asset_prices[i, j])
                option_values[i, j] = max(hold, exercise)
            else:
                option_values[i, j] = hold

    return option_values[0, 0], asset_prices, option_values

# Calculate and Display
price, asset_tree, option_tree = binomial_option_pricing(S, K, T, r, sigma, N, option_type, style)
st.subheader(f"📌 Option Price: `{price:.4f}`")

# Display Trees as Tables
asset_df = pd.DataFrame('', index=range(N+1), columns=range(N+1))
opt_df = pd.DataFrame('', index=range(N+1), columns=range(N+1))

for i in range(N+1):
    for j in range(i+1):
        asset_df.iloc[i, j] = f"{asset_tree[i, j]:.2f}"
        opt_df.iloc[i, j] = f"{option_tree[i, j]:.2f}"

st.write("📊 Asset Price Tree")
st.dataframe(asset_df)

st.write(f"📊 Option Value Tree ({style.title()} {option_type.title()})")
st.dataframe(opt_df)
