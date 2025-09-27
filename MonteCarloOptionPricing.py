import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


class MonteCarloPricing:
    def __init__(self, stockPrice, strike, time, riskFreeRate, volatility, nPaths):
        self.S0 = float(stockPrice)
        self.K = strike
        self.T = time / 252 
        self.R = riskFreeRate
        self.volatility = volatility
        self.sigma = volatility
        self.nPaths = nPaths


    def _create_geometric_brownian_motion(self):
        
        dt = 1.0/252
        
        # Vectorised implementation of asset path generation
        # including four prices per day, used to create OHLC
        assetPath = self.S0 * np.exp(
            np.cumsum((self.R - 0.5 * self.sigma**2) * dt +
                    self.sigma * np.sqrt(dt) * np.random.randn(252)))

        
        return assetPath
    
    def ItoLemma(self):
        Z = np.random.randn(self.nPaths)
        # Apply GBM closed form solution
        ST = self.S0 * np.exp(
            (self.R - 0.5 * self.sigma**2) * self.T +
            self.sigma * np.sqrt(self.T) * Z
        )
        return ST


    def assetPrices(self):
        stvals = []
        for i in range(1000):
            random_numbers = np.random.randint(1, 5000)
            randVal = self.ItoLemma()
            stvals.append(randVal)

        return stvals
    

    def callPayoffMax(self, ST):
        maxVal = np.maximum(ST - self.K, 0)
        return maxVal
    
    def putPayoffMax(self, ST):
        maxVal = np.maximum(self.K-ST, 0)
        return maxVal
    

    def discountVal(self, ST):
        discountCall = np.exp(-self.R * self.T) * np.mean(self.callPayoffMax(ST=ST))
        discountPut = np.exp(-self.R * self.T) * np.mean(self.putPayoffMax(ST=ST))
        return discountCall, discountPut

prices = yf.download("SPY", start="2015-01-01")["Close"]

S0 = prices.iloc[-1]

MC = MonteCarloPricing(stockPrice=S0, strike=2500, time=365, riskFreeRate=0.04, volatility=0.05, nPaths=100000)

ST = MC.ItoLemma()
print("ItoLemma:", ST)

call, put = MC.discountVal(ST)

print("Initial Price:", S0)

print("Mean Terminal Price:", np.mean(ST))
print("Standard deviation Terminal Price:", np.std(ST))

print("Call:", call)
print("Put:", put)


path = MC._create_geometric_brownian_motion()
print("Brownian Motion:", path)


plt.figure(figsize=(10,5))
for _ in range(10):
    path = MC._create_geometric_brownian_motion()
    plt.plot(path, linewidth=1)
plt.title("Simulated Geometric Brownian Motion Paths")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

# --- Histogram of terminal prices ---
plt.figure(figsize=(8,5))
plt.hist(ST, bins=50, density=True, alpha=0.6, color='blue')
plt.axvline(np.mean(ST), color='red', linestyle='--', label=f"Mean: {np.mean(ST):.2f}")
plt.title("Distribution of Terminal Prices (Monte Carlo)")
plt.xlabel("Price")
plt.ylabel("Density")
plt.legend()
plt.show()