#write a program to forecast production for 10 years using
#Exponential
# Hyperbolic
# Harmonic
# methods
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
# Sample production data (Year, Production)
data = {
    'Year': np.arange(2000, 2021),
    'Production': [1000, 950, 900, 850, 800, 760, 720, 680, 650, 620,
                   590, 560, 530, 500, 480, 460, 440, 420, 400, 380, 360]
}
df = pd.DataFrame(data)
years = df['Year'].values
production = df['Production'].values
# Define forecasting models
def exponential_model(x, a, b):
    return a * np.exp(-b * (x - years[0]))

def hyperbolic_model(x, a, b):
    return a / (1 + b * (x - years[0]))

def harmonic_model(x, a, b):
    return a / (1 + b * (x - years[0]))**2
# Fit models to data
exp_params, _ = curve_fit(exponential_model, years, production, p0=[1000, 0.1])
hyp_params, _ = curve_fit(hyperbolic_model, years, production, p0=[1000, 0.1])
har_params, _ = curve_fit(harmonic_model, years, production, p0=[1000, 0.1])
# Forecast for the next 10 years
future_years = np.arange(2021, 2031)
exp_forecast = exponential_model(future_years, *exp_params)
hyp_forecast = hyperbolic_model(future_years, *hyp_params)
har_forecast = harmonic_model(future_years, *har_params)
# Plot results
plt.figure(figsize=(12, 8))
plt.scatter(years, production, label='Actual Production', color='black')
plt.plot(future_years, exp_forecast, label='Exponential Forecast', color='blue')
plt.plot(future_years, hyp_forecast, label='Hyperbolic Forecast', color='green')
plt.plot(future_years, har_forecast, label='Harmonic Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('Production')
plt.title('Production Forecasting for Next 10 Years')
plt.legend()
plt.grid()
plt.show()
# Print forecasted values
forecast_df = pd.DataFrame({
    'Year': future_years,
    'Exponential_Forecast': exp_forecast,
    'Hyperbolic_Forecast': hyp_forecast,
    'Harmonic_Forecast': har_forecast
})
print(forecast_df)
