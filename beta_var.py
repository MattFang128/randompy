import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(123)

# Parameters
true_x = 10
time_intervals = [1, 5, 10]  # Using more extreme intervals to better showcase the effect
samples_per_interval = 30  # Equal sample size for each interval

# Generate data
data = []
for t in time_intervals:
    # Generate samples for current time interval: xi = x + N(0,1) * t
    xi_values = true_x + np.random.normal(0, 1, samples_per_interval) * t
    for xi in xi_values:
        data.append({'time_interval': t, 'xi': xi})

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate individual regression coefficients
individual_betas = {}
for t in time_intervals:
    subset = df[df['time_interval'] == t]
    X = subset[['xi']].values
    y = np.ones(len(subset)) * true_x  # True value is constant
    
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    individual_betas[t] = model.coef_[0]

# Calculate average beta
beta_bar = np.mean(list(individual_betas.values()))

# Calculate pooled beta
X_pooled = df[['xi']].values
y_pooled = np.ones(len(df)) * true_x
model_pooled = LinearRegression(fit_intercept=False)
model_pooled.fit(X_pooled, y_pooled)
pooled_beta = model_pooled.coef_[0]

# Print results
print(f"True x: {true_x}")
print("\nIndividual regression coefficients (β_i):")
for t, beta in individual_betas.items():
    print(f"t = {t}: β_{t} = {beta:.6f}")
print(f"\nMinimum β_i: {min(individual_betas.values()):.6f}")
print(f"Average of individual βs (β_bar): {beta_bar:.6f}")
print(f"Pooled regression coefficient (β): {pooled_beta:.6f}")

# Demonstrate why this happens by calculating the weighted average
print("\nWeighted Average Calculation:")
# Calculate weights (proportional to 1/variance)
weights = [1/(t**2) for t in time_intervals]
total_weight = sum(weights)
normalized_weights = [w/total_weight for w in weights]

print("Weights based on 1/variance:")
for t, w in zip(time_intervals, normalized_weights):
    print(f"t = {t}: weight = {w:.4f}")

weighted_beta = sum(individual_betas[t] * w for t, w in zip(time_intervals, normalized_weights))
print(f"\nWeighted average of βs: {weighted_beta:.6f}")
print(f"Pooled regression β: {pooled_beta:.6f}")

# Visualize the data
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red']
for i, t in enumerate(time_intervals):
    subset = df[df['time_interval'] == t]
    plt.scatter(subset['xi'], [true_x] * len(subset), alpha=0.6, 
                color=colors[i], label=f't = {t}')

# Plot regression lines
x_range = np.linspace(df['xi'].min(), df['xi'].max(), 100)
for i, t in enumerate(time_intervals):
    plt.plot(x_range, individual_betas[t] * x_range, '--', 
             color=colors[i], alpha=0.7, label=f'β_{t} = {individual_betas[t]:.4f}')

# Plot pooled regression line
plt.plot(x_range, pooled_beta * x_range, 'k-', linewidth=2, 
         label=f'Pooled β = {pooled_beta:.4f}')

plt.axhline(y=true_x, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=true_x, color='gray', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('Observed values (xi)')
plt.ylabel('True value (x)')
plt.title('Regression Lines: Individual vs. Pooled')
plt.legend()
plt.tight_layout()
