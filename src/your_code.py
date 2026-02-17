'''
Sample code using libraries(numpy, pandas, matplotlib, seaborn, and scikit-learn).
Purpose: To demonstrate the usage of these libraries for data analysis and visualization.
'''
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create sample data using NumPy
np.random.seed(42)
X = np.random.rand(100, 1) * 10      # 100 random numbers between 0 and 10
y = 2.5 * X.flatten() + np.random.randn(100) * 2  # Linear relationship with noise

# Convert to Pandas DataFrame
data = pd.DataFrame({'X': X.flatten(), 'y': y})
print(data.head())

# Visualize data using Matplotlib and Seaborn
plt.figure(figsize=(8,5))
plt.scatter(data['X'], data['y'], color='blue', label='Data points')  # Matplotlib scatter
plt.title('Scatter plot of X vs y')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Seaborn regression plot
sns.lmplot(x='X', y='y', data=data, height=5, aspect=1.5)
plt.title('Seaborn Linear Regression Plot')
plt.show()

# Prepare data for modeling
X_train, X_test, y_train, y_test = train_test_split(
    data[['X']], data['y'], test_size=0.2, random_state=42
)

# Train a simple linear regression model (scikit-learn)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model Coefficient: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")

# Plot predictions vs actual
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

