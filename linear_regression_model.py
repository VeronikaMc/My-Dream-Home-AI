import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data
data = {
    'income': [40000, 50000, 60000, 70000, 80000],
    'max_home_price': [120000, 150000, 180000, 210000, 240000]
}

df = pd.DataFrame(data)
X = df[['income']]
y = df['max_home_price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Print model parameters
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Predict for a new income
user_income = 75000
predicted_price = model.predict([[user_income]])
print(f"Estimated maximum home price: ${predicted_price[0]:,.2f}")

# Optional visualization
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Income')
plt.ylabel('Max Home Price')
plt.title('Income vs Home Affordability')
plt.show()
