import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model

# Load the CSV file
df = pd.read_csv("homeprices.csv")
print(df)

# Create and fit the linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])

# Predict the price for an area of 3300 sq. ft. using a DataFrame
predicted_price = reg.predict(pd.DataFrame([[3300]], columns=['area']))
print(f"Predicted price for an area of 3300 sq. ft. is: {predicted_price[0]}")

predicted_price = reg.predict(pd.DataFrame([[5500]], columns=['area']))
print(f"Predicted price for an area of 3300 sq. ft. is: {predicted_price[0]}") 
 
 # to check it 
 
m = reg.coef_
x = 3300
b = reg.intercept_ 
y = m * x + b
print(y)

m = reg.coef_
x = 5500
b = reg.intercept_ 
y = m * x + b
print(y)
