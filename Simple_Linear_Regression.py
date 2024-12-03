import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
file_path = 'All_pokemon.csv'  # Replace with the correct path
data = pd.read_csv(file_path)

# Extract 'Height' and 'Weight' columns and drop rows with missing values
pokemon_data = data[['Height', 'Weight']].dropna()

# Split the data into 80% training and 20% testing
train_data, test_data = train_test_split(pokemon_data, test_size=0.2, random_state=192)

# Section 1: Scatterplot of data
plt.figure(figsize=(8, 6))
plt.scatter(pokemon_data['Height'], pokemon_data['Weight'], color='blue', alpha=0.6)
plt.title('Scatterplot of Height vs. Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Use 80% of the data for training
train_x = train_data['Height']
train_y = train_data['Weight']

# Add a constant term for the intercept
train_x_const = sm.add_constant(train_x)

# Perform regression analysis on the training data
model = sm.OLS(train_y, train_x_const).fit()

# Display regression summary
print("Regression Summary (Training Data):")
print(model.summary())

# Fitted values and residuals for the training data
train_data['Fitted'] = model.predict(train_x_const)
train_data['Residual'] = train_y - train_data['Fitted']

# Plot fitted line with scatterplot for training data
plt.figure(figsize=(8, 6))
plt.scatter(train_x, train_y, color='blue', alpha=0.6, label='Observed Data')
plt.plot(train_x, train_data['Fitted'], color='red', label='Fitted Line')
plt.title('Fitted Line Plot (Training Data)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Residual plot for the training data
plt.figure(figsize=(8, 6))
plt.scatter(train_x, train_data['Residual'], color='purple', alpha=0.6)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residual Plot (Training Data)')
plt.xlabel('Height')
plt.ylabel('Residuals')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Calculate the errors for the training data
train_data['PE'] = train_data['Weight'] - train_data['Fitted']
train_data['APE'] = train_data['PE'].abs()
train_data['RE'] = train_data['PE'] / train_data['Weight'].abs()
train_data['ARE'] = train_data['RE'].abs()

# Calculate the mean absolute prediction error (MAPE) and mean absolute relative error (MARE) for training data
MAPE_train = train_data['APE'].mean()
MARE_train = train_data['ARE'].mean()

# Display the MAPE and MARE for the training data
print(f'\nMAPE (Mean Absolute Prediction Error) for Training Data: {MAPE_train:.4f}')
print(f'MARE (Mean Absolute Relative Error) for Training Data: {MARE_train:.4f}')

# Show the table with the training data predictions and errors
print("\nTraining Data with Predictions and Errors:")
print(train_data[['Height', 'Weight', 'Fitted', 'PE', 'APE', 'RE', 'ARE']])

# Testing data predictions
test_x = test_data['Height']
test_y = test_data['Weight']

# Add constant term for the intercept in the test data
test_x_const = sm.add_constant(test_x)

# Predict the weight using the model for the test data
test_data['Fitted'] = model.predict(test_x_const)

# Calculate the errors for the test data
test_data['PE'] = test_data['Weight'] - test_data['Fitted']
test_data['APE'] = test_data['PE'].abs()
test_data['RE'] = test_data['PE'] / test_data['Weight'].abs()
test_data['ARE'] = test_data['RE'].abs()

# Calculate the mean absolute prediction error (MAPE) and mean absolute relative error (MARE) for test data
MAPE_test = test_data['APE'].mean()
MARE_test = test_data['ARE'].mean()

# Display the MAPE and MARE for the test data
print(f'\nMAPE (Mean Absolute Prediction Error) for Test Data: {MAPE_test:.4f}')
print(f'MARE (Mean Absolute Relative Error) for Test Data: {MARE_test:.4f}')

# Show the table with the test data predictions and errors
print("\nTest Data with Predictions and Errors:")
print(test_data[['Height', 'Weight', 'Fitted', 'PE', 'APE', 'RE', 'ARE']])

# Calculate confidence and prediction intervals
height_value = 2.0  # Change this value to test for different heights
x_new = pd.DataFrame({'const': 1, 'Height': [height_value]})
predictions = model.get_prediction(x_new)
summary = predictions.summary_frame(alpha=0.05)  # 95% confidence

fitted_value = summary['mean'][0]
ci_lower = summary['mean_ci_lower'][0]
ci_upper = summary['mean_ci_upper'][0]
pi_lower = summary['obs_ci_lower'][0]
pi_upper = summary['obs_ci_upper'][0]

# Print intervals
print(f"\nFor Height = {height_value}:")
print(f"Fitted Value: {fitted_value:.2f}")
print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"95% Prediction Interval: [{pi_lower:.2f}, {pi_upper:.2f}]")

# Optionally, save the results to CSV files for both training and test data
train_data.to_csv('pokemon_train_predictions_with_errors.csv', index=False)
test_data.to_csv('pokemon_test_predictions_with_errors.csv', index=False)
