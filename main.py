import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load housing data
konut = pd.read_csv('Data/konut.csv')
data = pd.DataFrame(konut)

# Separate independent (X) and dependent (Y) variables. Exclude 'Price' and 'Port_no' columns for X.
X = data.drop(['Fiyat', 'Port_no'], axis=1)
Y = data['Fiyat']  # 'Price' is the dependent variable
print(X)
print(Y)

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Scale the independent variables
X_normalized = scaler.fit_transform(X)

# Convert normalized independent variables to a DataFrame
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)
print(X_normalized_df)

# Create and fit the linear regression model
lr_model1 = LinearRegression()
lr_model1.fit(X_normalized_df, Y)

# Calculate predicted values on the same independent variables
y_pred1 = lr_model1.predict(X_normalized_df)

# Calculate Mean Absolute Error (MAE)
mae1 = mean_absolute_error(Y, y_pred1)
print(f"Mean Absolute Error (MAE): {mae1}")
print(f"Predicted Values: {y_pred1}")

# Split the dataset into training and testing sets with a test size determined by a specific condition
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.16, random_state=42)

# Create and train the linear regression model on the training data
lr_model2 = LinearRegression()
lr_model2.fit(X_train, y_train)

# Predict on the test data
y_pred2 = lr_model2.predict(X_test)

# Calculate Mean Absolute Error (MAE) for the test data
mae2 = mean_absolute_error(y_test, y_pred2)
print(f"Mean Absolute Error (MAE) on Test Data: {mae2}")
print(f"Predicted Values on Test Data: {y_pred2}")

# Create DataFrames for coefficients and errors of both models
df_coefficients = pd.DataFrame(lr_model1.coef_, index=X.columns, columns=['coefficients_model1'])
df_coefficients['coefficients_model2'] = lr_model2.coef_

df_errors = pd.DataFrame({
    'coefficients_model1': [mae1, pd.NA],
    'coefficients_model2': [pd.NA, mae2]
}, index=['error_model1', 'error_model2'])

# Combine coefficients and errors DataFrames
df_comparison = pd.concat([df_coefficients, df_errors])
print(df_comparison)
