import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
# Loading the dataset
car_details = pd.read_csv("D:/Edge/CAR DETAILS.csv",encoding = 'latin1')
car_details.head()
# showing the sahpe
print(car_details.shape)
# Getting some information about the dataframe
car_details.info()
car_details.isnull().sum()
car_details.duplicated().value_counts()
car_details = car_details.drop_duplicates(subset='name')
car_details.shape
car_details.duplicated().value_counts()
car_details.describe()
car_details_1 = car_details.copy()
car_details_1.head()
correlation = car_details.corr()
# checking the distribution of categorical data
print(car_details.fuel.value_counts())
print(car_details.seller_type.value_counts())
print(car_details.transmission.value_counts())
print(car_details.owner.value_counts())
# encoding "fuel" column
car_details.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}},inplace=True)
# encoding "transmission" column
car_details.replace({'transmission':{'Automatic':0,'Manual':1}},inplace=True)
# encoding "seller_type" column
car_details.replace({'seller_type':{'Individual':0,'Dealer':1,'Trustmark Dealer':2}},inplace=True)
# encoding "owner" column
car_details.replace({'owner':{'First Owner':0, 'Second Owner':1,'Third Owner':2,'Fourth & Above Owner':3, 'Test Drive Car':4 }})
# Convert categorical variables to one-hot encoded columns
car_details = pd.get_dummies(car_details, columns=['fuel', 'seller_type', 'transmission', 'owner'],drop_first=True)
# Plotting the bar chart for 'selling_price'
plt.figure(figsize=(7,5))
plt.bar(car_details['year'], car_details['selling_price'])
plt.xticks(rotation=0, ha='right')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.title('Bar Chart for Selling Price')
plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(7,5))
plt.title('Correlation between km driven and selling price')
sns.regplot(x='km_driven', y='selling_price', data = car_details)
X = car_details.drop(['name','selling_price'], axis=1)
y = car_details['selling_price']
print(X)
print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print('X_test shape:', X_test.shape)
print('X_train shape:', X_train.shape)
print('y_test shape:', y_test.shape)
print('y_train shape:', y_train.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# loading the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,y_train)
prad = lin_reg_model.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print('MAE: ', (metrics.mean_absolute_error(prad,y_test)))
print('MSE: ', (metrics.mean_squared_error(prad,y_test)))
print('R2 score: ', (metrics.r2_score(prad,y_test)))
# Prediction on Training data
training_data_prediction = lin_reg_model.predict(X_train)
# R squared Error
error_score = metrics.r2_score(y_train, training_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(y_train, training_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
test_data_prediction = lin_reg_model.predict(X_test)
# R squared Error
error_score = metrics.r2_score(y_test,test_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(y_test, test_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel("Predicted Price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
# loading the liner regression model
lass_reg_model = Lasso()
lass_reg_model.fit(X_train,y_train)
sns.regplot(x=prad, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel("Actual Prices")
plt.title('Actual Price')
plt.title('Actual vs Predicted Price')
# Correlation Matrix
correlation_matrix = car_details.corr()
print(correlation_matrix)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define a list to store the models and their names
models = [
    ('Linear Regression', LinearRegression()),
    ('Decision Tree Regression', DecisionTreeRegressor()),
    ('Random Forest Regression', RandomForestRegressor()),
    ('Gradient Boosting Regression', GradientBoostingRegressor())
]
# Train and evaluate each model
results = {}
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
# Convert results to a DataFrame for better visualization
results_car_details = pd.DataFrame(results)
# Print the results
print(results_car_details)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Train the best model (Random Forest Regression) on the full training set
best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = best_model.predict(X_test)
# Evaluate the best model
mae = mean_absolute_error(y_test, y_pred)
print("MAE of the best model:", mae)
# Save the best model to a file using joblib
import joblib
model_car_details = 'best_model.joblib'
joblib.dump(best_model, model_car_details)
# Later, when you want to use the model for predictions:
# Load the model from the file
loaded_model = joblib.load(model_car_details)
# You can now use the loaded_model to make predictions on new data
# For example, if you have a new sample 'new_sample' containing the features, you can predict its selling price:
new_sample = [[2015, 80000, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]]
predicted_price = loaded_model.predict(new_sample)
print("Predicted Selling Price:", predicted_price[0])
import pandas as pd
import joblib
from random import sample
# Randomly select 20 data points to create the new dataset
c_details = car_details.sample(n=20, random_state=42)
# Split the new dataset into features (X_new) and target variable (y_new)
X_new = c_details.drop(['name', 'selling_price'], axis=1)
y_new = c_details['selling_price']
# Load the saved model from the file
loaded_model = joblib.load('best_model.joblib')
# Use the loaded model to predict the selling prices for the new dataset
y_pred_new = loaded_model.predict(X_new)
# Compare the predicted selling prices with the actual selling prices
result_df = pd.DataFrame({'Actual Selling Price': y_new, 'Predicted Selling Price': y_pred_new})
print(result_df)
