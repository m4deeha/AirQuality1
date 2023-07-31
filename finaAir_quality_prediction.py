#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import datetime
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


from statistics import mean, median, mode, stdev


# 1.DATASET IMPORT

# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


#city_day = "https://drive.google.com/file/d/1TugcD3ieiT_dOqlikVvSDJLWKkG9VH7n/view?usp=sharing"


# In[ ]:


airqualitydataset=pd.read_csv("city_day.csv")
print(airqualitydataset)


# In[ ]:


airqualitydataset.head()


# 2.EXPLORATORY DATA ANALYSIS

# In[ ]:


airqualitydataset.drop('NOx', inplace=True, axis=1)


# In[ ]:


airqualitydataset.drop('NH3', inplace=True, axis=1)
#droping off the irrelevant columns


# In[ ]:


airqualitydataset.drop('Date',inplace=True,axis=1)


# In[ ]:


airqualitydataset.drop(['Toluene','Xylene','SO2'],axis=1,inplace=True)


# In[ ]:


airqualitydataset.head()


# In[ ]:


airqualitydataset.shape


# In[ ]:


new_dataset=airqualitydataset.dropna()
#dropping the rows with null values


# In[ ]:


new_dataset.reset_index(drop=True, inplace=True)
#reseting the index


# In[ ]:


new_dataset.head()


# In[ ]:


new_dataset.info()


# In[ ]:


new_dataset.describe()


# In[ ]:


median(new_dataset['PM2.5'])


# In[ ]:


fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(new_dataset['PM2.5'],new_dataset['AQI'])


# 2.1 OUTLIERS DETECTION

# In[ ]:


Q1 = new_dataset['PM2.5'].quantile(0.25)
Q3 = new_dataset['PM2.5'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
median_value = new_dataset["PM2.5"].median()
new_dataset.loc[(new_dataset["PM2.5"] > upper) | (new_dataset['PM2.5'] < lower), "PM2.5"] = median_value


# In[ ]:


sns.boxplot(new_dataset['PM2.5'])


# In[ ]:


Q1 = new_dataset['PM10'].quantile(0.25)
Q3 = new_dataset['PM10'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
median_value = new_dataset["PM10"].median()
new_dataset.loc[(new_dataset["PM10"] > upper) | (new_dataset['PM10'] < lower), "PM10"] = median_value


# In[ ]:


sns.boxplot(new_dataset['PM10'])


# In[ ]:


fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(new_dataset['PM2.5'],new_dataset['PM10'])


# In[ ]:


Q1 = new_dataset['NO2'].quantile(0.25)
Q3 = new_dataset['NO2'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
median_value = new_dataset["NO2"].median()
new_dataset.loc[(new_dataset["NO2"] > upper) | (new_dataset['NO2'] < lower), "NO2"] = median_value


# In[ ]:


sns.boxplot(new_dataset['NO2'])


# In[ ]:


Q1 = new_dataset['CO'].quantile(0.25)
Q3 = new_dataset['CO'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
median_value = new_dataset["CO"].median()
new_dataset.loc[(new_dataset["CO"] > upper) | (new_dataset['CO'] < lower), "CO"] = median_value


# In[ ]:


sns.boxplot(new_dataset['CO'])


# In[ ]:


Q1 = new_dataset['O3'].quantile(0.25)
Q3 = new_dataset['O3'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
median_value = new_dataset["O3"].median()
new_dataset.loc[(new_dataset["O3"] > upper) | (new_dataset['O3'] < lower), "O3"] = median_value


# In[ ]:


sns.boxplot(new_dataset['O3'])


# In[ ]:


Q1 = new_dataset['Benzene'].quantile(0.25)
Q3 = new_dataset['Benzene'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
median_value = new_dataset["Benzene"].median()
new_dataset.loc[(new_dataset["Benzene"] > upper) | (new_dataset['Benzene'] < lower), "Benzene"] = median_value


# In[ ]:


sns.boxplot(new_dataset['Benzene'])


# In[ ]:


Q1 = new_dataset['AQI'].quantile(0.25)
Q3 = new_dataset['AQI'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
median_value = new_dataset["AQI"].median()
new_dataset.loc[(new_dataset["AQI"] > upper) | (new_dataset['AQI'] < lower), "AQI"] = median_value


# In[ ]:


sns.boxplot(new_dataset['AQI'])


# In[ ]:


fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(new_dataset['PM2.5'],new_dataset['AQI'])


# In[ ]:


fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(new_dataset['PM2.5'],new_dataset['CO'])


# In[ ]:


fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(new_dataset['PM2.5'],new_dataset['Benzene'])


# In[ ]:


columns = ["PM2.5", "PM10", "NO", "CO", "O3", "Benzene", "NO2", "AQI"]

# Create subplots for each column
fig, axes = plt.subplots(len(columns), 1, figsize=(8, 6*len(columns)))
plt.subplots_adjust(hspace=0.4)

# Generate line plots for each column
for i, column in enumerate(columns):
    axes[i].set_title(column)
    axes[i].plot(new_dataset[column])

plt.show()


# 2.2 CORRELATION BETWEEN COLUMNS

# In[ ]:


columns = ["PM2.5", "PM10", "NO2", "CO", "O3", "Benzene", "AQI",]

correlation_matrix = new_dataset[columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


# 3.FEATURE ENGINEERING

# In[ ]:


skewness = new_dataset.skew()
skewness


# In[ ]:


new_dataset['NO'] = np.log(new_dataset['NO'])
#new_dataset['NO'] = np.exp(new_dataset['NO'])


# In[ ]:


new_dataset['AQI'] = np.log(new_dataset['AQI'])


# In[ ]:


new_dataset['Benzene'] = np.log(new_dataset['Benzene'])


# In[ ]:


new_dataset['Benzene'] = np.exp(new_dataset['Benzene'])


# 3.1 LABEL ENCODING

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
new_dataset['City'] = label_encoder.fit_transform(new_dataset['City'])


# In[ ]:


custom_labels = {
    "Good": 0,
    "Moderate": 1,
    "Satisfactory": 2,
    "Poor":3,
    "Very Poor":4,
    "Severe":5
}
new_dataset['AQI_Bucket'] = new_dataset['AQI_Bucket'].map(custom_labels)


# In[ ]:


new_dataset['Benzene'].fillna(new_dataset['Benzene'].median(), inplace=True)


# In[ ]:


# from google.colab import files

# new_dataset.to_csv('new_dataset.csv', index=False)
# files.download('new_dataset.csv')


# In[ ]:


new_dataset


# In[ ]:


columns = ["City","PM2.5", "PM10", "NO2", "CO", "O3", "Benzene", "AQI","AQI_Bucket"]

correlation_matrix = new_dataset[columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
columns = ["City", "PM2.5", "PM10", "NO2", "CO", "O3", "Benzene", "AQI", "AQI_Bucket"]

# Separate the features (X) and the target variable (y)
X = new_dataset[columns]
y = new_dataset["AQI_Bucket"]

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the model to the data
rf.fit(X, y)

# Get the feature importances
importances = rf.feature_importances_

# Create a DataFrame to store the feature importances
feature_importance = pd.DataFrame({"Feature": columns, "Importance": importances})

# Sort the features by importance in descending order
feature_importance.sort_values("Importance", ascending=False, inplace=True)
feature_importance.reset_index(drop=True, inplace=True)

# Print the ranked feature list
print(feature_importance)


# 4.MACHINE LEARNING MODEL TRAINING

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pickle


# Assuming 'new_dataset' is your dataset DataFrame
#X = new_dataset.drop(columns=["AQI_Bucket"])  # Features
X = new_dataset[["City", "PM2.5", "PM10", "NO", "CO", "O3", "Benzene", "NO2", "AQI"]]
y = new_dataset["AQI_Bucket"]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression

# Assuming X_train, X_test, y_train, and y_test are already defined from the previous split

# Create a Linear Regression model
linear_reg = LinearRegression()

# Fit the model to the training data
linear_reg.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = linear_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mae= mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

rmse= np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("Training Accuracy  :",round(linear_reg.score(X_train,y_train)*100,2),"%")
print("Test Accuracy  :",round(linear_reg.score(X_test,y_test)*100,2),"%")


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already defined from the previous split

# Create a Decision Tree Regressor
tree = DecisionTreeRegressor()

# Fit the model to the training data
tree.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = tree.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mae= mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

rmse= np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("Training Accuracy  :",round(tree.score(X_train,y_train)*100,2),"%")
print("Test Accuracy  :",round(tree.score(X_test,y_test)*100,2),"%")


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already defined from the previous split

# Create a KNN Regressor
knn = KNeighborsRegressor()

# Fit the model to the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mae= mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

rmse= np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("Training Accuracy  :",round(knn.score(X_train,y_train)*100,2),"%")
print("Test Accuracy  :",round(knn.score(X_test,y_test)*100,2),"%")


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already defined from the previous split

# Create an AdaBoost Regressor
adaboost = AdaBoostRegressor()

# Fit the model to the training data
adaboost.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = adaboost.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mae= mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

rmse= np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("Training Accuracy  :",round(adaboost.score(X_train,y_train)*100,2),"%")
print("Test Accuracy  :",round(adaboost.score(X_test,y_test)*100,2),"%")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already defined from the previous split

# Create a Random Forest Regressor
random_forest = RandomForestRegressor()

# Fit the model to the training data
random_forest.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = random_forest.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mae= mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

rmse= np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("Training Accuracy  :",round(random_forest.score(X_train,y_train)*100,2),"%")
print("Test Accuracy  :",round(random_forest.score(X_test,y_test)*100,2),"%")


# In[ ]:


from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already defined from the previous split

# Create an XGBoost Regressor
xgb = XGBRegressor()

# Fit the model to the training data
xgb.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mae= mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

rmse= np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("Training Accuracy  :",round(xgb.score(X_train,y_train)*100,2),"%")
print("Test Accuracy  :",round(xgb.score(X_test,y_test)*100,2),"%")


# In[ ]:


from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

# Assuming X_train, X_test, y_train, and y_test are already defined from the previous split

# Create a Bayesian Ridge Regression model
bayesian_reg = BayesianRidge()

# Fit the model to the training data
bayesian_reg.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = bayesian_reg.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

mae= mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

rmse= np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("Training Accuracy  :",round(bayesian_reg.score(X_train,y_train)*100,2),"%")
print("Test Accuracy  :",round(bayesian_reg.score(X_test,y_test)*100,2),"%")


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}
print(param_grid)


# In[ ]:


xgb = XGBRegressor()


# In[ ]:


random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid,
                                   n_iter=10, scoring='r2', cv=5,
                                   verbose=1, random_state=42)

# Fit the model to the training data
random_search.fit(X_train,y_train)


# In[ ]:


# Print the best parameters and score
print("Best score: ", random_search.best_score_)
print("Best parameters found: ", random_search.best_params_)

# Evaluate the model on the test set
y_pred = random_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE Error: ", rmse)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)


# In[ ]:


print("Training Accuracy  :",round(random_search.score(X_train,y_train)*100,2),"%")
print("Test Accuracy  :",round(random_search.score(X_test,y_test)*100,2),"%")


# DEPLOYMENT

# In[ ]:


import pickle


# In[ ]:


pickle.dump(xgb,open('/content/airquality','wb'))


# In[ ]:


xgb_loaded=pickle.load(open('/content/airquality','rb'))


# In[ ]:


xgb_loaded.fit(X_train, y_train)


# In[ ]:


xgb_loaded.predict(X_test)

