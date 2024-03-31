import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score

# Reading the dataset
df=pd.read_csv(r'C:\Users\At\Downloads\archive (1)\IMDb Movies India.csv', encoding = 'unicode_escape')


#Exploratory data analysis
print(df.head())
print(df.info())


##Data Preprocessing (Data Cleaning & Transformation)

# Removing null values
df = df.dropna()
print(df)

# Handling the null values
df.dropna(subset=['Name', 'Year', 'Duration', 'Rating', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], inplace=True)

#Extracting only the text part from the Name column
df['Name'] = df['Name'].str.extract('([A-Za-z\s\'\-]+)')

# Replacing the brackets from year column as observed above
df['Year'] = df['Year'].str.replace(r'[()]', '', regex=True).astype(int)

# Convert 'Duration' to numeric and replacing the min, while keeping only numerical part
df['Duration'] = pd.to_numeric(df['Duration'].str.replace(r' min', '', regex=True), errors='coerce')

# Splitting the genre by , to keep only unique genres and replacing the null values with mode
df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre')
df['Genre'].fillna(df['Genre'].mode()[0], inplace=True)

# Convert 'Votes' to numeric and replace the , to keep only numerical part
df['Votes'] = pd.to_numeric(df['Votes'].str.replace(',', ''), errors='coerce')



# Encoding categorical variables
df = df.fillna(method='ffill')
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df['Director'] = le.fit_transform(df['Director'])
df['Actor 1'] = le.fit_transform(df['Actor 1'])
df['Actor 2'] = le.fit_transform(df['Actor 2'])
df['Actor 3'] = le.fit_transform(df['Actor 3'])

df.drop('Name',   axis = 1, inplace = True)



# Grouping the columns with their average rating and then creating a new feature

genre_mean_rating = df.groupby('Genre')['Rating'].transform('mean')
df['Genre_mean_rating'] = genre_mean_rating

director_mean_rating = df.groupby('Director')['Rating'].transform('mean')
df['Director_encoded'] = director_mean_rating

actor1_mean_rating = df.groupby('Actor 1')['Rating'].transform('mean')
df['Actor1_encoded'] = actor1_mean_rating

actor2_mean_rating = df.groupby('Actor 2')['Rating'].transform('mean')
df['Actor2_encoded'] = actor2_mean_rating

actor3_mean_rating = df.groupby('Actor 3')['Rating'].transform('mean')
df['Actor3_encoded'] = actor3_mean_rating


# Defining features and target variable
X = df[['Genre_mean_rating', 'Director_encoded', 'Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded']]
y = df['Rating']


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Training the model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)


# Making predictions
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_dt = model_dt.predict(X_test)


# Evaluating the model
print('The performance evaluation of Linear Regression is below: ', '\n')
mse_lr = mean_squared_error(y_test, y_pred_lr)
print('Mean Squared Error (Linear Regression):', mse_lr)
print('R2 score: ',r2_score(y_test, y_pred_lr))
print('\n', '_'*100, '\n')

print('The performance evaluation of Random Forest Regressor is below: ', '\n')
mse_rf = mean_squared_error(y_test, y_pred_rf)
print('Mean Squared Error:', mse_rf)
print('R2 score: ',r2_score(y_test, y_pred_rf))
rmse = np.sqrt(mse_rf)
print('Root Mean Squared Error:', rmse)
print('\n', '_'*100, '\n')

print('The performance evaluation of Decision Tree Regressor is below: ', '\n')
mse_dt = mean_squared_error(y_test, y_pred_dt)
print('Mean Squared Error (Decision Tree Regressor):', mse_dt)
print('R2 score: ',r2_score(y_test, y_pred_dt))
print('\n', '_'*100, '\n')

print(X.head(5))
print(y.head(5))

data = {'Genre_mean_rating': [5.811087], 'Director_encoded': [4.400000], 'Actor1_encoded': [5.250000], 'Actor2_encoded': [4.40], 'Actor3_encoded': [4.46]}
df = pd.DataFrame(data)


# Predict the movie rating
predicted_rating = model_rf.predict(df)
print("Predicted Rating:", predicted_rating[0])