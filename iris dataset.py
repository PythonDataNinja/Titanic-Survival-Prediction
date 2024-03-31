import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

df=pd.read_csv(r'C:\Users\At\Downloads\archive (2)\IRIS.csv', encoding = 'unicode_escape')

df.head()
df.describe()
df.info()

#Visualize the whole dataset
print(sns.pairplot(df, hue='species'))

