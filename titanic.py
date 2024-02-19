import seaborn as sns
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\At\Downloads\archive\tested.csv')

#Exploring Data
print(df.shape)
print(df.head(5))
print(df.info())
print(df.drop(['Cabin'], axis=1, inplace=True))
# print(df.info())
print(df.describe())
print(df[['PassengerId','Survived']].describe())

#Exploratory Data Analysis

# plotting a bar chart for Sex and it's count
ax = sns.countplot(x = 'Sex',data = df)

for bars in ax.containers:
    ax.bar_label(bars)
plt.show()

#Checking Actual Survival 
# print(df.groupby(['Sex'], as_index=False)['Survived'].sum().sort_values(by='Survived', ascending=False))#0 - Survived , 1 to any_num - Didnt Survived

#plotting a bar for count of Survived passengers and their Sex 
ax=sns.countplot(df,x='Survived',hue='Sex') #0 - Survived , 1 - Didn't Survived
plt.show()



