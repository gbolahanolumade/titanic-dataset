# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 23:10:45 2021

@author: GbolahanOlumade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df=pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.info()
test_df.info()

test_df['Survived']= -55

df = pd.concat((train_df,test_df))

male_passangers = df.loc[df.Sex == 'male']

male_first_class = df.loc[((df.Sex =='male') & (df.Pclass ==1))]


df.Fare.plot(kind='box')
df.describe(include='all')

df.Sex.value_counts()

df.Sex.value_counts(normalize='true')


df.Pclass.value_counts().plot(kind='bar')

# title : to set title, color : to set color,  rot : to rotate labels 
df.Pclass.value_counts().plot(kind='bar',rot = 0, title='Class wise passenger count', color='c');

df.Age.plot(kind='hist', title='histogram for Age', color='c', bins=20);

print('skewness for age : {0:.2f}'.format(df.Age.skew()))
print('skewness for fare : {0:.2f}'.format(df.Fare.skew()))

df.groupby(['Sex']).Fare.mean()
df.groupby(['Pclass']).Age.mean()

df.groupby(['Pclass'])['Age','Fare'].mean()

df.groupby(['Pclass']).agg({'Fare' : 'mean', 'Age' : 'median'})

# more complicated aggregations 
aggregations = {
    'Fare': { # work on the "Fare" column
        'mean_Fare': 'mean',  # get the mean fare
        'median_Fare': 'median', # get median fare
        'max_Fare': max,
        'min_Fare': np.min
    },
    'Age': {     # work on the "Age" column
        'median_Age': 'median',   # Find the max, call the result "max_date"
        'min_Age': min,
        'max_Age': max,
        'range_Age': lambda x: max(x) - min(x)  # Calculate the age range per group
    }
}


df.groupby(['Pclass']).agg(aggregations)

