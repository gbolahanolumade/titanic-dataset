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

pd.crosstab(df.Sex, df.Pclass)

pd.crosstab(df.Sex, df.Pclass).plot(kind='bar');


# pivot table
df.pivot_table(index='Sex',columns = 'Pclass',values='Age', aggfunc='mean')

df.groupby(['Sex','Pclass']).Age.mean().unstack()


# extract rows with Embarked as Null
df[df.Embarked.isnull()]

# how many people embarked at different points
df.Embarked.value_counts()

# which embarked point has higher survival count
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Embarked)




# Option 2 : explore the fare of each class for each embarkment point
df.groupby(['Pclass', 'Embarked']).Fare.median()


# replace the missing values with 'C'
df.Embarked.fillna('C', inplace=True)

df[df.Embarked.isnull()]

median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'),'Fare'].median()
print (median_fare)

df.Fare.fillna(median_fare, inplace=True)



df.Age.plot(kind='hist', bins=20, color='c');


# median values
df.groupby('Sex').Age.median()

# visualize using boxplot
df[df.Age.notnull()].boxplot('Age','Sex');


# option 3 : replace with median age of Pclass
df[df.Age.notnull()].boxplot('Age','Pclass');



# Function to extract the title from the name 
def GetTitle(name):
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title

# use map function to apply the function on each Name value row i
df.Name.map(lambda x : GetTitle(x)) # alternatively you can use : df.Name.map(GetTitle)


df.Name.map(lambda x : GetTitle(x)).unique()
# Function to extract the title from the name 
def GetTitle(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

# create Title feature
df['Title'] =  df.Name.map(lambda x : GetTitle(x))

# Box plot of Age with title
df[df.Age.notnull()].boxplot('Age','Title');

# Box plot of Age with title
df[df.Age.notnull()].boxplot('Age','Title');