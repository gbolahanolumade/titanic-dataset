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

# replace missing values
title_age_median = df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median , inplace=True)


#Exploring Outliers
df.Age.plot(kind='hist', bins=20, color='c')

df.loc[df.Age > 70]

df.Fare.plot(kind='box', title='histogram of Fare')

df.Fare.plot(kind='hist', title='histogram for Fare', bins=20, color='c');

#Reducing Skewness
LogFare = np.log(df.Fare + 1)

# Histogram of LogFare
LogFare.plot(kind='hist', color='c', bins=20);

# binning
pd.qcut(df.Fare, 4)

pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high']) # discretization

pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high']).value_counts().plot(kind='bar', color='c', rot=0);

# create fare bin feature
df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high'])


#Feauture Engineering

#Agestate
df['AgeState'] = np.where(df['Age']>=18, 'Adult','Child')

df['AgeState'].value_counts()

pd.crosstab(df[df.Survived != -55].Survived, df[df.Survived != -55].AgeState)


# Family : 
df['FamilySize'] = df.Parch + df.SibSp + 1

# explore the family feature
df['FamilySize'].plot(kind='hist', color='c');

# further explore this family with max family members
df.loc[df.FamilySize == df.FamilySize.max(),['Name','Survived','FamilySize','Ticket']]


pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].FamilySize)


# a lady aged more thana 18 who has Parch >0 and is married (not Miss)
df['IsMother'] = np.where(((df.Sex == 'female') & (df.Parch > 0) & (df.Age > 18) & (df.Title != 'Miss')), 1, 0)


# Crosstab with IsMother
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].IsMother)


# explore Cabin values
df.Cabin

# use unique to get unique values for Cabin feature
df.Cabin.unique()


# look at the Cabin = T
df.loc[df.Cabin == 'T']



# set the value to NaN
df.loc[df.Cabin == 'T', 'Cabin'] = np.NaN

# look at the unique values of Cabin again
df.Cabin.unique()
# extract first character of Cabin string to the deck
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
df['Deck'] = df['Cabin'].map(lambda x : get_deck(x))
# check counts
df.Deck.value_counts()



# use crosstab to look into survived feature cabin wise
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Deck)


#CATEGORICAL FEATURE ENGINEERING

# sex
df['IsMale'] = np.where(df.Sex == 'male', 1, 0)
# columns Deck, Pclass, Title, AgeState
df = pd.get_dummies(df,columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])

df.info()

# drop columns
df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1, inplace=True)

# reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]

df.info()


