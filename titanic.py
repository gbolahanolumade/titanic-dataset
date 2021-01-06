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
male_first_class = df.loc[[df.Sex =='male' & df.Pclass ==1]]

df.Fare.plot(kind='box')
df.describe(include='all')

df.Sex.value_counts()

df.Sex.value_counts(normalize='true')

