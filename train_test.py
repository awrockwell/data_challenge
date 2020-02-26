import pandas as pd
import numpy as np
import gzip
import os
import glob
import csv
from sklearn.model_selection import train_test_split
from linear_p_values import LinearRegression

os.chdir(os.getcwd() + '/data/')

df = pd.read_csv('test.csv')

y = df['y_percent_return_over_investment']
feature_columns = [colname for colname in list(df.columns) if colname not in {'y_percent_return_over_investment'}]
X = df[feature_columns]

reg = LinearRegression()
reg.fit(X, y)

print(reg.p)

# consider PCA


# Todo PCA for linear regression (predict paid back and total interest), random forests or (classify who will be a winner), bootstrapping and average for logistic regression
# needs to be simple, basically a heuristic, because it's a manual process of going to a website and select
# might be able to go/say the API route.
