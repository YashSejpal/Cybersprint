import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder


def data_clean():
    df = pd.read_csv("loans_clean_schema.csv")
    ##print(df.head())
    df.dropna(axis = 0, inplace=True)
    ##print(df.size)
    ##df.replace("NA", np.NaN)
    return df

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix = prefix)
    df = pd.concat([df, dummies], axis = 1)
    df = df.drop(column, axis = 1)
    return df

def data_encoding(): 
    #Retrieving data
    data = pd.read_csv('loans_clean_schema.csv')
 
    # Converting type of columns to category
    data['emp_title']=data['emp_title'].astype('category')
 
 
    #Assigning numerical values and storing it in another columns
    data['emp_title']=data['emp_title'].cat.codes
 
    #Create an instance of One-hot-encoder
    df = onehot_encode(data, 'homeownership', prefix = "ho")
    df = onehot_encode(data, 'loan_purpose', 'lp')

    return df


x = data_encoding()
print(x)