import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocessing_inputs():
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
    def data_encoding(data): 
    
        # Converting type of columns to category
        data['emp_title']=data['emp_title'].astype('category')
    
    
        #Assigning numerical values and storing it in another columns
        data['emp_title']=data['emp_title'].cat.codes
    
        #Create an instance of One-hot-encoder
        df = onehot_encode(data, 'homeownership', prefix = "ho")
        df = onehot_encode(df, 'loan_purpose', 'lp')

        return df
    def data_normalization(data):
        y=data['account_never_delinq_percent'].copy()
        X=data.drop('account_never_delinq_percent',axis=1).copy()
        scaling=StandardScaler()
        X=pd.DataFrame(scaling.fit_transform(X),columns=X.columns)
        return X,y
    data=data_clean()
    data1=data_encoding(data)
    X,y=data_normalization(data1)
    return X,y
X,y=preprocessing_inputs()
print(y)
print(X)






