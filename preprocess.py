import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

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
        pca = PCA(n_components=26)
        X = pca.fit_transform(X)
        return X,y
    
    data=data_clean()
    data1=data_encoding(data)
    X,y=data_normalization(data1)
    return X,y

def train():
    X,y=preprocessing_inputs()
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.9,random_state=42)
    model=Ridge()
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    param_grid = {'alpha': np.logspace(-3, 3, 10)}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train_poly,y_train)
    print('Best alpha:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)
    best_ridge = grid_search.best_estimator_
    best_ridge.fit(X_train_poly, y_train)
    return X_test_poly,best_ridge,y_test

def predict(X_test_poly,model,y_test):
    y_pred=model.predict(X_test_poly)
    print( r2_score(y_test, y_pred))
    print( mean_absolute_error(y_test, y_pred))
    print( mean_squared_error(y_test, y_pred))
X_test_poly,model,y_test=train()
predict(X_test_poly,model,y_test)

    







