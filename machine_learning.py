import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def label_encoding(df,col_names):
    label_encoder = preprocessing.LabelEncoder()
    for col in col_names:
        df[col] = label_encoder.fit_transform(df[col])
    return df


def random_forest_regressor_g3(df):
    # col_names = ["Mjob","school","sex","Pstatus","address","famsize"]
    # df = label_encoding(df,col_names)
    x = df.loc[:,["Mjob","Fedu","school","sex","Medu","age","Pstatus","address","famsize","G1","G2"]].values
    y1 = df.loc[:,["G3"]].values
    y1 = np.ravel(y1)
    X_train, X_test, y_train, y_test = train_test_split(x, y1, test_size=0.1, random_state=0)
    regressor = RandomForestRegressor(n_estimators=200, random_state=0, criterion="mse")
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    test_set_r2 = r2_score(y_test, y_pred)
    return regressor, test_set_r2


def multivariate_linear_regression(df):
    col_names = ["Mjob","school","sex","Pstatus","address","famsize"]
    df = label_encoding(df,col_names)
    x = df.loc[:,["Mjob","Fedu","school","sex","Medu","age","Pstatus","address","famsize","G1","G2"]].values
    y1 = df.loc[:,["G3"]].values
    y1 = np.ravel(y1)
    X_train, X_test, y_train, y_test = train_test_split(x, y1, test_size=0.1, random_state=0)
    lin_reg_mod = LinearRegression()
    lin_reg_mod.fit(X_train, y_train)
    y_pred = lin_reg_mod.predict(X_test)
    test_set_r2 = r2_score(y_test, y_pred)
    return lin_reg_mod, test_set_r2


def random_forest_regressor_g2(df):
    x = df.loc[:,["Fedu","Medu","studytime", "G1"]].values
    y1 = df.loc[:,["G2"]].values
    y1 = np.ravel(y1)
    X_train, X_test, y_train, y_test = train_test_split(x, y1, test_size=0.1, random_state=0)
    regressor = RandomForestRegressor(n_estimators=200, random_state=0, criterion="mse")
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    test_set_r2 = r2_score(y_test, y_pred)
    return regressor, test_set_r2