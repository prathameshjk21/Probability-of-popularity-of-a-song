# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 22:05:01 2020

@author: ASUS
"""
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import tree, svm ,linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


data = pd.read_csv("data.csv")

col = ["artists","release_date","duration_ms","year","name","id","explicit","mode",]
Data = data.drop(columns = col)

Data.columns = ['acousticness','danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness','speechiness', 'tempo','valence','popularity']

Data =Data.drop_duplicates(keep= 'first')
Data.isnull().sum()



#############normal distribution of popularity variable########################

plt.hist(Data['popularity'], bins = 'auto')


# detetcing outliers 

fig, axes = plt.subplots(nrows=3,ncols=4)
fig.set_size_inches(25,15)

#-- Plot total counts on y bar
sn.boxplot(data=Data, y="popularity",ax=axes[0][0])

#-- Plot temp on y bar
sn.boxplot(data=Data, y="acousticness",ax=axes[0][1])

#-- Plot atemp on y bar
sn.boxplot(data=Data, y="danceability",ax=axes[0][2])

#-- Plot hum on y bar
sn.boxplot(data=Data, y="energy",ax=axes[0][3])

#-- Plot windspeed on y bar
sn.boxplot(data=Data, y="instrumentalness",ax=axes[1][0])

#-- Plot total counts on y-bar and 'yr' on x-bar
sn.boxplot(data=Data,y="popularity",x="acousticness",ax=axes[1][1])



colr = Data.corr()

        
x = Data.iloc[:,0:10]
y= Data.iloc[:,10]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)





################################ linear regression##########################


regressor = LinearRegression()


regressor.fit(x,y)


predict_LR = model.predict(x_test)

error_LR = np.sqrt(mean_squared_error(y_test, predict_LR))



rf = RandomForestRegressor(n_estimators=300,max_features='log2').fit(x_train,y_train)


rf_predict = rf.predict(x_test)

error_RF = np.sqrt(mean_squared_error(y_test, rf_predict))

pickle.dump(regressor,open('music_popularity.pkl','wb'))
























