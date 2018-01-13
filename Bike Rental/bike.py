import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pylab
from scipy import stats
import calendar
# import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from sklearn import metrics
from sklearn.utils import class_weight
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
import math
from subprocess import check_output

sns.set(style='white', context='notebook', palette='deep')

def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    datetimecol=test['datetime']
    # print train.head()
    # print test.info()
    # train = train[
    #     np.abs(train["count"] - train["count"].mean())<= (3 * train["count"].std())]
    target = train['count']
    combined=train.append(test)
    combined.reset_index(inplace=True,drop=True)
    combined.drop('casual',axis=1,inplace=True)
    combined.drop('registered',axis=1,inplace=True)
    combined.drop('atemp',axis=1,inplace=True)
    combined["date"] = combined.datetime.apply(lambda x: x.split()[0])
    combined["hour"] = combined.datetime.apply(lambda x: x.split()[1].split(":")[0]).astype("int")
    combined["year"] = combined.datetime.apply(lambda x: x.split()[0].split("-")[0])
    combined["weekday"] = combined.date.apply(lambda dateString: datetime.strptime(dateString, "%Y-%m-%d").weekday())
    combined["month"] = combined.date.apply(lambda dateString: datetime.strptime(dateString, "%Y-%m-%d").month)
    categoryVariableList = ["hour", "weekday", "month", "season", "weather", "holiday", "workingday","year"]
    for var in categoryVariableList:
        combined[var] = combined[var].astype("category")
    # combined = combined.drop(["datetime"], axis=1)
    # g = sns.heatmap(train[combined.columns.values].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
    # plt.show()
    print combined.columns.values
    combined.drop('date', axis=1, inplace=True)
    # predict windspeed zero values

    combined.to_csv('data/combined.csv')
    dataWind0 = combined[combined["windspeed"] == 0]
    dataWindNot0 = combined[combined["windspeed"] != 0]
    rfModel_wind = RandomForestRegressor()
    windColumns = ["humidity",'hour','weekday','season','weather']
    rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])
    wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])
    dataWind0["windspeed"] = wind0Values
    combined = dataWindNot0.append(dataWind0)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    combined = pd.get_dummies(combined, columns=["hour"])
    combined = pd.get_dummies(combined, columns=["weekday"])
    combined = pd.get_dummies(combined, columns=["month"])
    combined = pd.get_dummies(combined, columns=["season"])
    combined = pd.get_dummies(combined, columns=["weather"])
    combined = pd.get_dummies(combined, columns=["holiday"])
    combined = pd.get_dummies(combined, columns=["workingday"])
    # combined = combined.drop(["year"], axis=1)
    # combined = combined.drop(['windspeed'], axis=1)
    combined.to_csv('data/combined.csv')
    # print combined.shape
    # # print (combined.head(2))
    train_new = combined[pd.notnull(combined['count'])].sort_values(by=["datetime"])
    test_new = combined[~pd.notnull(combined['count'])].sort_values(by=["datetime"])
    dropFeatures = ["count", "datetime"]
    train_new = train_new.drop(dropFeatures, axis=1)
    test_new = test_new.drop(dropFeatures, axis=1)

    # print combined.info()
    Y_train = target
    X_train = train_new
    kfold = StratifiedKFold(n_splits=10)
    RFC = RandomForestRegressor()
    #
    # # Search grid for optimal parameters
    # # rf_param_grid = {"max_features": [0.7],
    # #                 "random_state": [42],
    # #                  "n_estimators": [100],
    # #                  "criterion": ["entropy"]}
    # #
    # # gsRFC = GridSearchCV(RFC, param_grid={}, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    # # gsRFC.fit(X_train, Y_train)
    # # RFC_Best = gsRFC.best_estimator_
    # #
    # # # Best score
    # # print gsRFC.best_score_
    # # print "RFC is %s" % RFC_Best
    # # feature_imp_sorted_rf = pd.DataFrame({'feature': list(X_train),
    # #                                       'importance': gsRFC.best_estimator_.feature_importances_}).sort_values(
    # #     'importance', ascending=False)
    # # features_top_n_rf = feature_imp_sorted_rf.head(20)['feature']
    # # print features_top_n_rf
    # # rfModel = RandomForestRegressor(n_estimators=1500)
    # # yLabelsLog = np.log1p(Y_train)
    # # rfModel.fit(X_train,yLabelsLog)
    # # preds = rfModel.predict(X=X_train)
    # # print ("RMSLE Value For Random Forest: ", rmsle(np.exp(yLabelsLog), np.exp(preds), False))
    #
    etc = ExtraTreesRegressor(n_estimators=950);  ### Test 0.41
    yLabelsLog = np.log1p(Y_train)
    etc.fit(X_train, yLabelsLog)
    feature_imp_sorted_et = pd.DataFrame({'feature': list(X_train),
                                          'importance': etc.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(40)['feature']

    gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.001);  ### Test 0.41
    yLabelsLog = np.log1p(Y_train)
    gbm.fit(X_train, yLabelsLog)
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(X_train),
                                          'importance': gbm.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(40)['feature']
    features_top_n = pd.concat([features_top_n_gb, features_top_n_et],\
                               ignore_index=True).drop_duplicates()
    print features_top_n.shape
    importances = gbm.feature_importances_
    indices = np.argsort(importances)[::-1]
    # print "Top 20 Important Features\n"
    # for f in range(20): print("%d. %s (%f)" % (f + 1, train_new.columns[indices[f]], importances[indices[f]]))
    train_new=train_new[features_top_n_et]
    test_new=test_new[features_top_n_et]
    etc.fit(train_new, yLabelsLog)
    gbm.fit(train_new, yLabelsLog)
    preds_gb = gbm.predict(X=train_new)
    preds_etc = etc.predict(X=train_new)
    preds=(preds_gb+preds_etc)/2
    print ("RMSLE Value For Gradient Boost and Extra Trees: ", rmsle(np.exp(yLabelsLog), np.exp(preds), False))
    preds_gb = gbm.predict(X=test_new)
    preds_etc = etc.predict(X=test_new)
    predsTest = (preds_gb + preds_etc) / 2
    submission = pd.DataFrame({
        "datetime": datetimecol,
        "count": [max(0, x) for x in np.exp(predsTest)]
    })
    submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)