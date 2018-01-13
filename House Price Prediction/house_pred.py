import os
import warnings
warnings.filterwarnings("ignore")
import xgboost
import pandas as pd
import numpy as np
import pylab
from scipy import stats
from scipy.stats import norm, skew
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
import math
from subprocess import check_output

sns.set(style='white', context='notebook', palette='deep')


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print train.info()
    # Save the 'Id' column
    train_ID = train['Id']
    test_ID = test['Id']
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    # fig, ax = plt.subplots()
    # ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
    # plt.ylabel('SalePrice', fontsize=13)
    # plt.xlabel('GrLivArea', fontsize=13)
    # plt.show()
    # Deleting outliers
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
    # Check the graphic again
    # fig, ax = plt.subplots()
    # ax.scatter(train['GrLivArea'], train['SalePrice'])
    # plt.ylabel('SalePrice', fontsize=13)
    # plt.xlabel('GrLivArea', fontsize=13)
    # plt.show()
    #
    # sns.distplot(train['SalePrice'], fit=norm);
    #
    # # Get the fitted parameters used by the function
    # (mu, sigma) = norm.fit(train['SalePrice'])
    # print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    #
    # # Now plot the distribution
    # plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
    #            loc='best')
    # plt.ylabel('Frequency')
    # plt.title('SalePrice distribution')
    #
    # # Get also the QQ-plot
    # fig = plt.figure()
    # res = stats.probplot(train['SalePrice'], plot=plt)
    # plt.show()
    # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # # Check the new distribution
    # sns.distplot(train['SalePrice'], fit=norm);
    #
    # # Get the fitted parameters used by the function
    # (mu, sigma) = norm.fit(train['SalePrice'])
    # print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    #
    # # Now plot the distribution
    # plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
    #            loc='best')
    # plt.ylabel('Frequency')
    # plt.title('SalePrice distribution')
    #
    # # Get also the QQ-plot
    # fig = plt.figure()
    # res = stats.probplot(train['SalePrice'], plot=plt)
    # plt.show()

    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    print("all_data size is : {}".format(all_data.shape))

    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
    print missing_data.head(20)

    # f, ax = plt.subplots(figsize=(15, 12))
    # plt.xticks(rotation='90')
    # sns.barplot(x=all_data_na.index, y=all_data_na)
    # plt.xlabel('Features', fontsize=15)
    # plt.ylabel('Percent of missing values', fontsize=15)
    # plt.title('Percent missing data by feature', fontsize=15)
    # plt.show()

    # Correlation map to see how features are correlated with SalePrice
    # corrmat = train.corr()
    # plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat, vmax=0.9, square=True)
    # plt.show()

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    # print all_data['PoolQC'].unique()
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')
    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data = all_data.drop(['Utilities'], axis=1)
    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
    from sklearn.preprocessing import LabelEncoder

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(all_data[c].values))
        all_data[c] = lbl.transform(list(all_data[c].values))
    # Adding total sqfootage feature
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    print skewness.head(10)

    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        # all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)
    all_data = pd.get_dummies(all_data)
    print(all_data.shape)
    train = all_data[:ntrain]
    test = all_data[ntrain:]

    #  Modelling

    # Validation function
    n_folds = 5


    def rmsle_cv(model):
        kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
        rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
        return (rmse)


    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    # model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
    #                              learning_rate=0.05, max_depth=3,
    #                              min_child_weight=1.7817, n_estimators=2200,
    #                              reg_alpha=0.4640, reg_lambda=0.8571,
    #                              subsample=0.5213, silent=1,
    #                              random_state=7, nthread=-1)
    # model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
    #                               learning_rate=0.05, n_estimators=720,
    #                               max_bin=55, bagging_fraction=0.8,
    #                               bagging_freq=5, feature_fraction=0.2319,
    #                               feature_fraction_seed=9, bagging_seed=9,
    #                               min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    score = rmsle_cv(lasso)
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(ENet)
    print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(KRR)
    print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    score = rmsle_cv(GBoost)
    print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # score = rmsle_cv(model_xgb)
    # print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    # score = rmsle_cv(model_lgb)
    # print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
    score = rmsle_cv(averaged_models)
    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
                                                     meta_model=lasso)

    score = rmsle_cv(stacked_averaged_models)
    print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    stacked_averaged_models.fit(train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    print(rmsle(y_train, stacked_train_pred))

    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = stacked_pred
    sub.to_csv('submission.csv',index=False)