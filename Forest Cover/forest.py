import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from sklearn import metrics
from sklearn.utils import class_weight
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
import math

sns.set(style='white', context='notebook', palette='deep')

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == '__main__':

    # Load data
    ##### Load train and Test set

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train_len = len(train)

    # separating "ID" and response variable "Cover_Type"
    Id = test['Id']
    y = train['Cover_Type'].astype(int)
    # df = train[['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    #        'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    #        'Horizontal_Distance_To_Fire_Points' ]]
    # df = test[['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    #        'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    #        'Horizontal_Distance_To_Fire_Points']]
    # scaler = preprocessing.MinMaxScaler()
    # scaled_train = scaler.fit_transform(train)
    # train = pd.DataFrame(scaled_train,columns=train.columns.values)
    # scaled_test = scaler.fit_transform(test)
    # test = pd.DataFrame(scaled_test,columns=train.columns.values)
    train = train.drop(['Id', 'Cover_Type',], 1)
    test = test.drop(['Id'], 1)
    # From both train and test data
    train = train.drop(['Soil_Type7', 'Soil_Type15', 'Soil_Type8', 'Soil_Type25'], axis=1)
    test = test.drop(['Soil_Type7', 'Soil_Type15', 'Soil_Type8', 'Soil_Type25'], axis=1)

    # # Create two new columns named Slope hydrology and Slope hydrology percent and remove any infinite values that may result
    train['slope_hyd'] = np.sqrt(train.Vertical_Distance_To_Hydrology ** 2 + \
                                    train.Horizontal_Distance_To_Hydrology ** 2)
    train.slope_hyd = train.slope_hyd.map(lambda x: 0 if np.isnan(x) or np.isinf(x) else x)

    train['slope_hyd_pct'] = train.Vertical_Distance_To_Hydrology / train.Horizontal_Distance_To_Hydrology
    train.slope_hyd_pct = train.slope_hyd_pct.map(lambda x: 0 if np.isnan(x) or np.isinf(x) else x)
    # Apply changes to test.csv as well
    test['slope_hyd'] = np.sqrt(test.Vertical_Distance_To_Hydrology ** 2 + \
                                   test.Horizontal_Distance_To_Hydrology ** 2)
    test.slope_hyd = test.slope_hyd.map(lambda x: 0 if np.isnan(x)or np.isinf(x) else x)
    test['slope_hyd_pct'] = test.Vertical_Distance_To_Hydrology / test.Horizontal_Distance_To_Hydrology
    test.slope_hyd_pct = test.slope_hyd_pct.map(lambda x: 0 if np.isnan(x) or np.isinf(x) else x)

    # Elevation adjusted by Horizontal distance to Hyrdrology
    train['Elev_to_HD_Hyd'] = (train.Elevation - 0.2 * train.Horizontal_Distance_To_Hydrology)/train.Elevation.std()
    test['Elev_to_HD_Hyd'] = (test.Elevation - 0.2 * test.Horizontal_Distance_To_Hydrology)/test.Elevation.std()
    # Elevation adjusted by Horizontal distance to Roadways
    train['Elev_to_HD_Road'] = (train.Elevation - 0.05 * train.Horizontal_Distance_To_Roadways)/train.Elevation.std()
    test['Elev_to_HD_Road'] = (test.Elevation - 0.05 * test.Horizontal_Distance_To_Roadways)/test.Elevation.std()
    # Elevation adjusted by Vertical distance to Roadways
    train['Elev_to_VD_Hyd'] = (train.Elevation - train.Vertical_Distance_To_Hydrology)/train.Elevation.std()
    test['Elev_to_VD_Hyd'] = (test.Elevation - test.Vertical_Distance_To_Hydrology)/test.Elevation.std()
    # Mean distance to Amenities
    train['Mean_Amenities'] = (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3
    test['Mean_Amenities'] = ( test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3
    # Mean Distance to Fire and Water
    train['Mean_Fire_Hyd'] = (train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2
    test['Mean_Fire_Hyd'] = (test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2

    min_max_scaler = preprocessing.MinMaxScaler()

    def scaleColumns(df, cols_to_scale):
        for col in cols_to_scale:
            if(df[col].max()-df[col].min()) > 0:
                df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
                df[col] = df[col].round(3)
        return df
    scaleColumns(train,train.columns.values)
    scaleColumns(test,test.columns.values)
    train.to_csv("data/train_1.csv", index=False)
    test.to_csv("data/test_1.csv", index=False)

    for col in ['Mean_Amenities','Mean_Fire_Hyd','slope_hyd','Horizontal_Distance_To_Fire_Points',\
                'Horizontal_Distance_To_Fire_Points','Horizontal_Distance_To_Roadways','Vertical_Distance_To_Hydrology',\
                'Horizontal_Distance_To_Hydrology']:
        train[col] = np.sqrt(train[col])
        test[col] = np.sqrt(test[col])
    # print train.iloc[:,:].skew()

    #Modelling
    # cv
    Y_train = y
    X_train = train
    kfold = StratifiedKFold(n_splits=10)

    # random labels_dict
    class_weights = pd.DataFrame({'Class_Weights': [0,0.372499,0.370654,0.062706,0.003757,0.081941,0.048014,0.060428]}, index=None)
    sample_weights = class_weights.ix[Y_train]
    sample_weights.to_csv("data/sample_weights.csv", index=False)
    print sample_weights

    #ETC Parameters tunning
    RFC = RandomForestClassifier()
    ETC = ExtraTreesClassifier()
    SVCM= SVC(probability=True)

    ## Search grid for optimal parameters
    et_param_grid = {"max_depth":[30],
                     "max_features": [0.7],
                    "random_state":[42],
                     "n_estimators": [300],
                     "criterion": ["entropy"]}

    gsETC = GridSearchCV(ETC, param_grid=et_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsETC.fit(X_train, Y_train,sample_weight=sample_weights.Class_Weights.values)

    ETC_best = gsETC.best_estimator_

    # Best score
    print gsETC.best_score_
    print "ETC is %s" % ETC_best
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(X_train),
                                          'importance': gsETC.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(20)['feature']

    # importances = gsETC.best_estimator_.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # print "Top 20 Important Features\n"
    # for f in range(20): print("%d. %s (%f)" % (f + 1, train.columns[indices[f]], importances[indices[f]]))
    #
    # # Learning curve
    # title = "Learning Curves (Random Forests, n_estimators=%d, max_depth=%.6f)" % (ETC_best.n_estimators, \
    #                                                                                ETC_best.max_depth)
    # plot_learning_curve(ETC_best\
    #                     , title, X_train, Y_train, cv=kfold, n_jobs=4)
    # plt.show()

    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     "max_depth":[25],
                     "max_features": [0.7],
                     "random_state":[42],
                     "n_estimators": [300]
                     }

    gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsGBC.fit(X_train, Y_train,sample_weight=sample_weights.Class_Weights.values)
    GBC_best = gsGBC.best_estimator_

    # Best score
    print gsGBC.best_score_
    print "GBC %s" % GBC_best
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(X_train),
                                          'importance': gsGBC.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(250)['feature']

    # importances = gsGBC.best_estimator_.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # print "Top 20 Important Features\n"
    # for f in range(20): print("%d. %s (%f)" % (f + 1, train.columns[indices[f]], importances[indices[f]]))

    # # Learning curve
    # title = "Learning Curves (Random Forests, n_estimators=%d, max_depth=%.6f)" % (GBC_best.n_estimators, \
    #                                                                                GBC_best.max_depth)
    # plot_learning_curve(GBC_best \
    #                     , title, X_train, Y_train, cv=kfold, n_jobs=4)
    # plt.show()
    # classification accuracy
    # print '\n Classification Report:\n', metrics.classification_report(Y_train, RFC_best.predict(X_train))
    # print '\n Confusion matrix:\n', metrics.confusion_matrix(Y_train, RFC_best.predict(X_train))

    # Logistic classifier
    # logitC = LogisticRegression(intercept_scaling=1,
    #            dual=False, fit_intercept=True, penalty='l1', tol=0.0001)
    # logit_param_grid = {'max_iter': [2000]}
    #
    # gslogitC = GridSearchCV(logitC, param_grid=logit_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    #
    # gslogitC.fit(X_train, Y_train)
    #
    # logitC_best = gslogitC.best_estimator_
    #
    # #Best score
    # print gslogitC.best_score_
    # print "Logistic  %s" % logitC_best

    # KNN
    # knnC = KNeighborsClassifier()
    # knn_grid = {'algorithm': ["auto"],
    #             "n_neighbors": [9]}
    #
    # gsknnC = GridSearchCV(knnC, param_grid=knn_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    # gsknnC.fit(X_train, Y_train)
    # gsknnC_best = gsknnC.best_estimator_
    #
    # # Best score
    # print gsknnC.best_score_
    # print "KNN  %s" % gsknnC_best
    #
    votingC = VotingClassifier(estimators=[('etc',ETC_best),('gbc',GBC_best)], \
                               voting='soft', n_jobs=4)
    #
    # gVoting = GridSearchCV(votingC, param_grid={}, cv=kfold, scoring="accuracy", n_jobs=1,\
    #                        verbose=1)
    #
    votingC.fit(X_train, Y_train)
    # print gVoting.best_score_

    #
    # on test data
    # train=train[features_top_n_rf]
    # test=test[features_top_n_rf]
    # gsRFC = RandomForestClassifier()
    # gsRFC.fit(X_train, Y_train)
    ct = votingC.predict(test)
    output = pd.DataFrame(Id)
    output['Cover_Type'] = ct
    output.head()
    output.to_csv("data/output.csv", index=False)