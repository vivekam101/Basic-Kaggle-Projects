import os

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

def name_len_category(name_len):
    if (name_len <= 19):
        return 'Very_Short_Name'
    elif (name_len <= 28):
        return 'Short_Name'
    elif (name_len <= 45):
        return 'Medium_Name'
    else:
        return 'Long_Name'

def age_group_cat(age):
    if (age <= 1):
        return 'Baby'
    if (age <= 4):
        return 'Toddler'
    elif(age <= 12):
        return 'Child'
    elif (age <= 19):
        return 'Teenager'
    elif (age <= 30):
        return 'Adult'
    elif (age <= 50):
        return 'Middle_Aged'
    elif(age < 60):
        return 'Senior_Citizen'
    else:
        return 'Old'

def pclass_fare_category(df, Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare):
    if (df['Pclass'] == 1):
        if (df['Fare'] <= Pclass_1_mean_fare):
            return 'Pclass_1_Low_Fare'
        else:
            return 'Pclass_1_High_Fare'
    elif (df['Pclass'] == 2):
        if (df['Fare'] <= Pclass_2_mean_fare):
            return 'Pclass_2_Low_Fare'
        else:
            return 'Pclass_2_High_Fare'
    elif (df['Pclass'] == 3):
        if (df['Fare'] <= Pclass_3_mean_fare):
            return 'Pclass_3_Low_Fare'
        else:
            return 'Pclass_3_High_Fare'

    def family_size_category(family_size):
        if (family_size <= 1):
            return 'Single'
        elif (family_size <= 3):
            return 'Small_Family'
        else:
            return 'Large_Family'

def fare_category(fare):
    if (fare <= 4):
        return 'Very_Low_Fare'
    elif (fare <= 10):
        return 'Low_Fare'
    elif (fare <= 30):
        return 'Med_Fare'
    elif (fare <= 45):
        return 'High_Fare'
    else:
        return 'Very_High_Fare'

def loadData(dataset):

    # Fill empty and NaNs values with NaN
    dataset = dataset.fillna(np.nan)

    # Check for Null values
    print dataset.isnull().sum()

    # Info
    print train.info()
    print train.isnull().sum()
    print train.dtypes

    # Summarise and statistics
    print train.describe()

    # Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived
    # g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
    # plt.show()

    #visualising sibsp with survival
    # Explore SibSp feature vs Survived
    # g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 ,
    # palette = "muted")
    # g.despine(left=True)
    # g = g.set_ylabels("survival probability")

    # # Explore Parch feature vs Survived
    # g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 ,
    # palette = "muted")
    # g.despine(left=True)
    # g = g.set_ylabels("survival probability")
    #
    # # Explore Age vs Survived
    # g = sns.FacetGrid(train, col='Survived')
    # g = g.map(sns.distplot, "Age")
    #
    # # Explore Age distibution
    # g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
    # g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
    # g.set_xlabel("Age")
    # g.set_ylabel("Frequency")
    # g = g.legend(["Not Survived","Survived"])

    dataset["Fare"].isnull().sum()

    #Fill Fare missing values with the median value
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

    #Divide Fare for those sharing the same Ticket
    dataset['Group_Ticket'] = dataset['Fare'].groupby(
        by=dataset['Ticket']).transform('count')
    dataset['Fare'] = dataset['Fare'] / dataset['Group_Ticket']
    dataset.drop(['Group_Ticket'], axis=1, inplace=True)
    dataset['Fare_Category'] = dataset['Fare'].map(fare_category)
    # Explore Fare distribution
    # g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
    # g = g.legend(loc="best")
    # #plt.show()

    #after log
    # g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
    # g = g.legend(loc="best")
    # #plt.show()

    # g = sns.barplot(x="Sex",y="Survived",data=train)
    # g = g.set_ylabel("Survival Probability")

    # Explore Pclass vs Survived
    # g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 ,
    # palette = "muted")
    # g.despine(left=True)
    # g = g.set_ylabels("survival probability")

    # Explore Pclass vs Survived by Sex
    # g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
    #                    size=6, kind="bar", palette="muted")
    # g.despine(left=True)
    # g = g.set_ylabels("survival probability")

    dataset["Embarked"].isnull().sum()
    #Fill Embarked nan values of dataset set with 'S' most frequent value
    dataset["Embarked"] = dataset["Embarked"].fillna("S")

    # Explore Embarked vs Survived
    # g = sns.factorplot(x="Embarked", y="Survived",  data=train,
    #                    size=6, kind="bar", palette="muted")
    # g.despine(left=True)
    # g = g.set_ylabels("survival probability")

    # convert Sex into categorical value 0 for male and 1 for female
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})

    # Filling missing value of Age

    ## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
    # Index of NaN age rows
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

    for i in index_NaN_age :
        age_med = dataset["Age"].median()
        age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred) :
            dataset['Age'].iloc[i] = age_pred
        else :
            dataset['Age'].iloc[i] = age_med
    dataset['Age_Category'] = dataset['Age'].map(age_group_cat)

    # Get Title from Name
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Title"] = pd.Series(dataset_title)
    dataset["Title"].head()

    # Convert to categorical values Title
    dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    dataset["Title"] = dataset["Title"].astype(int)

    # # Name length
    dataset['Name_Length'] = dataset['Name'].str.len()
    #
    # # Name Length categories
    dataset['Name_Length_Category'] = dataset['Name_Length'].map(name_len_category)
    print dataset['Name_Length_Category'].head()
    #
    # # First Name, Second Name, Third Name lengths
    dataset['First_Name'] = dataset['Name'].str.extract('^(.+?),').str.strip()
    dataset['Last_Name'] = dataset['Name'].str.split("\.").str[1].str.strip()
    dataset['Last_Name'] = dataset['Last_Name'].str.strip("\([^)]*\)")
    dataset['Last_Name'].fillna(dataset['Name'].str.split("\.").str[1].str.strip())
    dataset['Original_Name'] = dataset['Name'].str.split("\((.*?)\)").str[1].str.strip(
        "\"").str.strip()

    # Create a family size descriptor from SibSp and Parch
    dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

    dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

    # convert to indicator values Name Length Category, Title and Embarked
    dataset = pd.get_dummies(dataset, columns = ["Title"])
    dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
    dataset = pd.get_dummies(dataset, columns=["Fare_Category"])
    dataset = pd.get_dummies(dataset, columns=["Age_Category"])
    dataset = pd.get_dummies(dataset, columns=["Original_Name"])
    dataset = pd.get_dummies(dataset, columns=["Last_Name"])
    dataset = pd.get_dummies(dataset, columns=["First_Name"])
    dataset = pd.get_dummies(dataset, columns=["Name_Length_Category"])

    dataset["Cabin"].head()
    dataset["Cabin"].describe()
    dataset["Cabin"].isnull().sum()
    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
    dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")

    ## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.

    dataset['Ticket_Letter'] = dataset['Ticket'].str.split().str[0]
    dataset['Ticket_Letter'] = dataset['Ticket_Letter'].apply(
        lambda x: np.NaN if x.isdigit() else x)
    dataset['Ticket_Number'] = dataset['Ticket'].apply(
        lambda x: pd.to_numeric(x, errors='coerce'))
    dataset['Ticket_Number'].fillna(0, inplace=True)
    dataset = pd.get_dummies(dataset, columns=['Ticket_Letter'])

    # Create mean fare Pclass
    Pclass_1_mean_fare = dataset['Fare'].groupby(by=dataset['Pclass']).mean().get([1]).values[0]
    Pclass_2_mean_fare = dataset['Fare'].groupby(by=dataset['Pclass']).mean().get([2]).values[0]
    Pclass_3_mean_fare = dataset['Fare'].groupby(by=dataset['Pclass']).mean().get([3]).values[0]
    dataset['Pclass_Fare_Category'] = dataset.apply(pclass_fare_category, args=(
    Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare), axis=1)
    dataset['Pclass'].replace([1, 2, 3], [Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare],
                                          inplace=True)
    dataset = pd.get_dummies(dataset, columns=["Pclass_Fare_Category"])
    # Drop useless variables
    dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
    dataset.drop(labels = ["Name"], axis = 1, inplace = True)
    dataset.drop(labels=["Ticket"], axis=1, inplace=True)

    #normalizing age and fare
    scale_age_fare = preprocessing.StandardScaler().fit(dataset[['Age', 'Fare','Pclass','Ticket_Number']])
    dataset[['Age', 'Fare','Pclass','Ticket_Number']] = scale_age_fare.transform(dataset[['Age', 'Fare', 'Pclass','Ticket_Number']])
    print dataset.head()
    return dataset

if __name__ == '__main__':

    # Load data
    ##### Load train and Test set

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    IDtest = test["PassengerId"]
    print train.describe()

    # detect outliers from Age, SibSp , Parch and Fare
    Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
    print Outliers_to_drop

    print train.loc[Outliers_to_drop]  # Show the outliers rows

    train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

    ## Join train and test datasets in order to obtain the same number of features during categorical conversion
    train_len = len(train)
    dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    dataset=loadData(dataset)
    # Separate train dataset and test dataset

    train = dataset[:train_len]
    test = dataset[train_len:]
    test.drop(labels=["Survived"],axis = 1,inplace=True)

    ## Separate train features and label

    train["Survived"] = train["Survived"].astype(int)
    Y_train = train["Survived"]
    X_train = train.drop(labels = ["Survived"],axis = 1)
    print X_train.columns.values
    dataset.to_csv("data/trainingSet.csv", index=False)

    # Modeling


    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits=10)

    # ### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING
    # Adaboost
    DTC = DecisionTreeClassifier()

    adaDTC = AdaBoostClassifier(DTC)

    ada_param_grid = {"base_estimator__criterion": ["entropy"],
                      "base_estimator__splitter": ["best"],
                      "algorithm": ["SAMME.R"],
                      "n_estimators": [500],
                      "learning_rate": [.01],
                      "random_state": [42]}

    gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsadaDTC.fit(X_train, Y_train)

    ada_best = gsadaDTC.best_estimator_

    print gsadaDTC.best_score_
    print "AB is %s" %ada_best
    feature_imp_sorted_ab = pd.DataFrame({'feature': list(X_train),
                                          'importance': gsadaDTC.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ab = feature_imp_sorted_ab.head(250)['feature']
    #
    # #ExtraTrees
    ExtC = ExtraTreesClassifier()

    # Search grid for optimal parameters
    ex_param_grid = {"max_depth": [70],
                     "max_features": ["sqrt"],
                     "n_estimators": [400],
                     "criterion": ["entropy"],
                     "random_state": [42]}

    gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsExtC.fit(X_train, Y_train)
    ExtC_best = gsExtC.best_estimator_

    # Best score
    print gsExtC.best_score_
    print "ET is %s" %ExtC_best
    feature_imp_sorted_et = pd.DataFrame({'feature': list(X_train),
                                          'importance': gsExtC.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(250)['feature']
    #
    # #RFC Parameters tunning
    RFC = RandomForestClassifier()

    ## Search grid for optimal parameters
    rf_param_grid = {"max_features": ["sqrt"],
                     "min_samples_split": [2],
                     "min_samples_leaf": [1],
                      "random_state": [42],
                     "n_estimators": [500],
                     "criterion": ["gini"]}

    gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsRFC.fit(X_train, Y_train)

    RFC_best = gsRFC.best_estimator_

    # Best score
    print gsRFC.best_score_
    print "RF is %s" % RFC_best
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(X_train),
                                          'importance': gsRFC.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(250)['feature']

    # Gradient boosting tuning


    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     'n_estimators': [500],
                     'learning_rate': [0.1],
                     'max_depth': [3],
                     "random_state": [42],
                     'min_samples_leaf': [1],

                     }

    gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsGBC.fit(X_train, Y_train)


    GBC_best = gsGBC.best_estimator_

    # Best score
    print gsGBC.best_score_
    print "GBC %s" % GBC_best
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(X_train),
                                          'importance': gsGBC.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(250)['feature']

    # SVC classifier
    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'],
                      'gamma': [0.01],
                      "random_state": [42],
                      'C': [60]}

    gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsSVMC.fit(X_train, Y_train)

    SVMC_best = gsSVMC.best_estimator_

    #Best score
    print gsSVMC.best_score_
    print "SVC  %s" % SVMC_best

    # Logistic classifier
    logitC = LogisticRegression(intercept_scaling=1,
               dual=False, fit_intercept=True, penalty='l1', tol=0.0001)
    logit_param_grid = {'max_iter': [4000]}

    gslogitC = GridSearchCV(logitC, param_grid=logit_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gslogitC.fit(X_train, Y_train)

    logitC_best = gslogitC.best_estimator_

    #Best score
    print gslogitC.best_score_
    print "SVC  %s" % logitC_best


    # Naive Bayes classifier
    gaussC = GaussianNB()
    # logit_param_grid = {'max_iter': [4000]}

    gsGausC = GridSearchCV(gaussC, param_grid={}, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsGausC.fit(X_train, Y_train)

    gsGausC_best = gsGausC.best_estimator_

    #Best score
    print gsGausC.best_score_
    print "SVC  %s" % gsGausC_best

    # KNN
    knnC = KNeighborsClassifier()
    knn_grid = {'algorithm': ["ball_tree"],
                "n_neighbors":[5]}

    gsknnC = GridSearchCV(knnC, param_grid=knn_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsknnC.fit(X_train, Y_train)

    gsknnC_best = gsknnC.best_estimator_

    #Best score
    print gsknnC.best_score_
    print "SVC  %s" % gsknnC_best

    # MLP
    mlpC = MLPClassifier()
    mlp_grid = {
        'learning_rate': ["adaptive"],
        'hidden_layer_sizes': [40],
        'activation': ["logistic"],
        'random_state':[42]
    }

    gsmlpC = GridSearchCV(mlpC, param_grid=mlp_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gsmlpC.fit(X_train, Y_train)

    gsmlpC_best = gsmlpC.best_estimator_

    #Best score
    print gsmlpC.best_score_
    print "SVC  %s" % gsmlpC_best


    features_top_n = pd.concat([features_top_n_gb,features_top_n_rf,features_top_n_et,features_top_n_ab], \
                               ignore_index = True).drop_duplicates()
    print "Picked Features: " + str(features_top_n.shape)
    X_train=X_train[features_top_n]
    test=test[features_top_n]
    # #
    #
    # votingC = VotingClassifier(estimators=[('gbc', GBC_best)], voting='soft',
    # #                            n_jobs=4)
    votingC = VotingClassifier(estimators=[('gbc', GBC_best),('et',ExtC_best),\
                                           ('rf',RFC_best),('mlp',gsmlpC_best),('logit',logitC_best),\
                                           ('knn',gsknnC_best),('svc',SVMC_best),('ada',ada_best)], \
                               voting='soft',n_jobs=4)

    gVoting = GridSearchCV(votingC, param_grid={}, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

    gVoting.fit(X_train, Y_train)
    print gVoting.best_score_

    test_Survived = pd.Series(gVoting.predict(test), name="Survived")
    results = pd.concat([IDtest, test_Survived], axis=1)

    results.to_csv("data/ensemble_python_voting.csv", index=False)