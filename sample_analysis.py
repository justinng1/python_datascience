import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import itertools
import random

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (12,8)

random.seed(584)

# load the data.
test = pd.read_csv('../data/test_features_2013-03-07.csv')
train_features = pd.read_csv('../data/train_features_2013-03-07.csv')
train_salaries = pd.read_csv('../data/train_salaries_2013-03-07.csv')

# do find out a little bit of information about the data.
train_features.head()
train_features.shape
train_features.dtypes

train_salaries.head()
train_salaries.shape
train_salaries.dtypes

test.head()
test.shape
test.dtypes

# are all the companies present in train and test?
test_company = test['companyId'].unique()
train_company = train_features['companyId'].unique()
np.in1d(test_company, train_company)
np.in1d(train_company, test_company)

# Note, I did a lot more data exploration and validation but did not put
# show it here to keep things simple.


# merge the features and salaries together.
train = pd.merge(train_features, train_salaries, on='jobId', how='inner')
train = train.drop(['jobId'], axis=1)

del train_features
del train_salaries

# Create a logSalary to predict.
train['logSalary'] = np.log(train.salary+0.5)


#######################################################################
# this function takes plots a histogram of the target variable grouped by levels in 
# a categorical variable.
# 
# inputs:
#    df - the pandas dataframe.
#    varName - the variable name of the categorical variable.
#    targetName - the variable name of the target.
#
# outputs:
#   plots a histogram.
#######################################################################
def histogram_by_level(df, varName, targetName):
    lvls = df[varName].value_counts()
    for i in lvls.index:
        plt.hist(df[df[varName] == i][targetName], alpha=0.5, label=i, bins=300)

    plt.legend(loc='upper right')
    plt.show()


histogram_by_level(train, 'companyId', 'salary')
histogram_by_level(train, 'jobType', 'salary')
histogram_by_level(train, 'degree', 'salary')
histogram_by_level(train, 'major', 'salary')
histogram_by_level(train, 'industry', 'salary')

#######################################################################
# implements riser encoding on all the categorical variables.
# Instead of using dummy variables to encode categorical variables
# I replace each level with the average salary for those levels.
# This could be more effective than dummy encoding for tree models when 
# there are a lot of levels in a category.
# 
# inputs:
#    df - the dataframe.
#    col_list - the list of variables to encode.
#    targetName - the target name (salary for this project).
# output:
#    ret_dict - a dictionary used to encode the variables. 
#               Used in the Dataframe.replace method.
#######################################################################
def riser_enc_dict(df, col_list, targetName):
    ret_dict = {}
    for col in col_list:
        gb = df.groupby([col])
        mean = gb[targetName].mean()
        riser_enc = {}
        
        for l in mean.index:
            riser_enc[l] = mean[l]
            
        ret_dict[col] = riser_enc
        
    riser_enc = {}
    return ret_dict

#######################################################################
# does kfold cross validation.
# The built in function cross_val_score in sklearn didn't quite fit
# what I was trying to test so I created my own function.
# 
# inputs:
#    clf - the classifier.
#    df - the dataframe.
#    feature_names - the list of features to use in the model fitting.
#    targetName - the target name (salary for this project).
#    nsplits - the number of folds in the cross validation.
#    enc_type - either 'riser' or 'dummy'. The encoding method for 
#               categorical variables.
# output:
#    train_scores - a list of scores on the training sets (rmse)
#    val_scores - a list of scores on the validation sets (rmse)
#######################################################################
def kfold_cv(clf, df, feature_names, targetName, nsplits, enc_type='riser'):
    val_scores = []
    train_scores = []
    
    feature_names.append(targetName)
    df = df[feature_names].copy()
    cols_to_enc = df.select_dtypes(include='object').columns
    
    if enc_type != 'riser':
        df = pd.get_dummies(df, columns=cols_to_enc)
        
    kf = KFold(n_splits=nsplits, shuffle=True)
    for train_ind, val_ind in kf.split(df):
        
        train = df.iloc[train_ind]
        val = df.iloc[val_ind]
        
        if enc_type == 'riser':
            # riser encoding.
            riser_dict = riser_enc_dict(train, cols_to_enc, targetName)
            train = train.replace(riser_dict, inplace=False)
            val = val.replace(riser_dict, inplace=False)

        
        X_train = train.drop([targetName], axis=1)
        y_train = train[targetName]
        X_val = val.drop([targetName], axis=1)
        y_val = val[targetName]
    
        clf.fit(X_train, y_train)
        
        y_val_pred = clf.predict(X_val)
        metric = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_scores.append(metric)
        
        y_train_pred = clf.predict(X_train)
        metric = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_scores.append(metric)      
        
    return train_scores, val_scores

# the features to use.
feature_names = train.drop(['logSalary', 'salary'], axis=1).columns # everything except the salaries.
num_features = len(feature_names)


# test linear regression models.
clf1 = linear_model.LinearRegression()

#
# feature selection
# Since aren't too many features we can afford to do an exhaustive fit of the variables.
# This does k-fold CV for the model on all possible variable combinations.
#
for num in range(1,num_features+1):
    it = itertools.combinations(feature_names, num) # choose the number of features to include
    for features in it:
        train_scores, val_scores = kfold_cv(clf1, train, list(features), 'logSalary', nsplits=5, enc_type='dummy')
        print(features)
        print("Training Accuracy: %0.5f (+/- %0.5f)" % (np.mean(train_scores), np.std(train_scores) * 2))
        print("Validation Accuracy: %0.5f (+/- %0.5f)" % (np.mean(val_scores), np.std(val_scores) * 2))


# test decision tree models.
clf2 = tree.DecisionTreeRegressor(min_samples_leaf=5)
train_scores, val_scores = kfold_cv(clf2, train, list(feature_names), 'logSalary', nsplits=5, enc_type='riser')
print("Training Accuracy: %0.5f (+/- %0.5f)" % (np.mean(train_scores), np.std(train_scores) * 2))
print("Validation Accuracy: %0.5f (+/- %0.5f)" % (np.mean(val_scores), np.std(val_scores) * 2))

# test random forest models.
clf3 = ensemble.RandomForestRegressor(n_estimators=100)
train_scores, val_scores = kfold_cv(clf3, train, list(feature_names), 'logSalary', nsplits=5, enc_type='riser')
print("Training Accuracy: %0.5f (+/- %0.5f)" % (np.mean(train_scores), np.std(train_scores) * 2))
print("Validation Accuracy: %0.5f (+/- %0.5f)" % (np.mean(val_scores), np.std(val_scores) * 2))


# Random Forest model seems to be the best. Use it as the final predictor.
clf = ensemble.RandomForestRegressor(n_estimators=100)
cols_to_enc = train.select_dtypes(include='object').columns
riser_dict = riser_enc_dict(train, cols_to_enc, 'salary')

X = train[feature_names]
X = X.replace(riser_dict, inplace=False)
y = train['logSalary']

clf.fit(X, y)

# look at the feature importances.
cols = X.columns
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature (%d) %s (%f)" % (f + 1, indices[f], cols[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# do the predictions on the test set.
X_test = test[feature_names]
X_test = X_test.replace(riser_dict, inplace=False)

y_test_pred = np.exp(clf.predict(X_test)) - 0.5 # clf predits log(salary). transform back to salary.
y_test_pred_df = pd.DataFrame(y_test_pred, columns=['salary'])

result = pd.concat([test['jobId'], y_test_pred_df], axis=1, join='inner')


# check on some of the test results to see if it makes sense.
test2 = pd.merge(test, result, on='jobId', how='inner')
test2.head()

# do histograms look comparable to the training set?
histogram_by_level(test2, 'major', 'salary')

# save to csv file.
result.to_csv('test_salaries.csv', index=False)
