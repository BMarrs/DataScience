import io
import zipfile as zf
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA

# set the window size for plotting
rcParams['figure.figsize'] = 10, 6
# static paths to the data
RAWPATH = 'C:/Problem 2 Input Data.zip'
RAWFILE = 'data.csv'


def zip_reader(path, file):
    """Function for reading in the contents of the zip file and creating a dataframe"""
    with zf.ZipFile(path) as z_dir:
        # convert the zip directory to a seekable in-memory file type
        zip_data = io.BytesIO(z_dir.read(file))
        return pd.read_csv(zip_data, low_memory=False, sep=',', header=0)


def data_create(set_type, source):
    """Function for creating test and train data sets"""
    if set_type == 'train':
        return source.loc[source['train'] == 1]
    elif set_type == 'test':
        return source.loc[source['train'] != 1]


def data_prop(src_table, src_name):
    """Function for printing attributes of a provided dataframe"""
    print("%s table is composed of rows/columns: %s" % (src_name, format(src_table.shape)))

# Open the raw data
raw_input = zip_reader(RAWPATH, RAWFILE)
# look at the variance and summary statistics for the data set.
print(raw_input.describe())
# look at the first few rows of the dataframe
print(raw_input[:5])
# Create the training data set and the test data set
training_data = data_create('train', raw_input)
test_data = data_create('test', raw_input)
# check the frame properties
data_prop(raw_input, 'Raw_Input')
data_prop(training_data, 'Training_data')
data_prop(test_data, 'Test_data')
# separate out the predictors from the results on the training set
result_train = training_data['target_eval']
# let pandas know that the table is not a copy to turn off the warning messages for this object only.
training_data.is_copy = False
# drop out the non-predictor variables from the training data.
training_data.drop(['id', 'target_eval', 'train'], inplace=True, axis=1)
# separate out the test data id
id_test = test_data['id']
test_data.is_copy = False
test_data.drop(['id', 'target_eval', 'train'], inplace=True, axis=1)
# create a training split of 80/20 on the training_data
x_train, x_test, y_train, y_test = train_test_split(
    training_data,
    result_train,
    train_size=0.8,
    stratify=result_train)


def alg_chooser(scoring_eval):
    """Evaluate a list of algorithms to return the most accurate result"""
    # provide a list of analysis algorithms to loop through and compare performance.
    names = [
        'Naive Bayes',
        'K Nearest Neighbors',
        'Random Forest',
        'Gradient Boosted Forest',
        'Logistic Regression',
        'Quadratic Discrimination',
        'XGB Forest',
        'Linear Discriminant',
        'Linear SVC',
        'RBF SVM',
        'Linear SVM',
        'AdaBoost',
        'SDC',
        'Decision Tree'
    ]
    # create a list of algorithms with some default settings
    alg_list = [
        GaussianNB(),
        KNeighborsClassifier(3),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        GradientBoostingClassifier(),
        LogisticRegression(),
        QuadraticDiscriminantAnalysis(),
        XGBClassifier(),
        LinearDiscriminantAnalysis(),
        LinearSVC(),
        SVC(gamma=2, C=1),
        SVC(kernel='linear', C=0.025),
        AdaBoostClassifier(),
        SGDClassifier(),
        DecisionTreeClassifier(max_depth=5)]
    # create an empty array to store the results
    alg_report = {}
    # loop through the algorithms and compare them.
    for name, classf in zip(names, alg_list):
        # get a time start timestamp
        start_time = time.time()
        # fit the training data
        classf.fit(x_train, y_train)
        # get a time end timestamp
        end_time = time.time()
        # evaluate the predictions
        predictions = classf.predict(x_test)
        # score it
        if scoring_eval == 'F1':
            scored = f1_score(y_test, predictions)
        elif scoring_eval == 'ROC':
            try:
                scored = roc_auc_score(y_test, classf.predict_proba(x_test)[:, 1], average='macro')
            except AttributeError:
                continue
        alg_report[name] = (str(end_time - start_time) + ' seconds', scored)
    # return the sorted list of algorithms based on success at predicting.
    return sorted(alg_report.items(), key=lambda x: x[1][1], reverse=True)

# Get the list of the different algorithms, sorted by efficacy.
scoring = alg_chooser('ROC')
for elem in scoring:
    print(elem)

# scale the data with a fit transform
scaled = StandardScaler()
x_train_data_scaled = scaled.fit_transform(x_train)
# principal component analysis
pca = PCA()
pca.fit(x_train_data_scaled)
ratio = pca.explained_variance_ratio_

# plot the variance explained by Principal components vs. the cumsum of explained variance
x_axis = np.arange(x_train_data_scaled.shape[0])
ax = plt.subplot()
ax.plot(x_axis, np.cumsum(ratio), '-o')
plt.ylim(0.0, 1.0)
plt.xlabel('Number of PCAs')
plt.ylabel('Cumulative Sum of Variance (explained)')
plt.title('PCA Variance')
plt.show()

# XGBoost (Gradient Boosted Forest) has one of the consistently highest area under ROC results.
# although it is among the slowest.  The best trade-off for speed would be Linear Discriminant Analysis.
# set the parameters to use for the model based on tuning iterations
xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 500,
    'seed': 80,
    'subsample': 0.8,
    'objective': 'binary:logistic',
    'min_child_weight': 3
}
# cv_parameters for model tuning.
cv_params = {
    # 'learning_rate': [0.01, 0.1],
    # 'n_estimators': [500, 1000, 1500],
    # 'max_depth': [3, 5, 7],
    # 'min_child_weight': [1, 3, 5],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
     'seed': [70, 80, 90]
}
# use the XGBClassifier
model = GridSearchCV(XGBClassifier(**xgb_params), cv_params, scoring='accuracy', cv=5, n_jobs=-1, refit=True)
# start a timer
model_start = time.time()
# run the model
model.fit(x_train, y_train)
# end the timer
model_end = time.time()
model_score = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1], average='macro')
# print the grid scores
print(model.grid_scores_)
# create a XGB DMatrix
xgdmat = xgb.DMatrix(x_train, y_train)
# set the final model
our_params = {
    'eta': 0.1,
    'seed': 80,
    'max_depth': 3,
    'n_estimators': 500,
    'subsample': 0.8,
    'min_child_weight': 3,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic'
}
final_model = xgb.train(our_params, xgdmat, num_boost_round=450)
# plot the importances
importance = final_model.get_fscore()
importance_frame = pd.DataFrame({'Importance': list(importance.values()), 'Feature': list(importance.keys())})
importance_frame.sort_values(by='Importance', inplace=True)
importance_frame.plot(kind='barh', x='Feature', figsize=(8, 20))
plt.show()
print("The results are: %s in %s seconds" % (format(model.best_estimator_), str(model_end - model_start)))
# xgboost accuracy test
testdmat = xgb.DMatrix(x_test)
y_pred = final_model.predict(testdmat)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0

print(accuracy_score(y_pred, y_test))
# now make the predictions
lbl_output = model.predict_proba(test_data)[:, 1]
# convert to a dataframe
lbl_output = pd.Series(lbl_output, index=np.arange(250, 20000))
# combine back with the original data
output_file = pd.concat([id_test, lbl_output], axis=1)
# rename the probability column
output_file.columns = ['id', 'Probability_Positive']
# create a predicted output column.
output_file['Predicted'] = np.where(output_file['Probability_Positive'] >= 0.5, 1, 0)
# save to disk
output_file.to_csv('C:/Model_Result/ModelOutput.csv', index=False)


