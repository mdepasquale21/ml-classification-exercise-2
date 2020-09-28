# Try different Classifiers

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from evaluate import *
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import sys

model_names = ('logit', 'knn', 'adaboost', 'rf', 'naive_bayes', 'svm', 'LDA', 'QDA')

try:
    model_name = sys.argv[1]
except:
    print("\nYou must specify a classifier method, using naive bayes as default one.\n")
    model_name = 'naive_bayes'

# Importing the dataset
dataset = pd.read_csv('./data/data.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

################################################################################################################################
print('Dropping User ID and encoding Gender...')
dataset = dataset.drop(columns=['User ID'])
dataset['Gender'].replace({"Male": 1, "Female": 0}, inplace=True)
print('DATA EXPLORATION')
print('SHAPE')
print(dataset.shape)
print('INFO')
dataset.info()
print('DESCRIPTION')
print(dataset.describe())
n_rows_head = 5
print('FIRST ' + str(n_rows_head) + ' ENTRIES')
print(dataset.head(n_rows_head))
print('\n')
################################################################################################################################

# Select Classifier
if model_name == model_names[0]:
    print('Using Logistic Regression.')
    classifier = LogisticRegression(class_weight='balanced')
elif model_name == model_names[1]:
    print('Using K-Nearest Neighbors Classifier.')
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
elif model_name == model_names[2]:
    print('Using AdaBoost Classifier.')
    classifier = AdaBoostClassifier()
elif model_name == model_names[3]:
    print('Using Random Forest Classifier.')
    classifier = RandomForestClassifier(class_weight='balanced')
elif model_name == model_names[4]:
    print('Using Naive Bayes Classifier.')
    classifier = GaussianNB()
elif model_name == model_names[5]:
    print('Using Support Vector Machines Classifier.')
    classifier = SVC(probability=True, class_weight='balanced')
elif model_name == model_names[6]:
    print('Using Linear Discriminant Analysis.')
    classifier = LinearDiscriminantAnalysis()
elif model_name == model_names[7]:
    print('Using Quadratic Discriminant Analysis.')
    classifier = QuadraticDiscriminantAnalysis()
else:
    print('Unknown option for classifier, using naive bayes as default one!')
    model_name = 'naive_bayes'
    classifier = GaussianNB()

# Feature Scaling
sc = StandardScaler()

print('Using Repeated Stratified K-Fold Cross Validation.\n')
# Using Repeated Stratified K-Fold Cross Validation
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=36851234)

accuracy_list = []
pr_auc_list = []
roc_auc_list = []
f05_list = []
f1_list = []
precision_list = []
recall_list = []
hl_list = []

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Fitting model to the Training set
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    probabilities_1 = classifier.predict_proba(X_test)[:,1] # take only positive probabilities

    accuracy = get_accuracy(y_test, y_pred)
    accuracy_list.append(accuracy)

    pr_auc = get_pr_auc(y_test, probabilities_1)
    pr_auc_list.append(pr_auc)

    roc_auc = get_roc_auc(y_test, probabilities_1)
    roc_auc_list.append(roc_auc)

    f05 = get_f05(y_test, y_pred)
    f05_list.append(f05)

    f1 = get_f1(y_test, y_pred)
    f1_list.append(f1)

    precision = get_precision(y_test, y_pred)
    precision_list.append(precision)

    recall = get_recall(y_test, y_pred)
    recall_list.append(recall)

    hamming_loss = get_hamming_loss(y_test, y_pred)
    hl_list.append(hamming_loss)

print('Performed ' + str(len(accuracy_list)) + ' different training + testing experiments.')
print(model_name.upper() + ' AVERAGE PERFORMANCE:')
print('ACCURACY')
print(np.mean(accuracy_list))
print('PR AUC')
print(np.mean(pr_auc_list))
print('ROC AUC')
print(np.mean(roc_auc_list))
print('F0.5')
print(np.mean(f05_list))
print('F1')
print(np.mean(f1_list))
print('PRECISION')
print(np.mean(precision_list))
print('RECALL')
print(np.mean(recall_list))
print('HAMMING LOSS')
print(np.mean(hl_list))
