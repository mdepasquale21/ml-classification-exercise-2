import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, balanced_accuracy_score, hamming_loss
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import make_scorer

def evaluate(model, test_features, test_labels):
    """
    This function evaluates mean absolute percentage error of a given model.
    It also returns the accuracy score.
    Both are printed during execution.
    """
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = accuracy_score(test_labels, predictions)
    print('Model Performance')
    print('Average Error: {:0.4f}'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

def get_accuracy(y_true, y_pred):
    """
    This function just wraps sklearn accuracy score.
    """
    return accuracy_score(y_true, y_pred)

def get_balanced_accuracy(y_true, y_pred):
    """
    This function just wraps sklearn balanced accuracy score.
    """
    return balanced_accuracy_score(y_true, y_pred)

type_of_average = 'binary'

def get_f1(y_true, y_pred):
    """
    This function just wraps sklearn f1 score.
    """
    return f1_score(y_true, y_pred, average=type_of_average)

def get_f05(y_true, y_pred):
    """
    This function just wraps sklearn fbeta score with beta=0.5.
    """
    return fbeta_score(y_true, y_pred, beta=0.5, average=type_of_average)

def get_precision(y_true, y_pred):
    """
    This function just wraps sklearn precision score.
    """
    return precision_score(y_true, y_pred, average=type_of_average)

def get_recall(y_true, y_pred):
    """
    This function just wraps sklearn recall score.
    """
    return recall_score(y_true, y_pred, average=type_of_average)

def get_roc_auc(y_true, y_score):
    """
    This function just wraps sklearn roc curve and auc.
    """
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_pr_auc(y_true, y_score):
    """
    This function just wraps sklearn pr curve and auc.
    """
    p, r, thresh = precision_recall_curve(y_true, y_score)
    pr_auc = auc(r, p)
    return pr_auc

def get_false_p_true_p_arrays(y_true, y_score):
    """
    This function just wraps sklearn roc curve to get false and true positive rates.
    """
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    return fpr, tpr

def get_precision_recall_arrays(y_true, y_score):
    """
    This function just wraps sklearn pr curve to get precision and recall values.
    """
    p, r, thresh = precision_recall_curve(y_true, y_score)
    return p, r

def get_hamming_loss(y_true, y_pred):
    """
    This function just wraps sklearn hamming loss.
    """
    return hamming_loss(y_true, y_pred)

def print_all_metrics(y_pred,probs,y_true):
    """
    This function prints all metrics.
    """
    print('ACCURACY')
    accuracy = get_accuracy(y_true, y_pred)
    print(accuracy)
    print('PR AUC')
    pr_auc = get_pr_auc(y_true, probs)
    print(pr_auc)
    print('ROC AUC')
    roc_auc = get_roc_auc(y_true, probs)
    print(roc_auc)
    print('F0.5')
    f05 = get_f05(y_true, y_pred)
    print(f05)
    print('F1')
    f1 = get_f1(y_true, y_pred)
    print(f1)
    print('PRECISION')
    precision = get_precision(y_true, y_pred)
    print(precision)
    print('RECALL')
    recall = get_recall(y_true, y_pred)
    print(recall)
    print('HAMMING LOSS')
    hamming_loss = get_hamming_loss(y_true, y_pred)
    print(hamming_loss)

def calc_learning_curve(cv, X, y, model, metrics_func):
    """
    This function evaluates a model's learning curve using a given Cross Validation with a
    given metrics function.
    """
    # define the model evaluation metric
    metric = make_scorer(metrics_func)
    # evaluate learning curve
    # learning_curve returns train_sizes, train_scores, test_scores
    # and fit_times, score_times only if return_times=True (default return_times=False)
    # using default value for train_sizes that is np.linspace(0.1, 1.0, 5)
    lc = learning_curve(estimator=model, X=X, y=y, cv=cv, scoring=metric)
    return lc

def cv_evaluate_model(cv, X, y, model, metrics_func):
    """
    This function evaluates a model using a given Cross Validation with a
    given metrics function.
    """
    # define the model evaluation metric
    metric = make_scorer(metrics_func)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
    return scores
