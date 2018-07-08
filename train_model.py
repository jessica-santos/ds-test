import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from ReliefF import ReliefF
from imblearn.combine import SMOTEENN

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc
from scipy import interp

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import feature_engineering as ft



def feature_selection_reliefF(X,Y):
    featReliefF = ReliefF()
    featReliefF.fit(X,Y)
    features = featReliefF.top_features;
    score = featReliefF.feature_scores
    n_feat = len(score[score > 500000])
    print(features[:n_feat])
    X[:, features[:n_feat]]
    return score, X

def resample_classes(X,y):
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_sample(X, y)
    return X_resampled, y_resampled

def classification_model(X, y, model, name):
    model.fit(X, y)
    print(name + " result: \n")
    y_prob = model.predict_proba(X)
    #fpr, tpr = roc_curve(y, y_prob)
    #roc_auc = auc(fpr, tpr)
    #print(roc_auc)

    if name == "Random Forest":
        feat_score = model.feature_importances_
        plt.plot(feat_score)
        print(feat_score.argsort())
        X = SelectFromModel(model, prefit=True).transform(X)
        print(X)

    return model, y_prob, X

def classification_model_cv(X, Y, model, name):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    kf = StratifiedKFold(n_splits=10, shuffle=False)
    i=0
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        print(y_test.value_counts())
        model.fit(X_train, y_train)
        print(name+" result: \n")
        probas_ = model.predict_proba(X_test)
        # Create confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        aucs.append(roc_auc)
        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print(mean_auc)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC '+name+' (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return mean_auc

def classification_rf(X,Y, cv, balanceada, nome):
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=2, random_state=0)
    if balanceada:
        rf_model = RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=2, random_state=0, class_weight = "balanced")
    if cv:
        return classification_model_cv(X,Y,rf_model,"Random Forest "+nome)
    else:
        return classification_model(X,Y,rf_model,"Random Forest "+nome)

def classification_log_reg(X, Y,nome):
    lg_model = LogisticRegression()
    classification_model_cv(X, Y, lg_model, "Logistic Regression "+nome)

def classification_naive_bayes(X, Y, nome):
    nb_model = BernoulliNB()
    classification_model_cv(X, Y, nb_model, "Naive Bayes "+nome)

def classification_voting(X,y, nome):
    clf2 = RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=2, random_state=0)
    clf3 = BernoulliNB()
    eclf2 = VotingClassifier(estimators=[('rf', clf2), ('bnb', clf3)],voting = 'soft')
    classification_model_cv(X, y, eclf2, "Voting Model "+nome)

def evaluate_classifications(X_train, y_train):
    #Selecionar com RF
    model, prob, Xs_rf = classification_rf(X_train, y_train, False, False, 'select')
    #Selecionar com ReliefF
    score, Xs_r = feature_selection_reliefF(X_train.as_matrix(), y_train)

    #Classificar com RF
    classification_rf(X_train, y_train, True, True, 'all_balanc')
    classification_rf(X_train, y_train, True, False, 'all')
    classification_rf(pd.DataFrame(Xs_rf), y_train, True, True, 'selRF_balanc')
    classification_rf(pd.DataFrame(Xs_rf), y_train, True, False, 'selRF')
    classification_rf(pd.DataFrame(Xs_r), y_train, True, True, 'selReliefF_balanc')
    classification_rf(pd.DataFrame(Xs_r), y_train, True, False, 'selReliefF')

    #Classificar com RG
    classification_log_reg(X_train, y_train, 'all')
    classification_log_reg(pd.DataFrame(Xs_rf), y_train, 'selRF')
    classification_log_reg(pd.DataFrame(Xs_r), y_train, 'selReliefF')

    #Classificar com NB
    classification_naive_bayes(X_train, y_train, 'all')
    classification_naive_bayes(pd.DataFrame(Xs_rf), y_train, 'selRF')
    classification_naive_bayes(pd.DataFrame(Xs_r), y_train, 'selReliefF')

    #Classificar com voting
    classification_voting(X_train, y_train, 'all')
    classification_voting(pd.DataFrame(Xs_rf), y_train, 'selRF')

    classification_voting(pd.DataFrame(Xs_r), y_train, 'selReliefF')

def evaluate_classifications_balanced(X_train, y_train):
    Xb, yb = resample_classes(X_train, y_train)
    yb = pd.DataFrame(yb.astype('float'))[0]
    Xb = pd.DataFrame(Xb)
    evaluate_classifications(Xb,yb)

if __name__=='__main__':
    #Read dataframe
    df = pd.read_csv('../input/test_data.csv')

    df = ft.treat_variables(df)

    X = df.drop(columns='class')
    Y = df['class']

    evaluate_classifications(X,Y)
    evaluate_classifications_balanced()