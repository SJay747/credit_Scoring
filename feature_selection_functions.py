from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression


# Recursive Feature Importance
def RFI(X, y):
    model = RandomForestRegressor(n_estimators=500, random_state=1)  # initialising a random regressor
    model.fit(X, y)
    names = X.columns.values
    ticks = [i for i in range(len(names))]

    return model.feature_importances_, ticks, names

def plot_RFI(X, i, t, n):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(t, i)
    ax.grid()
    plt.xticks(t, n)
    plt.xticks(range(len(t)), list(X.columns), rotation="40")
    st.write(fig)

def bottom_n_features(importances, n):
    bottom = []
    sorted_importances = np.sort(importances)
    for i in range(n):
        bottom.append(np.where(importances == sorted_importances[i])[0][0])

    return bottom

def top_n_features(importances, n):
    top = []
    full_length = len(importances) - 1
    sorted_importances = np.sort(importances)
    for i in range(full_length, full_length - n, -1):
        top.append(np.where(importances == sorted_importances[i])[0][0])

    return top

# Mutual Information Classifier
def muc(X, y):
    importances = mutual_info_classif(X, y)
    feat_importances = pd.Series(importances, X.columns[0:len(X.columns)])

    return feat_importances

def plot_muc(X, importances):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    importances.plot(kind="barh", color="teal")
    ax.set_title("Mututal Information Classification")
    ax.grid()
    #plt.show()
    return fig

# Correlation
def return_corr(X):

    corr = X.corr()
    return corr

# Principal Component Analysis
def PCA(n_comp, X):
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    return X_pca

def PCA_percentages(X):
    percentages = []
    cov = X.cov()
    eig_vals, eig_vecs = np.linalg.eig(cov)

    for c in range(len(X.columns)):
        PC = []
        for i in range(c):
            PC.append(100 * eig_vals[i] / np.sum(eig_vals))

        percentages.append(sum(PC))

    return percentages

def plot_pca_percentages(perc):
    plt.figure(figsize=(12, 6))
    plt.plot(perc)
    plt.title("Percentage of Variance Explained by Number of PCA Components")
    plt.ylabel("Percentage of Variance Explained")
    plt.xlabel("PCA Components")
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, 28), np.array(perc[1:]) - np.array(perc[0:-1]))
    plt.title("Incremental amount of variance explained per PCA Component")
    plt.ylabel("Incremental Percentage of Variance Explained")
    plt.xlabel("PCA Components")
    plt.grid()
    plt.show()

    return

# LASSO Regularization
def lasso_l1(X, y):
    # fit logistic regression model
    logistic = LogisticRegression(C=1, penalty="l1", solver="liblinear", random_state=7).fit(X, y)

    # obtain transformer from model - contains feature importance attribute to apply transformation with
    model = SelectFromModel(logistic, prefit=True)

    # applies transform and maintains a reduced number of columns
    X_new = model.transform(X)

    return X_new

# Model Initialisation
def initialise_models():
    # Function that initialises 6 classification models
    model_names = ["CART", "SVM", "BAG", "RF", "GBM", "XGB", "LR"]

    models = []
    models.append(DecisionTreeClassifier())
    models.append(SVC(gamma='scale'))
    models.append(BaggingClassifier(n_estimators=100))
    models.append(RandomForestClassifier(n_estimators=100))
    models.append(GradientBoostingClassifier(n_estimators=100))
    models.append(XGBClassifier())
    models.append(LogisticRegression())

    return models, model_names

# Model Evaluation
def evaluate_baseline_models(models, model_names, X, y):
    accuracy = []
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    results = []

    for i in range(len(models)):
        scores = cross_val_score(models[i], X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        results.append(scores)
        accuracy.append(np.mean(results[i]))

    return accuracy, results


def plot_accuracies(accuracies, ticks, results):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(np.arange(len(accuracies)), accuracies)
    ax[0].grid()
    plt.sca(ax[0])
    plt.xticks(ticks=np.arange(len(accuracies)), labels=ticks)

    ax[1].boxplot(results, labels=ticks, showmeans=True)
    ax[1].grid()

    return fig

def min_max_scale(df):
    # takes numerical feature matrix as input and scales using MinMaxScaler
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    return df

def PCA_percentages(X):
    percentages = []
    cov = X.cov()
    eig_vals, eig_vecs = np.linalg.eig(cov)

    for c in range(len(X.columns)):
        PC = []
        for i in range(c):
            PC.append(100 * eig_vals[i] / np.sum(eig_vals))

        percentages.append(sum(PC))

    return percentages