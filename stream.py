import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from function import load_data, WoE, woe_modify, one_hot_encode_modify
from feature_selection_functions import RFI, plot_RFI, min_max_scale, bottom_n_features, top_n_features, muc, plot_muc, initialise_models, evaluate_baseline_models, PCA_percentages, plot_accuracies
from PIL import Image
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


#st.set_page_config(layout="wide")

filename = "Credit Data.csv"
# labelling keys for employment status
keys_aes = ["retired", "private sector", "others", "self employed", "public sector", "government", "student",
        "unemployed", "housewife", "military", "no response"]
keys_res = ["owner", "other", "with parents", "tenant furnished", "tenant unfurnished"]


data = load_data(filename)
columns = data.columns
y = data.iloc[:, -1]

st.header("Credit Scoring Data Analysis and Prediction Project")
st.subheader("Looking at the available dataset:")
st.write(data.head())

st.subheader("Overview of the variables:")
st.text('The dataset contains the following information: \nYOB: Year of Birth '
        '\nNKID: # of Kids '
        '\nDEP: # of Dependents'
        '\nPHON: Home Phone'
        '\nSINC Spousal Income'
        '\nAES: Employment Status'
        '\nDAINC: Income'
        '\nDHVAL: Value of Home'
        '\nDMORT: Mortgage Outstanding'
        '\nDOUTM: Outgoings on Mortgage'
        '\nDOUTL: Outgoings on Loans'
        '\nDOUTHP: Outgoings on Hire Purchases'
        '\nDOUTCC: Outgoings on Credit Cards'
        '\nBAD: Whether Applicant has bad (1) or good (0) credit'
        )

# WoE preparation
aes_cat = data["AES"].unique()
res_cat = data["RES"].unique()
W, IV_aes = WoE(data, aes_cat, "AES")
W2, IV_res = WoE(data, res_cat, "RES")

# making two dictionaries
WoE_aes = {keys_aes[i]: W[i] for i in range(len(keys_aes))}
WoE_res = {keys_res[i]: W[i] for i in range(len(keys_res))}
X_woe = woe_modify(data, aes_cat, res_cat, WoE_aes, WoE_res, keys_aes, keys_res, filename)

st.header("Data Preprocessing Methods and Feature Engineering")
st.subheader("1) Weight of Evidence Matrix")
st.markdown('Weight of Evidence (WoE) is used for categorical data. A number is associated with the each category based on its predictive power over the dependent variable measured by the following equations:')
st.markdown('good = % of applicants from that category that have good credit')
st.markdown('bad = % of applicants from that category that have bad credit')
st.markdown('WoE = ln(good/bad)')
st.markdown('The individual categories are replaced with their WoE values: ')
st.write(X_woe)

def inputs():
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].set_title("WoE Values for Employment Status Categorical Data")
        ax[0].bar(range(len(WoE_aes)), list(WoE_aes.values()), align='center')
        plt.sca(ax[0])
        plt.xticks(range(len(WoE_aes)), list(WoE_aes.keys()), rotation="40")
        plt.grid()

        ax[1].set_title("WoE Values for Resiential Status Categorical Data")
        ax[1].bar(range(len(WoE_res)), list(WoE_res.values()), align='center')
        plt.sca(ax[1])
        plt.xticks(range(len(WoE_res)), list(WoE_res.keys()), rotation="40")
        plt.grid()

        return fig

st.pyplot(inputs())

# One Hot Encoding preparation
X_ohe, y_ohe = one_hot_encode_modify(columns)

st.subheader("2) One Hot Encoding Categorical Values")
st.markdown('One Hot Encoding is another method for engineering numerical categorical values. E.g. a 8 category variable is converted into 8 further features, a one in each location represents a particular category.')
st.write(X_ohe)

# Correlation matrix
X_woe = woe_modify(data, aes_cat, res_cat, WoE_aes, WoE_res, keys_aes, keys_res, filename)
st.subheader("Correlation Matrix of Input Variables:")
corr = X_woe.corr()
corr.style.background_gradient(cmap='Pastel2')
st.write(corr)


st.subheader("Top Correlations with Key Variables")
# top 3 correlations for the output variables
x = st.slider('Select a number of correlations', value=3, max_value=len(corr))

def correlations():
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))

        ax[0,0].set_title("Output Variable")
        ax[0,0].bar(np.arange(0, x), corr["BAD"].reindex(corr["BAD"].abs().sort_values(ascending=False).index).drop("BAD", axis=0)[0:x])
        plt.sca(ax[0,0])
        plt.xticks(np.arange(0,x), corr["BAD"].reindex(corr["BAD"].abs().sort_values(ascending=False).index).drop("BAD", axis=0)[0:x].index)
        ax[0,0].grid()

        ax[0,1].set_title("Income")
        ax[0,1].bar(np.arange(0, x), corr["DAINC"].reindex(corr["DAINC"].abs().sort_values(ascending=False).index).drop("DAINC", axis=0)[0:x])
        plt.sca(ax[0,1])
        plt.xticks(np.arange(0,x), corr["DAINC"].reindex(corr["DAINC"].abs().sort_values(ascending=False).index).drop("DAINC", axis=0)[0:x].index)
        ax[0,1].grid()

        ax[1,0].set_title("Value of Home")
        ax[1,0].bar(np.arange(0, x), corr["DHVAL"].reindex(corr["DHVAL"].abs().sort_values(ascending=False).index).drop("DHVAL", axis=0)[0:x])
        plt.sca(ax[1,0])
        plt.xticks(np.arange(0, x), corr["DHVAL"].reindex(corr["DHVAL"].abs().sort_values(ascending=False).index).drop("DHVAL", axis=0)[0:x].index)
        ax[1,0].grid()

        ax[1, 1].set_title("Credit Card Outgoings")
        ax[1, 1].bar(np.arange(0, x), corr["DOUTCC"].reindex(corr["DOUTCC"].abs().sort_values(ascending=False).index).drop("DOUTCC",                                                                                    axis=0)[0:x])
        plt.sca(ax[1, 1])
        plt.xticks(np.arange(0, x), corr["DOUTCC"].reindex(corr["DOUTCC"].abs().sort_values(ascending=False).index).drop("DOUTCC", axis=0)[0:x].index)
        ax[1, 1].grid()

        return fig

st.pyplot(correlations())

st.header("Feature Selection Analysis")

st.subheader("Random Forest Importance on Input Features: Embedded Method")
X_woe = min_max_scale(X_woe)
X_woe = X_woe.iloc[:, :-1]
i, t, n = RFI(X_woe, y)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(t, i)
ax.grid()
plt.xticks(range(len(t)), columns[:-1], rotation="40")
st.pyplot(fig)

bottom = bottom_n_features(i, 5)
top = top_n_features(i, 5)

st.write("Top 5 Features: ")
st.write(pd.DataFrame(top).T)
st.write("Bottom 5 Features: ")
st.write(pd.DataFrame(bottom).T)

st.subheader("Mutual Information Classifier: Filter Method")
X_woe = woe_modify(data, aes_cat, res_cat, WoE_aes, WoE_res, keys_aes, keys_res, filename)
X_woe = X_woe.iloc[:, 0:-1]
importances = muc(X_woe, y)
fig = plot_muc(X_woe, importances)
st.pyplot(fig)

st.subheader("Principal Component Analysis")
st.write("The following plot shows us the amount of variance explained by the Principal Components.")

perc = PCA_percentages(X_woe)
fig = plt.figure()
plt.plot(perc)
plt.title("Percentage of Variance Explained by Number of PCA Components")
plt.ylabel("Percentage of Variance Explained")
plt.xlabel("PCA Components")
plt.grid()
st.pyplot(fig)

fig = plt.figure()
plt.plot(np.arange(1, 14), np.array(perc[1:]) - np.array(perc[0:-1]))
plt.title("Incremental amount of variance explained per PCA Component")
plt.ylabel("Incremental Percentage of Variance Explained")
plt.xlabel("PCA Components")
plt.grid()
st.pyplot(fig)

PCA_components = 4
pca = PCA(n_components=PCA_components)
X_pca = pca.fit_transform(X_woe)
st.write("The PCA reduced input matrix is given below (prior to scaling): ")
st.write(X_pca)

st.header("Model Initialisation and Evaluation")
st.write("We will evaluate the 6 benchmark models.")
st.subheader("Running the model evaluation over the WoE modified dataset:")
models, model_names = initialise_models()
X_woe = min_max_scale(X_woe)
accuracies, results = evaluate_baseline_models(models, model_names, X_woe, y)
fig = plot_accuracies(accuracies, model_names, results)
st.pyplot(fig)

st.subheader("Running the model evaluation over the PCA reduced dataset:")
models, model_names = initialise_models()
X_pca = min_max_scale(X_pca)
accuracies, results = evaluate_baseline_models(models, model_names, X_pca, y)
fig = plot_accuracies(accuracies, model_names, results)
st.pyplot(fig)

st.subheader("Running the model evaluation over the One Hot Encoded dataset:")
models, model_names = initialise_models()
X_ohe = min_max_scale(X_ohe)
accuracies, results = evaluate_baseline_models(models, model_names, X_ohe, y)
fig = plot_accuracies(accuracies, model_names, results)
st.pyplot(fig)

st.subheader("Running the model evaluation using the top 10 most importance features of the dataset according to RFI:")
n = st.slider("Select the number of features to use:", value=5, max_value=13)
models, model_names = initialise_models()
top = top_n_features(i, n)
X_n_features = X_woe.iloc[:, top]
X_n_features = min_max_scale(X_n_features)
accuracies, results = evaluate_baseline_models(models, model_names, X_n_features, y)
fig = plot_accuracies(accuracies, model_names, results)
st.pyplot(fig)

st.subheader("Utilising the Logistic Regression Model and obtaining an the ROC Curve")
split = 0.25
X_train, X_test, y_train, y_test = train_test_split(X_woe, y, test_size=split, train_size=1-split)
ns_probs = [0 for _ in range(len(y_test))]
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
auc = roc_auc_score(y_test, lr_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

fig = plt.figure()
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
st.pyplot(fig)
st.write("ROC Accuracy: ", auc)