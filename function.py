### Data Wrangling, Preparation, and Feature Engineering Functions
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def load_data(file_name):
    data = pd.read_csv(file_name)
    data = data[data["YOB"] != 99]  # remove YOB 99 values

    return data

def import_SQL(columns):
    conn = sqlite3.connect('credit.db')
    c = conn.cursor()

    c.execute("SELECT rowid, * FROM borrowers")

    no_of_records = 3
    items = c.fetchall()

    conn.commit()

    conn.close()

    a_list = [a_tuple[1:] for a_tuple in items]

    credit_data = pd.DataFrame(a_list, columns=columns)

    # Removing rows with no YOB data
    credit_data = credit_data[credit_data["YOB"] != 99]

    return credit_data

def WoE(data, catdata, col_name):
    WoE = []
    IV = []

    all_good = len(data[data["BAD"] == 0])
    all_bad = len(data[data["BAD"] == 1])

    for i in range(len(catdata)):
        bad = len(data[(data[col_name] == catdata[i]) & (data["BAD"] == 1)])
        good = len(data[(data[col_name] == catdata[i]) & (data["BAD"] == 0)])

        if bad != 0:
            good_perc = good / (good + bad)
            bad_perc = bad / (good + bad)
            all_good_perc = all_good / (all_good + all_bad)
            all_bad_perc = all_bad / (all_good + all_bad)

            WoE.append(np.log(good_perc / bad_perc))
            IV.append((all_good_perc - all_bad_perc) * WoE[i])
        else:
            WoE.append(0)

    return WoE, IV

def woe_modify(data, aes, res, woe_aes, woe_res, keys_aes, keys_res, file_name):

    # load data
    data_mod = load_data(file_name)

    for i in range(len(aes)):
        data_mod["AES"] = data_mod["AES"].replace(to_replace=aes[i], value=woe_aes[keys_aes[i]])

        if i < len(res):
            data_mod["RES"] = data_mod["RES"].replace(to_replace=res[i], value=woe_res[keys_res[i]])

    return data_mod

def get_cat_data(df):
    # returns dataframe containing all categorical data in dataframe
    cat_ix = df.select_dtypes(include=['object', 'bool']).columns
    return df[cat_ix]

def encode_cat_data(df):
    # takes a dataframe of categorical data and outputs a dataframe containing the categorical data one hot coded
    encoder = OneHotEncoder(sparse=False)
    values = encoder.fit_transform(df)
    return values

def get_num_data(df):
    # returns dataframe containing numerical data
    num_ix = df.select_dtypes(include=['int64', 'float64']).columns
    return df[num_ix]


def one_hot_encode_modify(file_name):
    # load data
    data = import_SQL(file_name)

    # output
    y = data.iloc[:, -1]

    # retrieve categorical data
    cat_data = get_cat_data(data)

    # one hot encode
    onehotvals = encode_cat_data(cat_data)

    # get numerical data
    num_data = get_num_data(data)

    # drop final column (output) to get feature matrix
    data_mod = num_data.drop(data.columns[-1], axis=1)

    # join one hot encoded categorical values to feature matrix
    data_mod = data_mod.join(pd.DataFrame(onehotvals, index=data.index))  # only contains numerical features

    return data_mod, y

def min_max_scale(df):
    # takes numerical feature matrix as input and scales using MinMaxScaler
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    return df

