""" Customisable function script that searches for a particular constraints
of the borrowers in the database """

import sqlite3
import pandas as pd

credit_data = pd.read_csv("Credit Data.csv")


def tuple_convert(data):

    data_tuples = [list(data.iloc[i]) for i in range(len(data))]

    for i in range(len(data_tuples)):
        for j in range(len(data_tuples[i])):
            if type(data_tuples[i][j]) != str:
                data_tuples[i][j] = int(data_tuples[i][j])

    return data_tuples


db_data = tuple_convert(credit_data)
for i in range(len(db_data)):
    db_data[i] = tuple(db_data[i])


conn = sqlite3.connect('credit.db')
c = conn.cursor()

c.executemany("INSERT INTO borrowers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", db_data)

# Commit our command
conn.commit()

# Close our connections
conn.close()


