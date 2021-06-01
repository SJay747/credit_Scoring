import sqlite3

# Connect to database
conn = sqlite3.connect('credit.db')

c = conn.cursor()

# Creating a table
c.execute("""CREATE TABLE borrowers  (
    yob INTEGER,
    num_children INTEGER,
    num_dependents INTEGER,
    home_phone INTEGER,
    spousal_income INTEGER, 
    employment_status TEXT,
    income INTEGER,
    residential_status TEXT,
    value_of_home INTEGER,
    mortgage_outstanding INTEGER,
    mortgage_outgoings INTEGER,
    loan_outgoings INTEGER,
    hire_purchase_outgoings INTEGER,
    credit_card_outgoings INTEGER, 
    good_bad INTEGER   
)""")

conn.commit()

conn.close()