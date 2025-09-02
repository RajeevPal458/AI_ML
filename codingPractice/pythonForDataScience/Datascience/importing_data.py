# how many ways we can import different types of data (exel, csv, database, API Call) in python 
# # In Python, you can import different types of data using various libraries and methods. Here are some common ways to import data from different sources:
##

# what is nymerican and ordinal data 
# what is the difference between them
# 
# what is the difference between categorical and continuous data
# what is the difference between numerical and ordinal data 
# what is the difference between categorical and numerical data
# what is the difference between categorical and continuous data
# what is the difference between categorical and quantitative data
# what is the difference between categorical and discrete data
# what is the difference between categorical and qualitative data


#
# 1. **CSV Files**: You can use the `pandas` library to read CSV files easily. The `read_csv()` function allows you to import data from a CSV file into a DataFrame.
# ```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import xlrd
import html5lib
import requests
import json
import os
import openpyxl
import pyodbc
#import mysql.connector
import sqlalchemy
import csv
import xml.etree.ElementTree as ET
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
from scipy.io import loadmat


# give code example for each type of data import
# 1. CSV Files
# ```python
df_csv = pd.read_csv('../files/csv1_sample.csv')
print(df_csv.head())
# ```
#
# 2. Excel Files
# ```python
df_excel = pd.read_excel('../files/excel1_sample.xls', sheet_name='Sheet1')
print(df_excel.head())
# ```
#
# 3. JSON Files
# ```python
df_json = pd.read_json('../files/json_sample.json')
print(df_json.head())

file = open('../files/txt_sample.txt', 'r')
data = file.read()  
print(data)
file.close()
print(file.closed)
# ```
# with context manager
# ```python
with open('../files/txt_sample.txt', 'r') as file:
    data = file.read()
    print(data)
# ```   



# 4. Text Files
# ```python
df_txt = pd.read_csv('../files/csv2_sample.csv', sep='\t')
print(df_txt.head())
# ```

# .M Matlab Files
# ```python
df_mat = loadmat('../files/matlab_sample.mat')
print(df_mat)
# ```

# 4. SQL Databases
# ```python
conn = sqlite3.connect('data.db')
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS Employees (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')
cur.execute('''INSERT INTO Employees (name, age) VALUES ('John Doe', 30)''')
cur.execute('''INSERT INTO Employees (name, age) VALUES ('raj Doe', 35)''')
conn.commit()

cur.execute('select * from Employees')
rows = cur.fetchall()
for row in rows:
    print(row)
df_sql = pd.read_sql_query('SELECT * FROM Employees', conn)
conn.close()
# print(df_sql.head())
# ```
#
# 5. API Calls
# ```python
try:
    response = requests.get('https://fake-json-api.mock.beeceptor.com/users')
    print(f"fake users data := {response}")
    data = response.json()
    df_api = pd.DataFrame(data)
    print(df_api.head())
except Exception as e:
    print(f"Error fetching API data: {e}")

# ```
#
# 6. HTML Tables
# ```python
#try:
    #df_html = pd.read_html('https://help.hcl-software.com/onedb/2.0.1/rest/rest_010.html')[0]
    #print(df_html)
#except Exception as e:
 #   print(f"Error reading HTML: {e}")

# ```

html_data = requests.get('https://fake-json-api.mock.beeceptor.com/users')
json_data = json.loads(html_data.content)
print(json_data)
df_html = pd.DataFrame(json_data)
print(df_html.head())





# 7. HDF5 Files
# ```python
# df_hdf5 = pd.read_hdf('data.h5', key='data')
# print(df_hdf5.head())
# ```
#
# 8. Parquet Files
# ```python
# df_parquet = pd.read_parquet('data.parquet')
# print(df_parquet.head())
# ```
#
# 9. Feather Files
# ```python
# df_feather = pd.read_feather('data.feather')
# print(df_feather.head())
# ```
#
# 10. Stata Files
# ```python
# df_stata = pd.read_stata('data.dta')
# print(df_stata.head())
# ```
#
# 11. SPSS Files
# ```python
# df_spss = pd.read_spss('data.sav')
# print(df_spss.head())
# ```
#
# 12. SAS Files
# ```python
# df_sas = pd.read_sas('data.sas7bdat')
# print(df_sas.head())
# ```
#
##########################################################################