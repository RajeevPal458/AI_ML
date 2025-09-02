import numpy as np
import pandas as pd
import sqlite3


# Create a connection to the SQLite database
connection = sqlite3.connect('practice.db')
# Create a cursor object to execute SQL commands
cursor = connection.cursor()

cursor.execute('''DROP TABLE IF EXISTS employee''')

# Create list of  Employee data json sample DataFrame
data = {
    'name': ['John', 'Jane', 'Doe', 'Alice', 'Bob'],
    'age': [28, 34, 29, 42, 25],
    'salary': [70000, 80000, 60000, 90000, 50000]
}

df = pd.DataFrame(data)
# Create a table in the SQLite database
print(df)

print('=====================================================')
print()
sql = ''' CREATE TABLE IF NOT EXISTS employee (
    name TEXT,
    age INTEGER,
    salary REAL
) '''

# Execute the SQL command to create the table
cursor.execute(sql)
# Insert the DataFrame into the SQLite database
cursor.executemany('INSERT INTO employee (name, age, salary) VALUES (?, ?, ?)', df.values.tolist())

connection.commit()
# Read the data from the SQLite database into a DataFrame
cursor.execute('SELECT * FROM Employee')
# Fetch all rows from the result of the query
result = cursor.fetchall()

# Convert the result to a DataFrame
df_result = pd.DataFrame(result, columns=['name', 'age', 'salary'])
# Print the DataFrame
print('=====================================================')
print('Data from SQLite database:')
print(df_result)

# prameterized sql query with where clause
sql = ''' SELECT * FROM employee WHERE age < ? AND salary < ? '''
# Execute the SQL command with parameters
cursor.execute(sql, (30, 80000))
# Fetch all rows from the result of the query
result = cursor.fetchall()
# Convert the result to a DataFrame
df_result = pd.DataFrame(result, columns=['name', 'age', 'salary'])
# Print the DataFrame
print('=====================================================')
print('Data from SQLite database with parameterized query:')
print(df_result)

# update the data in the SQLite database
sql = ''' UPDATE employee SET salary = ? WHERE name = ? '''
# Execute the SQL command with parameters
cursor.execute(sql, (75000, 'John'))
# Commit the changes to the database
connection.commit()
# Read the updated data from the SQLite database into a DataFrame
cursor.execute('SELECT * FROM employee')
# Fetch all rows from the result of the query
result = cursor.fetchall()
# Convert the result to a DataFrame
df_result = pd.DataFrame(result, columns=['name', 'age', 'salary'])
# Print the DataFrame
print('=====================================================')
print('Updated data from SQLite database:')
print(df_result)

# insert new data into the SQLite database
sql = ''' INSERT INTO employee (name, age, salary) VALUES (?, ?, ?) '''
# Execute the SQL command with parameters
cursor.execute(sql, ('Charlie', 30, 70000))
# Commit the changes to the database
connection.commit()
# Read the updated data from the SQLite database into a DataFrame   
cursor.execute('SELECT * FROM employee')
# Fetch all rows from the result of the query
result = cursor.fetchmany(6)
# Convert the result to a DataFrame
df_result = pd.DataFrame(result, columns=['name', 'age', 'salary'])
# Print the DataFrame
print('=====================================================')
print('Inserted data from SQLite database:')
print(df_result)
# Delete data from the SQLite database
sql = ''' DELETE FROM employee WHERE name = ? '''
# Execute the SQL command with parameters
cursor.execute(sql, ('Charlie1',))
# Commit the changes to the database
connection.commit()

# Fetch rows as dictionaries:
cursor.execute('SELECT * FROM employee')
rows = cursor.fetchall()
# Convert the result to a DataFrame
df_result = pd.DataFrame(rows, columns=['name', 'age', 'salary'])
# Print the DataFrame
print('=====================================================')
print('Data from SQLite database with fetchall:')
print(df_result)

# sqlite3.Row factory
connection.row_factory = sqlite3.Row
# Create a cursor object to execute SQL commands
cursor = connection.cursor()
# Execute a query
cursor.execute('SELECT * FROM employee')
# Fetch all rows from the result of the query
rows = cursor.fetchall()
# Convert the result to a DataFrame
df_result = pd.DataFrame([dict(row) for row in rows])
# Print the DataFrame
print('=====================================================')
print('Data from SQLite database with sqlite3.Row factory:')
print(df_result)


# Close the cursor and connection
cursor.close()
connection.close()

#======================================================================================


# Using with Statement for Safe Resource Handling:
# The with statement ensures that the connection is properly closed after use, even if an error occurs.
with sqlite3.connect('practice.db') as connection:
    # Create a cursor object to execute SQL commands
    cursor = connection.cursor()
    connection.row_factory = sqlite3.Row
    # Execute a query
    cursor.execute('SELECT * FROM employee')
    # Fetch all rows from the result of the query
    rows = cursor.fetchall()
    # count rows
    print('=====================================================')
    print('Number of rows:', len(rows))
  
    # Convert the result to a DataFrame
    df_result = pd.DataFrame(rows, columns=['name', 'age', 'salary'])
    # Print the DataFrame
    print('=====================================================')
    print('Data from SQLite database with with statement:')
    print(df_result)






