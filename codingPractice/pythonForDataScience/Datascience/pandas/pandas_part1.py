import numpy as np
import pandas as pd

# Create a Series
print("Creating a Series with random integers:")
series = pd.Series(np.random.choice([1,2,3,4,5], size=10))
print(series)

print("\nCreating a Series with random integers from 1 to 10:")
arr = np.arange(1, 11)
np.random.shuffle(arr)
series = pd.Series(arr)
print(series)

arr = pd.Series(np.random.permutation(np.arange(50,70)))
print("\nCreating a Series with random integers from 50 to 70:")
print(arr)

arr = pd.Series(np.linspace(1,5,10))
print("\nCreating a Series with 10 evenly spaced numbers between 1 and 5:")
print(arr)

function = lambda x,y: y**2 + 4*x + 1

arr = pd.Series(np.full((2, 5), 10).flatten())
print("\nCreating a Series with a constant value of 10:")
print(arr)



# Create a DataFrame ==============================================
print("\nCreating a DataFrame with random integers:")
df = pd.DataFrame(np.fromfunction(function,(5, 5),dtype=int),columns=['A', 'B', 'C', 'D', 'E'])
print(df)
print(df.shape)
print(df.columns)
print(df.index)
print(df.dtypes)
print(df.head())
print(df.tail())
print(df.sample(3))  # Randomly sample 3 rows
print(df.describe())  # Summary statistics
print(df.info())  # DataFrame information


print( "data frame from dictionary:")

data = {
    'randint': np.random.randint(1,10,5),
    'randn': np.random.randn(5),
    'rand': np.random.rand(5),
    'choice': np.random.choice(np.arange(1,10), size=5),
    'linspace': np.linspace(1,6, 5)
}
print(data)
print()
df = pd.DataFrame(data)
print(df)

#================================================================
print("read_csv, read_excel, read_json, read_html, read_sql, read_sql_table, read_sql_query")
# Note: The following functions require specific files or databases to work.

# df = pd.read_csv('file.csv')  # Read a CSV file
# df = pd.read_excel('file.xlsx')  # Read an Excel file
# df = pd.read_json('file.json')  # Read a JSON file

#===============================================================
print(" Inspecting DataFrame")
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns)
print("DataFrame index:", df.index)
print("DataFrame data types:", df.dtypes)
print("DataFrame head:\n", df.head(2))
print("DataFrame tail:\n", df.tail(3))
print("DataFrame sample:\n", df.sample(3))  # Randomly sample 3 rows
print("DataFrame describe:\n", df.describe())  # Summary statistics
print("DataFrame info:\n", df.info())  # DataFrame information
print("DataFrame memory usage:\n", df.memory_usage(deep=True))  # Memory usage of DataFrame
print("DataFrame columns with NaN values:\n", df.isna().any())  # Check for NaN values in each column
print("DataFrame columns with NaN values count:\n", df.isna().sum())  # Count of NaN values in each column

#===============================================================

print("4. Selecting & Filtering Data")
print("df['col'], df[['col1', 'col2']] df.iloc[0], df.iloc[:, 0], df.iloc[1:5, 0:2] df.loc[1], df.loc[:, 'col'], df.loc[2:5, ['col1', 'col2']] ,df.query('col > 5') ,df[df['col'] > 10]")

# create some real word case dataframe dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [24, 27, 22, 32, 29],
    'Salary': [50000, 60000, 55000, 70000, 65000],
    'Department': ['HR', 'Finance', 'IT', 'Marketing', 'Sales']
}
df = pd.DataFrame(data)
print("\nDataFrame for selection and filtering:")
print(df)
print("\nSelecting a single column 'Name':")
print(df['Name'])
print("\nSelecting multiple columns 'Name' and 'Age':")
print(df[['Name','Age']])

print("\nupdate the any column value like df['col']:")
df['Salary'] = df['Salary'] * 0.1  # decrease salary by 10%

print("\nUpdated DataFrame after modifying 'Salary':")
print(df)

print("\nupdate the any column value like df.loc & df iloc finctions:")
df.loc[2, 'Salary'] = df.loc[2, 'Salary'] + df.loc[2, 'Salary'] * 0.2 # increase Charlie's salary by 20%
print("\nUpdated DataFrame after modifying 'Salary' using loc:")
print(df)

print("\nSelecting a row by index using loc:")
print(df.loc[1])  # Select the row with index 1 (Bob)
print("========")
print(df.iloc[2:3,2:3]) # Select rows 2 to 3 and columns 2 to 3 (Charlie and David's Salary)
print("========")
print(df.loc[:3, ['Name', 'Salary']])  # Select rows up to index 3 and columns 'Name' and 'Salary'
print("========")
print(df.loc[[1,3], ['Name', 'Department']])  # Select rows 1 and 3 and columns 'Name' and 'Department'
print("========")
print(df.loc[(df['Age']> 25) & (df['Salary'] > 60000)]) # Select rows where Age > 25 and Salary > 60000
print("========")
print(df.query('Age > 25 & Salary >= 60000'))  # Using query method to filter rows
print("\nFiltering rows where 'Age' is greater than 25:")
print(df[df['Age'] > 25])  # Filter rows where Age is greater than 25


print("\nSelecting and updating a row by index using iloc:")
print(df.iloc[0])  # Select the first row (Alice)
print("\nSelecting a specific cell using iloc:")
print(df.iloc[0, 1])  # Select the cell at row 0, column 1 (Alice's Age)
print("\nSelecting a specific cell using loc:")
print(df.iloc) # Select the cell at row 0, column 1 (Alice's Age)



"""
df['new'] = df['col1'] + df['col2']
df['col'] = df['col'].astype('int')
df.rename(columns={'old': 'new'})
df.drop(columns=['col1']), df.drop(index=3)
df.insert(loc=1, column='new', value=val)
df.replace({'val1': 'val2'}), df.map(), df.apply()
df.sort_values('col')
df.sort_index()
df.rank()
"""
print("\nAdding a new column 'NewCol' as the sum of 'Age' and 'Salary':")
df['newCol'] = df['Age']*100 + df['Salary']
print(df)

print("\nChanging the data type of 'newCol' column to float:")
df['newCol'] = df['newCol'].astype(float)
print(df)

print("\nRenaming the 'newCol' column to 'salary_2025':")
df.rename(columns={'newCol': 'salary_2025'},inplace=True)
print(df)

print("\n create new gender column if salary_2025 is greater than 8800 its male otherwise female")

df['Gender'] = np.where(df['salary_2025'] > 8800, 'Male', 'Female')
print(df)

df['Temp'] = df['salary_2025'].apply(lambda x: 'High' if x > 8800 else 'Low')
print(df)

print("\nDropping the 'Temp' column:")
df.drop(columns=['Temp'], inplace=True)
print(df)

print("\nInserting a new column 'Bonus' at index 2 with a constant value of 1000:")
df.insert(loc=2, column='Bonus', value=1000)
print(df)

print("\nReplacing 'HR' with 'Human Resources' in the 'Department' column:")
df.replace({'DEpartment': {'HR' : 'Human Resources'}}, inplace=True)
print(df)

print("\nSorting the DataFrame by 'Salary' in ascending order:")
df.sort_values(by = 'Salary', ascending=True, inplace=True)
print(df)
print("\nSorting the DataFrame by index:")
df.sort_index(inplace=True)
print(df)
print("\nRanking the 'Salary' column:")
df['Salary_Rank'] = df['Salary'].rank()
print(df)
print("\nDataFrame after ranking 'Salary':")
print(df)

print("adding new multiple rows with dummy data having some N/A values enclude new created column also:")
new_data = {
    'Name': ['Frank', 'Grace'],
    'Age': [28, np.nan],
    'Bonus': [np.nan, 1500],
    'Salary': [np.nan, 72000],
    'Department': ['IT', 'Sales'],
    'salary_2025': [np.nan, 9000],
    'Gender': ['np.nan',np.nan],
    'Salary_Rank': [np.nan, np.nan]
}
new_df = pd.DataFrame(new_data)
print("\nNew DataFrame to be appended:")
print(new_df)
print("\nAppending new DataFrame to the existing DataFrame:")
df = pd.concat([df, new_df], ignore_index=True, sort=False)
print(df)

"""
df.isnull(), df.notnull()
df.fillna(method='ffill'), df.fillna(method='bfill'), df.fillna(0),
df.dropna(), df.dropna(axis=1)
df.interpolate()

"""

print("\nChecking for null values in the DataFrame:")
print(df.isnull())
print("\nChecking for non-null values in the DataFrame:")
print(df.notnull())
print("\nFilling null values with forward fill method:")
df.ffill(inplace=True)
print(df)
print("\nFilling null values with backward fill method:")
df.bfill(inplace=True)
print(df)
print("\ndrop na by axis one:")
df.dropna(axis=1, inplace=True)
print("\nDataFrame after dropping columns with NaN values:")
print(df)

df.dropna(axis=0, inplace=True)  # Drop rows with NaN values
print("\nDataFrame after dropping rows with NaN values:")
print(df)
# print("\nInterpolating missing values:")
# df = df.infer_objects(copy=False) # Infer types of columns 
# df.interpolate(inplace=True)  # Interpolate missing values
# print(df)
print("\nFinal DataFrame after all operations:")
print(df)
print("\nDataFrame memory usage after all operations:")
print(df.memory_usage(deep=True))  # Memory usage of DataFrame after all operations
print("\nDataFrame columns with NaN values after all operations:")
print(df.isna().any())  # Check for NaN values in each column after all operations
print("\nDataFrame columns with NaN values count after all operations:")
print(df.isna().sum())  # Count of NaN values in each column after all operations
# print("\nDataFrame info after all operations:")
# print(df.info())  # DataFrame information after all operations
# print("\nDataFrame shape after all operations:", df.shape)  # DataFrame shape after all operations
# print("\nDataFrame columns after all operations:", df.columns)  # DataFrame columns after all operations
# print("\nDataFrame index after all operations:", df.index)  # DataFrame index after all operations
# print("\nDataFrame data types after all operations:", df.dtypes)  # DataFrame data types after all operations
# print("\nDataFrame head after all operations:")
# print(df.head(2))  # DataFrame head after all operations
# print("\nDataFrame tail after all operations:")
# print(df.tail(3))  # DataFrame tail after all operations
# print("\nDataFrame sample after all operations:")
# print(df.sample(3))  # Randomly sample 3 rows after all operations
# print("\nDataFrame describe after all operations:")
# print(df.describe())  # Summary statistics after all operations


print("\nDataFrame after all operations:")
print(df)  # DataFrame information after all operations


"""
8. Grouping & Aggregation
df.groupby('col').mean()
df.groupby(['col1', 'col2']).agg(['sum', 'mean'])
df.pivot_table(values='val', index='A', columns='B', aggfunc='sum')
df.groupby('col').agg({'col1': 'sum', 'col2': 'mean'})
"""


"""
9. Merging, Joining, and Concatenation
pd.concat([df1, df2], axis=0)
pd.merge(df1, df2, on='key', how='inner')
df1.join(df2, how='left')

"""

"""
11. Text and String Operations
df['col'].str.lower(), str.upper(), str.strip()
df['col'].str.contains('word'), str.replace(), str.extract()
df['col'].str.split('-')

"""


