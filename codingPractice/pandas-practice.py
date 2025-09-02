import numpy as np
import pandas as pd


# create emty dataframe

df = pd.DataFrame()
print(f'{df} empty dataframe')
print()

# create dataframe from list

data = [1,2,3,4,5,6,7,8,9,10]
df = pd.DataFrame(data)
print(f'{df} dataframe from list')
print()

# create dataframe using 2D array
data = np.random.randint(1, 10, size=(2, 3))
df = pd.DataFrame(data,columns=['A', 'B', 'C'],index=['row1', 'row2'])
print(f'{df} dataframe from 2D array')
print()

# create dataframe using Dictionary
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
print(f'{df} dataframe from dictionary')
print()

data = {'Name': ['John', 'Alice', 'Bob', 'Eve', 'Charlie'],
        'Age': [25, 30, 22, 35, 28],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Salary': [50000, 55000, 40000, 70000, 48000]}
df = pd.DataFrame(data)
print(f'{df} dataframe from dictionary')
print()
print(df.index,'index')  # Accessing the index
print()
print(df.columns,'columns')  # Accessing the columns
print()
print(df.values,'values')  # Accessing the values
print()
print(df.dtypes,'dtypes')  # Accessing the data types
print()
print(df.shape,'shape')  # Accessing the shape of the DataFrame
print()



data = [[1,2,3,4,5],[1,3,5,7,9],[2,4,6,8,10]]
df = pd.DataFrame(data,columns=['A', 'B', 'C', 'D', 'E'],index=['row1', 'row2', 'row3'])
print(df)

num = df.loc['row2', 'C']
print(num)
print()


num = df.loc[:'row2', 'C'].head()
print(num)
print()

num = df.iloc[:2, 3].head()
print(num)
print()

num = df.iloc[:2, 3].tail()
print(num)

num = df.iloc[:2, 3].shape
print(num)

subset = df.loc[:'row3', ['A', 'B']]
print(subset)

print()

data = {'Name': ['John', 'Alice', 'Bob', 'Eve', 'Charlie'], 
        'Age': [25, 30, 22, 35, 28], 
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'], 
        'Salary': [50000, 55000, 40000, 70000, 48000]}

df = pd.DataFrame(data)
# Display the entire DataFrame
print(df)
print()

age_column = df['Age']
print(age_column)
print()

# Access the first three rows and the 'Name' and 'Age' columns
subset = df.loc[0:2, ['Name', 'Age']]
print(subset)
print()

# Access rows where 'Age' is greater than 25
filtered_data = df[df['Age'] > 25]
print(filtered_data)
print()

filtered_data = df[(df['Age'] > 25) & ( df['Salary'] > 50000)]
print(filtered_data)
print()

# Access the 'Salary' of the row with label 2
salary_at_index_2 = df.at[2, 'Salary']
print(salary_at_index_2)
print()
# Access the 'Salary' of the row with index 2
salary_at_index_2 = df.iat[2, 3]
print(salary_at_index_2)
print()

print(df)
print()
df.set_index('Name', inplace=True)
print(df)
print()
print(df.iloc[[0,1,2]])

df['City'] =['Chicago', 'New York', 'Los Angeles', 'Chicago', 'Houston']
print(df)

print()
df['Salary'] = df['Salary'].astype(float)

print(df)
print()

print(df.loc[ (df['Salary'] > 50000) & (df['City'].str.startswith('C'))])

print()
print(df.loc[ (df['Salary'] > 50000) & (df['City'].str.startswith('C')), ['Gender', 'Salary']])

print(df)
# filter DataFrame using numpy
print()

print('filter DataFrame using numpy')
filtered_data = np.where(df['Age'] > 25)
print(filtered_data)
print()
print(df.iloc[filtered_data])
print()

# how to use dataframe query method to filter data

print('how to use dataframe query method to filter data')
filtered_data = df.query('Age > 25')
print(filtered_data)
print()
# how to use dataframe query method to filter data with multiple conditions
filtered_data = df.query('Age > 25 and Salary > 50000')
print(filtered_data)
print()

data1 = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
         'Age': [27, 24, 22, 32],
         'Address': ['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
         'Qualification': ['Msc', 'MA', 'MCA', 'Phd']}

# Define a dictionary containing employee data
data2 = {'Name': ['Abhi', 'Ayushi', 'Dhiraj', 'Hitesh'],
         'Age': [17, 14, 12, 52],
         'Address': ['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj'],
         'Qualification': ['Btech', 'B.A', 'Bcom', 'B.hons']}

print("Concatenating two dataframes use above data1 and data2")
df1 = pd.DataFrame(data1, index=[0,1,2,3])
df2 = pd.DataFrame(data2, index=[4,5,6,7])
print(df1)
print()
print(df2)
print()

# Concatenate the two DataFrames along rows
df = pd.concat([df1, df2], axis=0)
print(df)
print()

# Concatenate the two DataFrames along columns
df = pd.concat([df1, df2], axis=1)
print(df)
print()

# Concatenate the two DataFrames along rows with ignore_index=True
df = pd.concat([df1, df2], axis=1, ignore_index=True)
print(df)
print()

# Concatenate the two DataFrames along columns
df2.reset_index(drop=True, inplace=True)
df = pd.concat([df1, df2], axis=1)
print(df)
print()

# Concatenate the two DataFrames along rows with ignore_index=True
df = pd.concat([df1, df2], axis=1, ignore_index=True,join='inner')
print(df)
print()

# concat/add new row and new collumn in df1

data = {'Name': 'Ankit', 'Age': 25, 'Address': 'Delhi', 'Qualification': 'MCA'}
df3 = pd.DataFrame(data, index=[0])
print(df3)
print()
df1 = pd.concat([df1, df3], axis=0, ignore_index=True)
print(df1)

print()
# Add a new column to df1
df1['Salary'] = [50000, 60000, 70000, 80000, 90000]
print(df1)
print()
# Add a new column to df1
df1['Experience'] = [2, 3, 4, 5, 6]
print(df1)
print()

df2.loc[1, 'Name'] = 'Jai'
print(df2)

#merging and join (inner, left, right, and outer) use df1 and df2 for all operations
print()
merged_df = pd.merge(df1, df2, on='Name', how='inner')
print(merged_df)

print()
merged_df = pd.merge(df1, df2, on='Name', how='left')

print(merged_df)
print()
merged_df = pd.merge(df1, df2, on='Name', how='right')
print(merged_df)
print()
merged_df = pd.merge(df1, df2, on='Name', how='outer')
print(merged_df)
print()
# Merge with multiple keys

# practice with pandas pivote table use df1 for all aggregate operations
print()
print('practice with pandas pivote table use df1 for all aggregate operations')
print()

sum_aggregate = df1.groupby('Name').agg({'Salary': 'sum'})
print(sum_aggregate)
print()
mean_aggregate = df1.groupby('Age').agg({'Salary': 'mean'})
print(mean_aggregate)
print()
median_aggregate = df1.groupby('Age').agg({'Salary': 'median'})
print(median_aggregate)
print()
min_aggregate = df1.groupby('Name').agg({'Salary': 'min'})
print(min_aggregate)
print()

max_aggregate = df1.groupby('Name').agg({'Salary': 'max'})
print(max_aggregate)
print()
std_aggregate = df1.groupby('Name').agg({'Salary': 'std'})
print(std_aggregate)
print()
var_aggregate = df1.groupby('Name').agg({'Salary': 'var'})
print(var_aggregate)
print()
count_aggregate = df1.groupby('Name').agg({'Salary': 'count'})
print(count_aggregate)
print()
# practice with pandas pivote table use df1 for all aggregate operations
print()
print('practice with pandas pivote table use df1 for all aggregate operations')
print()
pivot_table = df1.pivot_table(index='Name', values='Salary', aggfunc='sum')
print(pivot_table)
print()
pivot_table = df1.pivot_table(index='Name', values='Salary', aggfunc='mean')
print(pivot_table)
print()
pivot_table = df1.pivot_table(index='Name', values='Salary', aggfunc='median')
print(pivot_table)
print()
pivot_table = df1.pivot_table(index='Name', values='Salary', aggfunc='min') 
print(pivot_table)
print()
pivot_table = df1.pivot_table(index='Name', values='Salary', aggfunc='max')
print(pivot_table)
print()
pivot_table = df1.pivot_table(index='Name', values='Salary', aggfunc='std')
print(pivot_table)
print()
pivot_table = df1.pivot_table(index='Name', values='Salary', aggfunc='var')
print(pivot_table)
print()
pivot_table = df1.pivot_table(index='Name', values='Salary', aggfunc='count')
print(pivot_table)
print()

# use multiple index columns
pivot_table = df1.pivot_table(index=['Name', 'Age'], values='Salary', aggfunc='sum')
print(pivot_table)
print()

# sort dataframe
print('sort dataframe')
print()
df1.sort_values(by='Salary', ascending=True, inplace=True)
print(df1)
print()
df1.sort_values(by=['Salary', 'Age'], ascending=[True, False], inplace=True)
print(df1)
print()
df1.sort_index(ascending=True, inplace=True)
print(df1)
print()
df1.sort_index(ascending=False, inplace=True)
print(df1)
print()







