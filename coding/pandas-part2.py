import numpy as np
import pandas as pd


# practice pandas Series all methods and concepts
print()
# Creating a Series
data = [1, 2, 3, 4, 5]
s = pd.Series(data)
print("Series:")
print(s)
print()
# Accessing elements
print("Accessing elements:")
print(s[0])  # First element
print(s[1:4])  # Slicing    
print()
# Modifying elements
s[0] = 10
print("Modified Series:")
print(s)
print()
# Adding elements
s[5] = 6
print("Series after adding an element:")
print(s)
print()
# Removing elements
s = s.drop(0)
print("Series after removing an element:")
print(s)
print()

# use numpy array to create a series
data = np.array([1, 2, 3, 4, 5])
s = pd.Series(data)
print("Series from numpy array:")
print(s)
print()

# Creating a Series with custom index
data = [1, 2, 3, 4, 5]
index = ['a', 'b', 'c', 'd', 'e']
s = pd.Series(data, index=index)
print("Series with custom index:")
print(s)
print()
# Creating a Series with a dictionary
data = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
s = pd.Series(data)
print("Series from dictionary:")
print(s)
print()
# Creating a Series with a date range index
date_range = pd.date_range(start='2023-01-01', periods=5)
data = [1, 2, 3, 4, 5]
s = pd.Series(data, index=date_range)
print("Series with date range index:")
print(s)
print()

# Creating a Series with NaN values
data = [1, 2, np.nan, 4, 5]
s = pd.Series(data)
print("Series with NaN values:")
print(s)
print()

# Creating a Series with a specific data type
data = [1, 2, 3, 4, 5]
s = pd.Series(data, dtype='float32')
print("Series with specific data type:")
print(s)
print()

# Creating a Series with a specific data type and custom index
data = [1, 2, 3, 4, 5]
index = ['a', 'b', 'c', 'd', 'e']
s = pd.Series(data, index=index, dtype='float32')
print("Series with specific data type and custom index:")
print(s)
print()

# Creating a Series with a specific data type and date range index
date_range = pd.date_range(start='2023-01-01', periods=5)
data = [1, 2, 3, 4, 5]
s = pd.Series(data, index=date_range, dtype='float32')
print("Series with specific data type and date range index:")
print(s)
print()

# Accessing multiple elements using index labels
data = [1, 2, 3, 4, 5]
index = ['a', 'b', 'c', 'd', 'e']
s = pd.Series(data, index=index)
print("Accessing multiple elements using index labels:")   
print(s[['a', 'c', 'e']])
print()

# Accessing multiple elements from last or from start
data = [1, 2, 3, 4, 5,6,7,8,9,10]
index = ['a', 'b', 'c', 'd', 'e','f','g','h','i','j']
s = pd.Series(data, index=index)
print("Accessing multiple elements from last or from start:")
print(s[-3:])  # Last 3 elements
print(s[:3])  # First 3 elements
print()
# Accessing multiple elements using boolean indexing
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
s = pd.Series(data, index=index)
print("Accessing multiple elements using boolean indexing:")
print(s[s > 5])  # Elements greater than 5
print(s[s < 5])  # Elements less than 5
print(s[s == 5])  # Elements equal to 5
print(s[s != 5])  # Elements not equal to 5
print()
# Accessing multiple elements using boolean indexing with custom condition
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
s = pd.Series(data, index=index)
print("Accessing multiple elements using boolean indexing with custom condition:")
print(s[(s > 5) & (s < 8)])  # Elements greater than 5 and less than 8
print(s[(s < 5) | (s > 8)])  # Elements less than 5 or greater than 8

# logical operations on two pandas series
data1 = [1, 2, 3, 4, 5]
data2 = [6, 7, 8, 9, 10]
s1 = pd.Series(data1)
s2 = pd.Series(data2)
print("Logical operations on two pandas series:")
print(s1 > s2)  # Greater than
print(s1 < s2)  # Less than
print(s1 == s2)  # Equal to
print(s1 != s2)  # Not equal to
print(s1 >= s2)  # Greater than or equal to
print(s1 <= s2)  # Less than or equal to
print()
# Mathematical operations on pandas series
data1 = [1, 2, 3, 4, 5]
data2 = [6, 7, 8, 9, 10]
s1 = pd.Series(data1)
s2 = pd.Series(data2)
print("Mathematical operations on pandas series:")
print(s1 + s2)  # Addition
print(s1 - s2)  # Subtraction
print(s1 * s2)  # Multiplication
print(s1 / s2)  # Division
print(s1 % s2)  # Modulus
print(s1 ** s2)  # Exponentiation
print()
# unary operations on two pandas series
data1 = [1, 2, 3, 4, 5]
data2 = [6, 7, 8, 9, 10]
s1 = pd.Series(data1)
s2 = pd.Series(data2)
print("Unary operations on two pandas series:")
print(s1 + 1)  # Addition
print(s1 - 1)  # Subtraction
print(s1 * 2)  # Multiplication
print(s1 / 2)  # Division
print(s1 % 2)  # Modulus
print(s1 ** 2)  # Exponentiation
print()
# Mathematical operations on pandas series with custom function
def custom_function(x):
    return x ** 2 + 2 * x + 1
data = [1, 2, 3, 4, 5]
s = pd.Series(data)
print("Mathematical operations on pandas series with custom function:")
print(s.apply(custom_function))  # Applying custom function
print()
# Mathematical operations on pandas series with lambda function
data = [1, 2, 3, 4, 5]
s = pd.Series(data)
print("Mathematical operations on pandas series with lambda function:")
print(s.apply(lambda x: x ** 2 + 2 * x + 1))  # Applying lambda function
print()

# binary operations on two pandas series
data1 = [1, 2, 3, 4, 5]
data2 = [6, 7, 8, 9, 10]
s1 = pd.Series(data1)
s2 = pd.Series(data2)
print("Binary operations on two pandas series:")
print(s1.add(s2))  # Addition
print(s1.sub(s2))  # Subtraction
print(s1.mul(s2))  # Multiplication
print(s1.div(s2))  # Division
print(s1.mod(s2))  # Modulus
print(s1.pow(s2))  # Exponentiation
print()

# & , | , ~ operations on two boolean pandas series\
data1 = [True, False, True, False, True]
data2 = [False, True, False, True, False]
s1 = pd.Series(data1)
s2 = pd.Series(data2)
print("Logical operations on two boolean pandas series:")
print(s1 & s2)  # AND
print(s1 | s2)  # OR
print(~s1)  # NOT
print()


