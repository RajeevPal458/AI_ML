import numpy as np


#arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr = np.array([1,2,3,4,5,6,7,8,9])
print(arr.shape)  # Print the shape of the array
print(arr.dtype)
print(arr.ndim)  # Print the number of dimensions of the array
print(arr.size)  # Print the total number of elements in the array
print(arr.itemsize)  # Print the size of each element in bytes
print(arr.nbytes)  # Print the total number of bytes consumed by the array
print(arr)  # Print the array

arr = arr.reshape(3, 3)  # Reshape to a 3x3 array
print(arr.shape)
print()
print(arr)  # Print the reshaped array
# rand - Generates random numbers from a uniform distribution over the interval [0,1).
data = np.random.rand(10)
print(data)  # Print the random data

print("=================rand")
data = np.random.rand(2, 2) 
print(data)  # Print the random 2x2 array
print("=================rand3d")

data = np.random.rand(2, 2, 2)
print(data)  # Print the random 2x2x2 array

print("=================randn")
data = np.random.randn(4,4)
print(data)  # Print the random 2x2 array with normal distribution

print("================= c")
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
c = a+b
print(c)  # Print the result of element-wise addition
print("=================a+10")
print(a + 10)  # Print the result of adding 10 to each element of array a
print("=================a*2")
print(a * 2)  # Print the result of multiplying each element of array a by 2
print("=================a*b")
print(a * b)  # Print the result of element-wise multiplication

print("=================array using tuple")
arr = np.array((1,2,3,4,5,6,7,8,9))
print(arr)  # Print the original array

print("=================array using given built-in function")

arr = np.zeros((3, 3))  # Create a 3x3 array filled with zeros
print(arr)  # Print the array filled with zeros

print("====================================")

arr = np.ones(10)
print(arr)  # Print the array filled with ones

arr = np.ones((3, 3))  # Create a 3x3 array filled with ones
print(arr)  # Print the array filled with ones

print("====================================")
arr = np.full(10,2) # Create an array of size 10 filled with the value 2
print(arr)  # Print the array filled with the value 2
arr = np.full((3,4),5) # Create a 3x4 array filled with the value 5
print(arr)  # Print the 3x4 array filled with the value 5
print("====================================")
arr = np.eye(3) # Create a 3x3 identity matrix
print(arr)  # Print the identity matrix
arr = np.eye(3,4) # Create a 3x4 identity matrix
print(arr)  # Print the 3x4 identity matrix
arr = np.eye(3, k=1) # Create a 3x3 identity matrix with diagonal offset by 1
print(arr)  # Print the 3x3 identity matrix with diagonal offset by 1
arr = np.eye(3, k=-1) # Create a 3x3 identity matrix with diagonal offset by -1
print(arr)  # Print the 3x3 identity matrix with diagonal offset by -1

arr = np.eye(4,4) # Create a 4x4 identity matrix
print(arr)  # Print the 4x4 identity matrix

print("====================================")
arr = np.arange(10) # Create an array with values from 0 to 9
print(arr)  # Print the array with values from 0 to 9
arr = np.arange(1,20) # Create an array with values from 1 to 19
print(arr)  # Print the array with values from 1 to 19
arr = np.arange(1,20,2) # Create an array with values from 1 to 19 with a step of 2
print(arr)  # Print the array with values from 1 to 19 with a step of 2
arr = np.arange(1,20,3) # Create an array with values from 1 to 19 with a step of 3
print(arr)  # Print the array with values from 1 to 19 with a step of 3

print("====================================")
arr = np.linspace(20,50, 10) # Create an array with 10 values evenly spaced between 20 and 50
print(arr)  # Print the array with 10 values evenly spaced between 20 and 50
arr = np.linspace(20,50, 10, endpoint=False) # Create an array with 10 values evenly spaced between 20 and 50, excluding the endpoint
print(arr)  # Print the array with 10 values evenly spaced between 20 and 50, excluding the endpoint
arr = np.linspace(20,50, 10, retstep=True) # Create an array with 10 values evenly spaced between 20 and 50, returning the step size
print(arr)  # Print the array with 10 values evenly spaced between 20 and 50, returning the step size
arr = np.linspace(20,50, 10, retstep=True, endpoint=False) # Create an array with 10 values evenly spaced between 20 and 50, excluding the endpoint and returning the step size
print(arr)  # Print the array with 10 values evenly spaced between 20 and 50, excluding the endpoint and returning the step size

print("====================================")
arr = np.random.rand(10) # Create an array with 10 random values from a uniform distribution over the interval [0,1)
print(arr)  # Print the array with 10 random values from a uniform distribution over the interval [0,1)
arr = np.random.rand(3,3) # Create a 3x3 array with random values from a uniform distribution over the interval [0,1)
print(arr)  # Print the 3x3 array with random values from a uniform distribution over the interval [0,1)
arr = np.random.rand(3,3,3) # Create a 3x3x3 array with random values from a uniform distribution over the interval [0,1)
print(arr)  # Print the 3x3x3 array with random values from a uniform distribution over the interval [0,1)

print("====================================")
arr = np.random.randn(10) # Create an array with 10 random values from a mean 0 and standard deviation 1 normal distribution
print(arr)  # Print the array with 10 random values from a normal distribution
arr = np.random.randn(3,3) # Create a 3x3 array with random values from a normal distribution
print(arr)  # Print the 3x3 array with random values from a normal distribution

print("====================================")
arr = np.random.randint(1,10, size=(3, 3))  # Create a 3x3 array with random integers between 1 and 10
print(arr)  # Print the 3x3 array with random integers between 1 and 10
arr = np.random.randint(1,10, (2, 3))  # Create a 3x3x3 array with random integers between 1 and 10
print(arr)  # Print the 3x3x3 array with random integers between 1 and 10

print("====================================")
arr = np.random.choice([1, 2, 3, 4, 5], size=(3, 3))  # Create a 3x3 array with random choices from the list    
print(arr)  # Print the 3x3 array with random choices from the list
arr = np.random.choice([1,2,3,4,5,6,7,8,9,10],(3,3),replace=False) # Create a 3x3 array with random choices from the list
print(arr)  # Print the 3x3 array with random choices from the list

print("====================================")
arr = np.random.permutation([1, 2, 3, 4, 5])  # Create a random permutation of the list
print(arr)  # Print the random permutation of the list

arr = np.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Create a random permutation of the list
print(arr)  # Print the random permutation of the list

print("====================================")
arr = np.random.shuffle([1, 2, 3, 4, 5])  # Shuffle the list in place
print(arr)  # Print the shuffled list (Note: This will print None because shuffle modifies the list in place and returns None)
arr = np.random.shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Shuffle the list in place
print(arr)  # Print the shuffled list (Note: This will print None because shuffle modifies the list in place and returns None)
list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
np.random.shuffle(list)  # Shuffle the list in place
print(list)  # Print the shuffled list (This will print the shuffled list)


print("====================================")
arr = np.random.seed(42)  # Set the random seed for reproducibility
print(arr)  # Print the random seed (This will print None because seed does not return anything)

print("====================================diagonal")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr = np.diagonal(arr)  # Get the diagonal elements of the array
print(arr)  # Print the diagonal elements of the array

print("====================================transpose")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr = np.transpose(arr)  # Transpose the array
print(arr)  # Print the transposed array
print("====================================flatten")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr = arr.flatten()  # Flatten the array to a 1D array
print(arr)  # Print the flattened array

print(" zeros_like")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr_zeros = np.zeros_like(arr)  # Create an array of zeros with the same shape as arr
print(arr_zeros)  # Print the array of zeros with the same shape as arr

print(" ones_like")
arr_ones = np.ones_like(arr)  # Create an array of ones with the same shape as arr
print(arr_ones)  # Print the array of ones with the same shape as arr



# ========================================================================================
print("Array Indexing, Slicing, Reshaping, and Iterating")
# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing
print("Indexing:")
print("Element at (0, 1):", arr[0, 1])  # Access element at row 0, column 1
print("Element at (2, 2):", arr[2, 2])  # Access element at row 2, column 2
# Slicing
print("\nSlicing:")
print("First row:", arr[0, :])  # Access first row
print("Second column:", arr[:, 1])  # Access second column
print("Sub-array (0:2, 1:3):", arr[0:2, 1:3])  # Access sub-array from row 0 to 1 and column 1 to 2
# Reshaping
print("\nReshaping:")
reshaped_arr = arr.reshape(1, 9)  # Reshape to a 1x9 array
print("Reshaped array (1x9):", reshaped_arr)
reshaped_arr = arr.reshape(9, 1)  # Reshape to a 9x1 array
print("Reshaped array (9x1):", reshaped_arr)
reshaped_arr = arr.reshape(3, 3)  # Reshape back to a 3x3 array
print("Reshaped array (3x3):", reshaped_arr)
# Iterating
print("\nIterating:")
for row in arr:
    for element in row:
        print(element, end=' ')  # Print each element in the row
    print()  # New line after each row  

# Iterating using np.nditer
print("\nIterating using np.nditer:")
for element in np.nditer(arr):
    print(element, end=' ')  # Print each element in the array
    print()  # New line after all elements

# Iterating using np.ndenumerate
print("\nIterating using np.ndenumerate:")
for index, element in np.ndenumerate(arr):
    print(f"Index: {index}, Element: {element}")  # Print index and element

# Iterating using np.ndindex
print("\nIterating using np.ndindex:")
for index in np.ndindex(arr.shape):
    print(f"Index: {index}, Element: {arr[index]}")  # Print index and corresponding element


# Advanced Topics meshgrid, unique, sort, argsort, isin, setdiff1d, intersect1d, tile, repeat, fromfunction, vectorize
print("\nAdvanced Topics")
# Create two 1D arrays
x = np.array([1, 2, 3])
y = np.array([4, 5, 6, 8])
# Create a meshgrid from the two 1D arrays
X, Y = np.meshgrid(x, y)
print("\nMeshgrid:")
print("X:\n", X)  # Print the meshgrid X
print("Y:\n", Y)  # Print the meshgrid Y

# Unique elements in an array
arr = np.array([1, 2, 2, 3, 4, 4, 5])
unique_elements = np.unique(arr)  # Get unique elements in the array
print("\nUnique elements in the array:", unique_elements)  # Print unique elements

# Sort an array
arr = np.array([5, 2, 9, 1, 5, 6])
sorted_arr = np.sort(arr)  # Sort the array
print("\nSorted array:", sorted_arr)  # Print the sorted array

# Argsort an array
argsorted_indices = np.argsort(arr)  # Get indices that would sort the array
print("\nIndices that would sort the array:", argsorted_indices)  # Print indices that would sort the array

# Check if elements are in an array
arr = np.array([1, 2, 3, 4, 5])
check_elements = np.array([2, 4, 6])
isin_result = np.isin(check_elements, arr)  # Check if elements are in the array
print("\nElements in check_elements that are in arr:", isin_result)  # Print the result of isin

# Set difference between two arrays
set1 = np.array([1, 2, 3, 4, 5])
set2 = np.array([4, 5, 6, 7, 8])
set_diff = np.setdiff1d(set1, set2)  # Get elements in set1 that are not in set2
print("\nElements in set1 that are not in set2:", set_diff)  # Print the set difference

# Intersection of two arrays
intersected_elements = np.intersect1d(set1, set2)  # Get elements that are in both arrays
print("\nElements in both set1 and set2:", intersected_elements)  # Print the intersection of the two arrays

# Tile an array
tiled_arr = np.tile(arr, (2, 3))  # Tile the array 2 times along the first axis and 3 times along the second axis
print("\nTiled array:\n", tiled_arr)  # Print the tiled array
# Repeat elements in an array

repeated_arr = np.repeat(arr, 2)  # Repeat each element in the array 2 times
print("\nRepeated array:", repeated_arr)  # Print the repeated array
# Create an array from a function
def func(i, j):
    return i + j

arr_from_func = np.fromfunction(func, (3, 3), dtype=int)  # Create a 3x3 array from the function
print("\nArray created from function:\n", arr_from_func)  # Print the array created from the function

# Vectorize a function
def my_function(x):
    return x ** 2 + 2 * x + 1
vectorized_function = np.vectorize(my_function)  # Vectorize the function
x_values = np.array([1, 2, 3, 4, 5])    
result = vectorized_function(x_values)  # Apply the vectorized function to the array
print("\nResult of vectorized function:", result)  # Print the result of the vectorized function

# ========================================================================================

    
