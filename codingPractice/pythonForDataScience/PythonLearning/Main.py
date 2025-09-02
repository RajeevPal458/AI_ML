
"""
This is a simple program to demonstrate the use of the `__name__` variable in Python.
It prints the name of the module when run as a script and when imported as a module."""

# My name is here
name = "rajeev pal"
age = 22

if(__name__ == "__main__"):
    print("This script is being run directly")
    print(f"My name is {name} and I am {age} years old.")
else:
    print("This script has been imported as a module")
    print(f"My name is {name} and I am {age} years old.")

if(age > 18):
    print("I am an adult.")
else:
    print("I am a minor.")  

if(age > 18 and age < 60):
    print("I am an adult.")
else:
    print("I am not an adult.")


x = str(3)    # x will be '3'
y = int(3)    # y will be 3
z = float(3)  # z will be 3.0


print(x, y, z)
print(type(x), type(y), type(z))
print("x is of type", type(x))
print("y is of type", type(y))
print("z is of type", type(z))

print("x is of type", type(x), "and its value is", x)
print("y is of type", type(y), "and its value is", y)
print(f"z is of type {type(z)} and its value is {z}")
print(f"z is of type {type(z)} and its value is {z}")

p,q,r = 1,2,3
print(f"p is {p}, q is {q}, r is {r}")

#unpack a collection

my_list = [1, 2, 3]
a, b, c = my_list
print(f"a is {a}, b is {b}, c is {c}")

#unpack a dictionary
my_dict = {"name": "raj", "age": 22}
a, b = my_dict.items()
print(f"a is {a}, b is {b}")

#unpack a set
my_set = {1, 2, 3}
a, b, c = my_set
print(f"a is {a}, b is {b}, c is {c}")

#unpack a tuple
my_tuple = (1, 2, 3)
a, b, c = my_tuple
print(f"a is {a}, b is {b}, c is {c}")

#unpack a string
my_string = "raj"
a, b, c = my_string
print(f"a is {a}, b is {b}, c is {c}")

#unpack a range
my_range = range(1, 4)
a, b, c = my_range
print(f"a is {a}, b is {b}, c is {c}")

#unpack a file
my_file = open("test.txt", "r")
a, b, c = my_file.readlines()
print(f"a is {a}, b is {b}, c is {c}")
my_file.close()

#unpack a generator
my_generator = (x for x in range(1, 4))
a, b, c = my_generator
print(f"a is {a}, b is {b}, c is {c}")

#unpack a zip
my_zip = zip([1, 2, 3], ["a", "b", "c"])
a, b, c = my_zip
print(f"a is {a}, b is {b}, c is {c}")

#unpack a list of tuples
my_list_of_tuples = [(1, "a"), (2, "b"), (3, "c")]
a, b, c = my_list_of_tuples
print(f"a is {a}, b is {b}, c is {c}")

#unpack a list of lists

my_list_of_lists = [[1, "a"], [2, "b"], [3, "c"]]
a, b, c = my_list_of_lists
print(f"a is {a}, b is {b}, c is {c}")

#unpack a list of sets

my_list_of_sets = [{1, "a"}, {2, "b"}, {3, "c"}]
a, b, c = my_list_of_sets
print(f"a is {a}, b is {b}, c is {c}")

#unpack a list of dictionaries

my_list_of_dicts = [{"name": "raj", "age": 22}, {"name": "pal", "age": 23}, {"name": "raj", "age": 24}]
a, b, c = my_list_of_dicts
print(f"a is {a}, b is {b}, c is {c}")

#unpack a list of strings

my_list_of_strings = ["raj", "pal", "raj"]
a, b, c = my_list_of_strings
print(f"a is {a}, b is {b}, c is {c}")

#unpack a list of ranges

my_list_of_ranges = [range(1, 4), range(4, 7), range(7, 10)]
a, b, c = my_list_of_ranges
print(f"a is {a}, b is {b}, c is {c}")

# python demonstration of using function in python how to create and use it 
def my_function(x, y):
    return x + y

def my_function2(x, y):
    return x - y

def my_function3(x, y):
    return x * y

def my_function4(x, y):

    return x / y

def my_function5(x, y):
    return x % y

def my_function6(x, y):
    return x ** y

# demonstrate the use of Lambda in python give sample code
def my_lambda(x, y):
    return (lambda x, y: x + y)(x, y)

# how to use the lambda function in pyth

