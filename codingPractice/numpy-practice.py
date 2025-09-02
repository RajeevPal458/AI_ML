import numpy as np

num_list = [1, 2, 3, 4, 5]
numpy_arr = np.array(num_list)
print(f'numpy array : {numpy_arr}')
print('numpy array :', numpy_arr)

num_list = np.zeros(5)
print(f'{num_list} numpy zeros array 1D')
print()
num_list = np.zeros((2, 3))
print(f'{num_list} numpy zeros array 2 D')

print()
num_list = np.ones(5)
print(f'{num_list} numpy ones array 1D')
print()
num_list = np.ones((2, 3))
print(f'{num_list} numpy ones array 2D')
print()

num_list = np.empty(5)
print(f'{num_list} numpy empty array 1D')
print()
num_list = np.empty((2, 3))
print(f'{num_list} numpy empty array 2D')
print()

num_list = np.full(5, 7)
print(f'{num_list} numpy full array 1D')
print()

num_list = np.full((2, 3), 7)
print(f'{num_list} numpy full array 2D')
print()

num_list = np.full((2, 3), [1, 2, 3])
print(f'{num_list} numpy full array 2D')
print()

num_list = np.arange(5)
print(f'{num_list} numpy arange array 1D')
print()

num_list = np.arange(5, 10)
print(f'{num_list} numpy arange array 1D')
print()

num_list = np.arange(5, 10, 2)
print(f'{num_list} numpy arange array 1D')
print()

num_list = np.arange(5, 10, 0.5)
print(f'{num_list} numpy arange array 1D')
print()

num_list = np.arange(5, 10, 0.5, dtype=int)
print(f'{num_list} numpy arange array 1D')
print()

num_list = np.linspace(5, 10, 5)
print(f'{num_list} numpy linspace array 1D')
print()
num_list = np.linspace(5, 10, 5, dtype=int)
print(f'{num_list} numpy linspace array 1D')
print()

num_list = np.linspace(5, 10, 5, endpoint=False)
print(f'{num_list} numpy linspace array 1D')
print()

num_list = np.linspace(5, 10, 5, retstep=True)
print(f'{num_list} numpy linspace array 1D')
print()

num_list = np.linspace(5, 10, 5, retstep=True, dtype=int)
print(f'{num_list} numpy linspace array 1D')
print()

num_list = np.linspace(5, 10, 5, retstep=True, endpoint=False)
print(f'{num_list} numpy linspace array 1D')
print()

num_list = np.linspace(5, 10, 5, retstep=True, endpoint=False, dtype=int)
print(f'{num_list} numpy linspace array 1D')
print()

num_list = np.eye(3)
print(f'{num_list} numpy eye array 2D')
print()
num_list = np.eye(3, 4)
print(f'{num_list} numpy eye array 2D')
print()
num_list = np.eye(3, 4, k=1)
print(f'{num_list} numpy eye array 2D')
print()

num_list = np.eye(3, 4, k=-1)
print(f'{num_list} numpy eye array 2D')
print()

num_list = np.eye(3, 4, k=1, dtype=int)
print(f'{num_list} numpy eye array 2D')
print()

num_list = np.eye(3, 4, k=-1, dtype=int)
print(f'{num_list} numpy eye array 2D')
print()

num_list = np.identity(3)
print(f'{num_list} numpy identity array 2D')
print()

num_list = np.identity(3, dtype=int)
print(f'{num_list} numpy identity array 2D')
print()

num_list = np.identity(3, dtype=float)
print(f'{num_list} numpy identity array 2D')
print()

num_list = np.identity(3, dtype=complex)
print(f'{num_list} numpy identity array 2D')
print()

num_list = np.identity(3, dtype=bool)
print(f'{num_list} numpy identity array 2D')
print()

random_arr = np.random.rand(5)
print(f'{random_arr} numpy rand array 1D')
print()

random_arr = np.random.rand(2, 3)
print(f'{random_arr} numpy rand array 2D')
print()

random_arr = np.random.randn(5)
print(f'{random_arr} numpy randn array 1D')
print()

random_arr = np.random.randn(2, 3)
print(f'{random_arr} numpy randn array 2D')
print()
random_arr = np.random.randint(5)
print(f'{random_arr} numpy randint array 1D')
print()
random_arr = np.random.randint(5, 10)
print(f'{random_arr} numpy randint array 1D')
print()

random_arr = np.random.randint(5, 10, 5)
print(f'{random_arr} numpy randint array 1D')
print()

random_arr = np.random.randint(5, 10, (2, 3))
print(f'{random_arr} numpy randint array 2D')
print()

random_arr = np.random.randint(5, 10, (2, 3), dtype=int)
print(f'{random_arr} numpy randint array 2D')
print()