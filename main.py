import numpy as np

print(np.zeros((3, 4)))
print(np.ones((2, 5)))
print(np.full((2, 3), 7))
print(np.zeros((2,3), dtype=int)+7)
print(np.eye(4))
print(np.arange(0, 10, 2))
print(np.empty((3, 3)))

print(np.linspace(6., 15., num=10))
import timeit
print(timeit.timeit(lambda: np.zeros((1000, 1000)), number=1000), "s")