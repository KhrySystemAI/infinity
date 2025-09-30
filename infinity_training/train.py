import numpy as np

a = np.array([[-5, -1],
          [6, 4]])

e = np.array([4,2])
f = np.array([[-1],[2]])
print(a)
print(e)
print(f)
print(a * f)
print( e * (a * f))