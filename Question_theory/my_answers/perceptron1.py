import numpy as np

# prepare data
np.random.seed(0)
x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
t = np.array([[-1], [-1], [-1], [1]], dtype=np.float32)
w = np.random.normal(0., 1, (3))

# calculate y
x2 = np.insert(x, 2, 1, axis=1)
y = x2.dot(w)

# output
for v1, v2 in zip(x2, y):
    print(f'in >> {v1} y >> {v2}')