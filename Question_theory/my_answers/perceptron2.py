import numpy as np

# prepare data
np.random.seed(0)
x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
_x = np.hstack([x, [[1] for i in range(4)]])
t = np.array([[-1], [-1], [-1], [1]], dtype=np.float32)
w = np.random.normal(0., 1, (3))
print(f'weight >> {w}')

# learning w
lr = 0.1
iteration = 1
while True:
    y = _x.dot(w)
    print(f'iteration: {iteration} y >> {y}')
    En = np.zeros(3, dtype=np.float32)
    for i in range(4):
        if y[i] * t[i] < 0:  # 出力とラベルが逆
            En += t[i] * _x[i]
    if np.any(En != 0):
        w += lr * En
        iteration += 1
    else:
        break

# output
print('training finished!')
print(f'weight >> {w}')
for v1, v2 in zip(_x, y):
    print(f'in >> {v1} y >> {v2}')