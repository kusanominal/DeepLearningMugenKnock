import numpy as np
import matplotlib.pyplot as plt
from logzero import logger

# prepare data
np.random.seed(0)
x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
_x = np.hstack([x, [[1] for i in range(4)]])
t = np.array([[-1], [-1], [-1], [1]], dtype=np.float32)
w = np.random.normal(0., 1, (3))
logger.info(f'weight >> {w}')

# learning w
def learn_weight(w, lr):
    history = [[], [], []]
    iteration = 1

    while True:
        # update history
        for i in range(3):
            history[i].append(w[i])
        y = _x.dot(w)
        logger.info(f'iteration: {iteration} y >> {y}')
        En = np.zeros(3, dtype=np.float32)
        for i in range(4):
            if y[i] * t[i] < 0:  # 出力とラベルが逆
                En += t[i] * _x[i]
        if np.any(En != 0):
            w += lr * En
            iteration += 1
        else:
            break

    # output text
    logger.info('training finished!')
    logger.info(f'weight >> {w}')
    for v1, v2 in zip(_x, y):
        logger.info(f'in >> {v1} y >> {v2}')

    return history

h1 = learn_weight(w.copy(), 0.1)
h2 = learn_weight(w.copy(), 0.01)

# output figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(3):
    ax.plot(h1[i], linestyle='solid')
    ax.plot(h2[i], linestyle='--')

ax.set_title('weight plot')
ax.set_xlabel('iteration')
ax.set_ylabel('weight')
ax.grid(True)
plt.show()