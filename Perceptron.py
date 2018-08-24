import numpy as np
import random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Activation Function
def f_at(x):
    if x >= 0:
        return 1
    else:
        return -1

# Training set

# Input set
x = np.array([[-1, 0.1, 0.4, 0.7], [-1, 0.3, 0.7, 0.2], [-1, 0.6, 0.9, 0.8], [-1, 0.5, 0.7, 0.1]])

# Result set
d = np.array([1, -1, 1, 1])
#x[0] ~ x[3] and x[1] ~ x[2]

# bias
bias = 0.5

# Weight set
w = np.array([bias, 0.1, 0.2, 0.3])

# Learning constant
eta = 0.1

# Epochs
epoca = 0
erro = 1
while erro == 1:
    erro = 0
    for i in range(len(x)):
        u = np.sum(x[i] * w)
        y = f_at(u)
        if y != d[i]:
            w = w + eta*(d[i] - y)*x[i]
            erro = 1
        epoca += 1

print("Quantidade de epocas: ", epoca)
print("Valores finais dos pesos:", w)

# Classificação

#Dados para serem classificados
x2 = np.array([ [-1, rd.random(), rd.random(), rd.random()],
                [-1, rd.random(), rd.random(), rd.random()],
                [-1, rd.random(), rd.random(), rd.random()]])

cor = np.array(['','',''], dtype=object)

for i in range(len(x2)):
    u = np.sum(w*x2[i])
    y = f_at(u)
    if y == 1:
        cor[i] = 'red'
    else:
        cor[i] = 'green'

# Gráfico

fig = plt.figure()
ax = fig.gca(projection='3d')

xx,yy = np.meshgrid(range(2), range(2))

z = (-w[1]*xx -w[2]*yy + w[0])/w[3]
ax.plot_surface(xx, yy,z, color='white', alpha = 0.9)

#ax.scatter(x[0][1], x[0][2], x[0][3], color='green')
#ax.scatter(x[1][1], x[1][2], x[1][3], color='red')
#ax.scatter(x[2][1], x[2][2], x[2][3], color='green')
#ax.scatter(x[3][1], x[3][2], x[3][3], color='green')

ax.scatter(x2[0][1], x2[0][2], x2[0][3], color=cor[0])
ax.scatter(x2[1][1], x2[1][2], x2[1][3], color=cor[1])
ax.scatter(x2[2][1], x2[2][2], x2[2][3], color=cor[2])

plt.show()
