#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

for i in range(0, len(fruit[0])):
    plt.bar(i, fruit[0][i], width=0.5, color='red')
    bottom = fruit[0][i]
    plt.bar(i, fruit[1][i], width=0.5, bottom=bottom, color='yellow')
    bottom += fruit[1][i]
    plt.bar(i, fruit[2][i], width=0.5, bottom=bottom, color='#ff8000')
    bottom += fruit[2][i]
    plt.bar(i, fruit[3][i], width=0.5, bottom=bottom, color='#ffe5b4')

plt.xticks([0, 1, 2], labels=["Farrah", "Fred", "Felicia"])
plt.ylabel("Quantity of Fruit")
plt.yticks(ticks=range(0, 81, 10))
plt.legend(["apples", "bananas", "oranges", "peaches"])
plt.show()
