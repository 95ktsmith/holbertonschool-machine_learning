#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, range(0, 101, 10), histtype='bar', edgecolor='black')
plt.title("Project A")
plt.xlabel("Grades")
plt.xlim(0, 100)
plt.xticks(ticks=range(0, 101, 10))
plt.ylabel("Number of Students")
plt.ylim(0, 30)
plt.show()
