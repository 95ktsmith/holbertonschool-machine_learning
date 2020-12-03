#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, axs = plt.subplots(3, 2, gridspec_kw={'wspace': .4, 'hspace': 1})
fig.suptitle("All in One")

# Plot 0
axs[0][0].set_xlim(0, 10)
axs[0][0].plot(range(0, 11), y0, color='red')

# Plot 1
axs[0][1].scatter(x1, y1, s=9, color='magenta')
axs[0][1].set_title("Men's Height vs Weight", size='x-small')
axs[0][1].set_xlabel("Height (in)", size='x-small')
axs[0][1].set_ylabel("Weight (lbs)", size='x-small')

# Plot 2
axs[1][0].plot(x2, y2)
axs[1][0].set_yscale("log")
axs[1][0].set_title("Exponential Decay of C-14", size='x-small')
axs[1][0].set_ylabel("Fraction Remaining", size='x-small')
axs[1][0].set_xlabel("Time (years)", size='x-small')
axs[1][0].set_xlim(0, 28650)

# Plot 3
axs[1][1].set_title("Exponential Decay of Radioactive Elements",
                    size='x-small')
axs[1][1].set_xlim(0, 20000)
axs[1][1].set_xlabel("Time (years)", size='x-small')
axs[1][1].set_ylim(0, 1)
axs[1][1].set_ylabel("Fraction Remaining", size='x-small')
axs[1][1].plot(x3, y31, 'r--')
axs[1][1].plot(x3, y32, 'g-')
axs[1][1].legend(["C-14", "Ra-226"], fontsize="x-small")

# Plot 4
ax4 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2)
ax4.hist(student_grades, range(0, 101, 10), histtype='bar', edgecolor='black')
ax4.set_title("Project A", size='x-small')
ax4.set_xlabel("Grades", size='x-small')
ax4.set_xlim(0, 100)
ax4.set_xticks(ticks=range(0, 101, 10))
ax4.set_ylabel("Number of Students", size='x-small')
ax4.set_ylim(0, 30)
ax4.set_aspect(1/2)

plt.show()
