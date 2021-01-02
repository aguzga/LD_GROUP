import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv

k = 8
b = 6
arr = np.load("data/st_heat_k_{}_b_{}_final.npy".format(k, b)) # static
arr2 = np.load("data/heat_k_{}_b_{}_final.npy".format(k, b))   # shuffled

as_st = np.load("data/st_assort_k_{}_b_{}_final.npy".format(k, b)) # static


arr_ = arr - arr2 # difference beteen coop levels of shuffled vs static with same params



# create new range for the heatmap (0.4 to 0.8 in the paper)
OldMax = np.amax(arr)
OldMin = np.amin(arr)
NewMax = 0.8
NewMin = 0.4
OldRange = (OldMax - OldMin)
NewRange = (NewMax - NewMin)
for row in arr:
    for NewValue in row:
        NewValue = (((NewValue - OldMin) * NewRange) / OldRange) + NewMin


OldMax = np.amax(arr2)
OldMin = np.amin(arr2)
NewMax = 0.8
NewMin = 0.4
OldRange = (OldMax - OldMin)
NewRange = (NewMax - NewMin)
for row in arr2:
    for NewValue in row:
        NewValue = (((NewValue - OldMin) * NewRange) / OldRange) + NewMin


OldMax = np.amax(arr_)
OldMin = np.amin(arr_)
NewMax = 0.2
NewMin = -0.2
OldRange = (OldMax - OldMin)
NewRange = (NewMax - NewMin)
for row in arr_:
    for NewValue in row:
        NewValue = (((NewValue - OldMin) * NewRange) / OldRange) + NewMin


OldMax = np.amax(as_st)
OldMin = np.amin(as_st)
NewMax = 0.2
NewMin = -0.2
OldRange = (OldMax - OldMin)
NewRange = (NewMax - NewMin)
for row in as_st:
    for NewValue in row:
        NewValue = (((NewValue - OldMin) * NewRange) / OldRange) + NewMin




# static
cm = sns.color_palette("rainbow", as_cmap=True)
ax = sns.heatmap(arr, square = True, vmin = 0.4, vmax = 0.8, annot = False, fmt = ".4f", cmap = cm, cbar = False)
ax.invert_yaxis()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
ax.tick_params(bottom=True, top=True, left=True, right=True)
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
plt.title("k = {}".format(k), size = 40)
if(k == 2):
    ytl = [None for i in range(3)]
    ytl.insert(0, 0.0)
    ytl.append(0.5)
    ax.set_yticklabels(ytl, size = 40)
    ax.set_xticklabels([None for i in range(5)], size = 40)
else:
    ax.set_xticklabels([None for i in range(5)], size = 40)
    ax.set_yticklabels([None for i in range(5)], size = 40)
if ('-s' in argv):
    plt.savefig("_output/Static_k_{}_b_{}.png".format(k, b))
else:
    plt.show()
plt.clf()


# shuffled
cm = sns.color_palette("rainbow", as_cmap=True)
ax = sns.heatmap(arr2, square = True, vmin = 0.4, vmax = 0.8, annot = False, fmt = ".4f", cmap = cm, cbar = False)
ax.invert_yaxis()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
ax.tick_params(bottom=True, top=True, left=True, right=True)
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
if(k == 2):
    ytl = [None for i in range(3)]
    ytl.insert(0, 0.0)
    ytl.append(0.5)
    ax.set_yticklabels(ytl, size = 40)
    ax.set_xticklabels([None for i in range(5)], size = 40)
else:
    ax.set_xticklabels([None for i in range(5)], size = 40)
    ax.set_yticklabels([None for i in range(5)], size = 40)
if ('-s' in argv):
    plt.savefig("_output/Shuffled_k_{}_b_{}.png".format(k, b))
else:
    plt.show()
plt.clf()


# difference
cm = sns.color_palette("vlag", as_cmap=True)
ax = sns.heatmap(arr_, square = True, vmin = -0.2, vmax = 0.2, annot = False, fmt = ".4f", cmap = cm, cbar = False)
ax.invert_yaxis()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
ax.tick_params(bottom=True, top=True, left=True, right=True)
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
if(k == 2):
    ytl = [None for i in range(3)]
    ytl.insert(0, 0.0)
    ytl.append(0.5)
    ax.set_yticklabels(ytl, size = 40)
    ax.set_xticklabels([None for i in range(5)], size = 40)
else:
    ax.set_xticklabels([None for i in range(5)], size = 40)
    ax.set_yticklabels([None for i in range(5)], size = 40)
if ('-s' in argv):
    plt.savefig("_output/Difference_k_{}_b_{}.png".format(k, b))
else:
    plt.show()
plt.clf()


# assortment
cm = sns.color_palette("vlag", as_cmap=True)
ax = sns.heatmap(as_st, square = True, vmin = -0.2, vmax = 0.2, annot = False, fmt = ".4f", cmap = cm, cbar = False)
ax.invert_yaxis()
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 20))
ax.yaxis.set_ticks(np.arange(start, end, 20))
ax.tick_params(bottom=True, top=True, left=True, right=True)
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
if(k == 2):
    ytl = [None for i in range(3)]
    xtl = [None for i in range(3)]
    ytl.insert(0, 0.0)
    xtl.insert(0, -1)
    ytl.append(0.5)
    xtl.append(5)
    ax.set_yticklabels(ytl, size = 40)
    ax.set_xticklabels(xtl, size = 40, rotation = -360)
else:
    xtl = [None for i in range(3)]
    xtl.insert(0, -1)
    xtl.append(5)
    ax.set_xticklabels(xtl, size = 40, rotation = -360)
    ax.set_yticklabels([None for i in range(5)], size = 40)
if ('-s' in argv):
    plt.savefig("_output/Assortment_k_{}_b_{}.png".format(k, b))
else:
    plt.show()
plt.clf()
