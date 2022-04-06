

import cycler
import itertools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["grid.linewidth"] = 0.75
matplotlib.rcParams['lines.markersize'] = 8.0
matplotlib.rcParams['lines.markeredgewidth'] = 0.0
sns.set(font_scale=1.5, font="Arial")
sns.set_palette("tab10")
sns.set_style("whitegrid", {'grid.linestyle': '--'})

cmap = plt.get_cmap("tab10")
