# Plot settings
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from matplotlib.colors import Normalize
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

plt.style.use('default')
sns.set_style("ticks")
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter

textwidth = 3.31314*2
aspect_ratio = 6/8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio

titlesize = 14
axessize = 12
xticksize = 10
yticksize = 10
legendsize = 12
figtitlesize = 16
plt.rcParams['axes.titlesize'] = titlesize     # Font size for axes titles
plt.rcParams['axes.labelsize'] = axessize     # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = xticksize    # Font size for x tick labels
plt.rcParams['ytick.labelsize'] = yticksize    # Font size for y tick labels
plt.rcParams['legend.fontsize'] = legendsize    # Font size for the legend
plt.rcParams['figure.titlesize'] = figtitlesize   # Font size for figure title

# Scalar formatter for scientific notation
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))  # Display scientific notation for values outside this range

# Other settings
fontsize = 12
ticks_fontsize = 10
label_x = 0.025
label_y = 0.975





