import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

# creating the dataset
data = {'DReaLNav':99 , 'SBP-DRL':95, 'GRainbow':91, 'Baseline TD3':90, 'Rainbow': 73}
courses = list(data.keys())
values = list(data.values())

fig= plt.figure(figsize = (10, 6))
plt.grid(True, linestyle='--')
ax = plt.gca()
ax.set_axisbelow(True)
# creating the bar plot
plt.bar(courses, values,
        width = 0.7, color=['purple', 'red', 'green', 'blue', 'brown'])
plt.rc('axes', axisbelow=True)
plt.xlabel("Algorithm", fontsize=25, fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.ylabel("Success Rate (%)", fontsize=25,fontweight='bold')
# plt.title("")
plt.ylim([60, 100])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# XX = pd.Series([99,95,92,91,90],index=['DReaLNav','SME-DRL','ProVe','GRainbow', 'baseline TD3'])
# fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,
#                          figsize=(5,6), gridspec_kw={'height_ratios': [4, 1]})
# ax1.spines['bottom'].set_visible(False)
# ax1.tick_params(axis='x',which='both',bottom=False)
# ax2.spines['top'].set_visible(False)
# ax2.set_ylim(0,20)
# ax1.set_ylim(80,100)
# # ax1.set_yticks(np.arange(1,101,1))
# XX.plot(ax=ax1,kind='bar',color=['black', 'red', 'green', 'blue', 'cyan'])
# XX.plot(ax=ax2,kind='bar', color=['black', 'red', 'green', 'blue', 'cyan'])
# for tick in ax2.get_xticklabels():
#     tick.set_rotation(0)
# d = .015
# kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
# ax1.plot((-d, +d), (-d, +d), **kwargs)
# ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
# kwargs.update(transform=ax2.transAxes)
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
# fig.tight_layout()

# fig.supylabel('Success Rate (%)')
# fig.supxlabel('Algorithm')
# plt.grid(True, linestyle='--')
# plt.show()