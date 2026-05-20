#!/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import sys

if len(sys.argv) < 2:
    print('Usage: python plot.py <dataset1> <dataset2> ... <datasetN> [optional] --name=filename --title=plottitle --save')
    sys.exit(1)

saveflag = next((arg for arg in sys.argv if arg == '--save'), None)
if saveflag in sys.argv:
    sys.argv.remove(saveflag)
    save = True
else:
    save = False

plot_title = None
titleflag = next((arg for arg in sys.argv if '--title=' in arg), None)
if titleflag is not None:
    plot_title = titleflag.split('=')[1]
    sys.argv.remove(titleflag)
file_name = 'comparison.pdf'
nameflag = next((arg for arg in sys.argv if '--name=' in arg), None)
if nameflag is not None:
    file_name = nameflag.split('=')[1]
    sys.argv.remove(nameflag)
log = False
if '--log' in sys.argv:
    log = True
    sys.argv.remove('--log')

linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
markers = ['o', 's', 'v', '^', 'h', 'D', '<', '>']
colours = ['royalblue', 'forestgreen', 'tomato', 'darkorange', 'gold', 'slateblue', 'slategray', 'mediumorchid']

datasets = {}
for arg in sys.argv[1:]:
    datasets[arg] = pd.read_csv(arg + '.csv')
    plt.errorbar(datasets[arg]['size'], datasets[arg]['avg'], yerr=datasets[arg]['std'], label=arg, linestyle=linestyles.pop(0), marker=markers.pop(0), color=colours.pop(0), markersize=5, linewidth=1.5)

plt.xlabel('Dataset size', fontsize=14)
plt.ylabel('Execution time (ms)', fontsize=14)
plt.xscale('log', base=2)
if log:
    plt.yscale('log')
plt.legend(fontsize=14)
if plot_title is not None:
    plt.title(plot_title, fontsize=16)
plt.grid(lw=0.5, ls='--')
if save:
    plt.savefig(file_name)
else:
    plt.show()
