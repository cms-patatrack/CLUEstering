
from generate_data import generate_data as gen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../../CLUEstering/')
import CLUEstering as clue

nruns = 10

point_values = [2**i for i in range(10, 18)]

line_map = {'cpu serial': '-',
            'cpu tbb': '--',
            'gpu cuda': '-.',
            'gpu hip': ':'}
marker_map = {'cpu serial': 'o',
              'cpu tbb': 'v',
              'gpu cuda': 's',
              'gpu hip': '^'}
color_map = {'cpu serial': 'b',
             'cpu tbb': 'r',
             'gpu cuda': 'g',
             'gpu hip': 'y'}

c = clue.clusterer(2., 10, 1.)

parameters = {i: (0., 0., 0.) for i in point_values}
parameters[point_values[0]] = (2., 10., 1.)
parameters[point_values[1]] = (2., 10., 1.)
parameters[point_values[2]] = (2., 10., 1.)
parameters[point_values[3]] = (2., 10., 1.)
parameters[point_values[4]] = (2., 10., 1.)
parameters[point_values[5]] = (2., 10., 1.)

for backend in clue.backends:
    times = []
    stds = []
    for val in point_values:
        c.set_params(*parameters[val])
        data = gen(val, 2, 20, (-50., 50.))
        c.read_data(data)
        runs = []
        for _ in range(nruns):
            c.run_clue(backend)
            runs.append(c.elapsed_time)
        times.append(np.mean(runs))
        stds.append(np.std(runs))
        # c.cluster_plotter()
    style = f"{line_map[backend]}{marker_map[backend]}{color_map[backend]}"
    if sys.argv[1] == 'show':
        plt.errorbar(x=point_values, y=times, yerr=stds, fmt=style)

pd.DataFrame({'points': point_values,
              'cpu serial': times[0],
              'cpu tbb': times[1],
              'gpu cuda': times[2],
              'cpu serial std': stds[0],
              'cpu tbb std': stds[1],
              'gpu cuda std': stds[2]}).to_csv('dset_size.csv', index=False)

if sys.argv[1] == 'show':
    plt.grid(lw=0.5, ls='--', alpha=0.5)
    plt.legend(clue.backends)
    plt.xlabel("Number of points in dataset")
    plt.ylabel("Execution time (ms)")
    plt.xscale('log', base=2)
    plt.show()
