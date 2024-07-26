
from generate_data import generate_data as gen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../../CLUEstering/')
import CLUEstering as clue

nruns = 10
# nruns = 1

point_values = [2**i for i in range(10, 20)]

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
parameters[point_values[2]] = (1.4, 10., 1.)
parameters[point_values[3]] = (1.4, 10., 1.)
parameters[point_values[4]] = (1.4, 10., 1.)
parameters[point_values[5]] = (1.4, 10., 1.)
parameters[point_values[6]] = (1., 10., 1.)
parameters[point_values[7]] = (1., 20., 1.)
parameters[point_values[8]] = (1., 10., 1.5)
parameters[point_values[9]] = (1.5, 30., 1.2)

df = pd.DataFrame({'points': point_values})
for backend in clue.backends:
    times = []
    stds = []
    for val in point_values:
        c.set_params(*parameters[val])
        data = gen(val, 2, 20, (-100., 100.), 1.)
        c.read_data(data)
        runs = []
        for _ in range(nruns):
            if backend == 'gpu cuda':
                c.run_clue(backend)
            c.run_clue(backend)
            runs.append(c.elapsed_time)
        times.append(np.mean(runs))
        stds.append(np.std(runs))
        # c.cluster_plotter()
    df = pd.concat((df,
                    pd.DataFrame({backend: times,
                                  str(backend + ' std'): stds})),
                   axis=1)
    style = f"{line_map[backend]}{marker_map[backend]}{color_map[backend]}"
    if sys.argv[1] == 'show':
        plt.errorbar(x=point_values, y=times, yerr=stds, fmt=style)

df.to_csv('dset_size.csv', index=False)

if sys.argv[1] == 'show':
    plt.grid(lw=0.5, ls='--', alpha=0.5)
    plt.legend(clue.backends)
    plt.xlabel("Number of points in dataset")
    plt.ylabel("Execution time (ms)")
    plt.xscale('log', base=2)
    plt.show()
