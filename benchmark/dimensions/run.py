
from generate_data import generate_data as gen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../../CLUEstering/')
import CLUEstering as clue

nruns = 10

dims = np.arange(1, 11)

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

parameters = {i: (0., 0., 0.) for i in dims}
parameters[dims[0]] = (2., 10., 1.)
parameters[dims[1]] = (2., 10., 1.)
parameters[dims[2]] = (1.4, 10., 1.)
parameters[dims[3]] = (1.4, 10., 1.)
parameters[dims[4]] = (1.4, 10., 1.)
parameters[dims[5]] = (1.4, 10., 1.)
parameters[dims[6]] = (1., 10., 1.)
parameters[dims[7]] = (1., 20., 1.)
parameters[dims[8]] = (1., 10., 1.5)
parameters[dims[9]] = (1.5, 30., 1.2)

df = pd.DataFrame({'dims': dims})
for backend in clue.backends:
    times = []
    stds = []
    for dim in dims:
        c.set_params(*parameters[dim])
        data = gen(10000, dim, 20, (-50., 50.), 1.)
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
        plt.errorbar(x=dims, y=times, yerr=stds, fmt=style)

df.to_csv('data/dims.csv', index=False)

if sys.argv[1] == 'show':
    plt.grid(lw=0.5, ls='--', alpha=0.5)
    plt.legend(clue.backends)
    plt.xlabel("Number of dimensions")
    plt.ylabel("Execution time (ms)")
    plt.show()
