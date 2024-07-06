
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import sys
sys.path.insert(1, '../../CLUEstering/')
import CLUEstering as clue

data_folder = "../../tests/test_datasets/"

save = False
if len(sys.argv) > 1:
    if sys.argv[1] == "save":
        save = True


def run(dc: float, rhoc: float, odf: float,
        dataset: str, backend: str, block_size: int) -> float:
    c = clue.clusterer(dc, rhoc, odf)
    c.read_data(data_folder + dataset)
    c.run_clue(backend, block_size)

    return c.elapsed_time


def benchmark(nruns: int, dataset: str, backend: str, threads: list) -> np.ndarray:
    times = np.zeros((len(threads), 2))
    for i, bs in enumerate(threads):
        partial_times = np.zeros(nruns)
        for r in range(nruns):
            partial_times[r] = run(0.2, 5, 1., dataset, backend, bs)
            sleep(.1)
        times[i] = np.array([np.mean(partial_times), np.std(partial_times)])

    return times


if __name__ == "__main__":
    nruns = 100
    threads = [2**i for i in range(0, 11)]
    line_map = {'cpu serial': '|',
                'cpu tbb': '--',
                'gpu cuda': '.-',
                'gpu hip': ':'}
    marker_map = {'cpu serial': 'O',
                  'cpu tbb': '*',
                  'gpu cuda': '^',
                  'gpu hip': 's'}
    color_map = {'cpu serial': 'b',
                 'cpu tbb': 'r',
                 'gpu cuda': 'g',
                 'gpu hip': 'y'}

    #
    # blobs 2D dataset
    for backend in clue.backends:
        if backend == "cpu serial":
            continue
        times = benchmark(nruns, "blobs.csv", backend, threads)

        style = line_map[backend] + marker_map[backend] + color_map[backend]
        # plt.plot(threads, times.T[0], style)
        plt.errorbar(x=threads, y=times.T[0], yerr=times.T[1], fmt=style)
    plt.title("Blob dataset")
    plt.grid(ls="--", lw=0.5, axis='y')
    plt.legend(clue.backends[1:])
    plt.xlabel("Number of threads per block")
    plt.ylabel("Execution time (ms)")
    if save:
        plt.savefig("blobs_threads_backends.png")
    else:
        plt.show()
