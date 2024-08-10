
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def partial_sum(arr: list, idx: int) -> float:
    return sum([arr[i] for i in range(idx)])

legend = ['Tiles setup',
          'Memcpy HostToDevice',
          'Calculate local density',
          'Calculate nearest higher',
          'Find clusters',
          'Assign clusters',
          'Memcpy DeviceToHost']
cols = ['setup', 'HtD', 'CLD', 'CNH', 'FC', 'AC', 'DtH']

toycpu = pd.read_csv('./data/gpu-c2a02-35-01/toy_cpu.csv')
# dataframe containing the mean of each column
toycpu_means = [toycpu[col].values.mean() for col in toycpu.columns]
# stack the data in the array 'toycpu_mean' in vertical bars
plt.bar('toy_cpu', toycpu_means[0], bottom=0, color='C0')
plt.bar('toy_cpu', toycpu_means[1], bottom=partial_sum(toycpu_means, 1), color='C1')
plt.bar('toy_cpu', toycpu_means[2], bottom=partial_sum(toycpu_means, 2), color='C2')
plt.bar('toy_cpu', toycpu_means[3], bottom=partial_sum(toycpu_means, 3), color='C3')
plt.bar('toy_cpu', toycpu_means[4], bottom=partial_sum(toycpu_means, 4), color='C4')
plt.bar('toy_cpu', toycpu_means[5], bottom=partial_sum(toycpu_means, 5), color='C5')
plt.bar('toy_cpu', toycpu_means[6], bottom=partial_sum(toycpu_means, 6), color='darkgreen')

toytbb = pd.read_csv('./data/gpu-c2a02-35-01/toy_tbb.csv')
# dataframe containing the mean of each column
toytbb_means = [toytbb[col].values.mean() for col in toytbb.columns]
# stack the data in the array 'toycpu_mean' in vertical bars
plt.bar('toy_tbb', toytbb_means[0], bottom=0, color='C0')
plt.bar('toy_tbb', toytbb_means[1], bottom=partial_sum(toytbb_means, 1), color='C1')
plt.bar('toy_tbb', toytbb_means[2], bottom=partial_sum(toytbb_means, 2), color='C2')
plt.bar('toy_tbb', toytbb_means[3], bottom=partial_sum(toytbb_means, 3), color='C3')
plt.bar('toy_tbb', toytbb_means[4], bottom=partial_sum(toytbb_means, 4), color='C4')
plt.bar('toy_tbb', toytbb_means[5], bottom=partial_sum(toytbb_means, 5), color='C5')
plt.bar('toy_tbb', toytbb_means[6], bottom=partial_sum(toytbb_means, 6), color='darkgreen')

toygpu = pd.read_csv('./data/gpu-c2a02-35-01/toy_gpu.csv')
# dataframe containing the mean of each column
toygpu_means = [toygpu[col].values.mean() for col in toygpu.columns]
# stack the data in the array 'toycpu_mean' in vertical bars
plt.bar('toy_gpu', toygpu_means[0], bottom=0, color='C0')
plt.bar('toy_gpu', toygpu_means[1], bottom=partial_sum(toygpu_means, 1), color='C1')
plt.bar('toy_gpu', toygpu_means[2], bottom=partial_sum(toygpu_means, 2), color='C2')
plt.bar('toy_gpu', toygpu_means[3], bottom=partial_sum(toygpu_means, 3), color='C3')
plt.bar('toy_gpu', toygpu_means[4], bottom=partial_sum(toygpu_means, 4), color='C4')
plt.bar('toy_gpu', toygpu_means[5], bottom=partial_sum(toygpu_means, 5), color='C5')
plt.bar('toy_gpu', toygpu_means[6], bottom=partial_sum(toygpu_means, 6), color='darkgreen')

plt.ylabel('Time ($\mu$s)', fontsize=12)
plt.legend(legend)
plt.title('Intel(R) Xeon(R) Gold 6130 CPU, Tesla T4 GPU', fontsize=14)
plt.savefig('toy_tracing.png')

plt.clf()
sissacpu = pd.read_csv('./data/gpu-c2a02-35-01/sissa_cpu.csv')
# dataframe containing the mean of each column
sissacpu_means = [sissacpu[col].values.mean() for col in sissacpu.columns]
# stack the data in the array 'toycpu_mean' in vertical bars
plt.bar('sissa_cpu', sissacpu_means[0], bottom=0, color='C0')
plt.bar('sissa_cpu', sissacpu_means[1], bottom=partial_sum(sissacpu_means, 1), color='C1')
plt.bar('sissa_cpu', sissacpu_means[2], bottom=partial_sum(sissacpu_means, 2), color='C2')
plt.bar('sissa_cpu', sissacpu_means[3], bottom=partial_sum(sissacpu_means, 3), color='C3')
plt.bar('sissa_cpu', sissacpu_means[4], bottom=partial_sum(sissacpu_means, 4), color='C4')
plt.bar('sissa_cpu', sissacpu_means[5], bottom=partial_sum(sissacpu_means, 5), color='C5')
plt.bar('sissa_cpu', sissacpu_means[6], bottom=partial_sum(sissacpu_means, 6), color='darkgreen')

sissatbb = pd.read_csv('./data/gpu-c2a02-35-01/sissa_tbb.csv')
# dataframe containing the mean of each column
sissatbb_means = [sissatbb[col].values.mean() for col in sissatbb.columns]
# stack the data in the array 'toycpu_mean' in vertical bars
plt.bar('sissa_tbb', sissatbb_means[0], bottom=0, color='C0')
plt.bar('sissa_tbb', sissatbb_means[1], bottom=partial_sum(sissatbb_means, 1), color='C1')
plt.bar('sissa_tbb', sissatbb_means[2], bottom=partial_sum(sissatbb_means, 2), color='C2')
plt.bar('sissa_tbb', sissatbb_means[3], bottom=partial_sum(sissatbb_means, 3), color='C3')
plt.bar('sissa_tbb', sissatbb_means[4], bottom=partial_sum(sissatbb_means, 4), color='C4')
plt.bar('sissa_tbb', sissatbb_means[5], bottom=partial_sum(sissatbb_means, 5), color='C5')
plt.bar('sissa_tbb', sissatbb_means[6], bottom=partial_sum(sissatbb_means, 6), color='darkgreen')

sissagpu = pd.read_csv('./data/gpu-c2a02-35-01/sissa_gpu.csv')
# dataframe containing the mean of each column
sissagpu_means = [sissagpu[col].values.mean() for col in sissagpu.columns]
# stack the data in the array 'toycpu_mean' in vertical bars
plt.bar('sissa_gpu', sissagpu_means[0], bottom=0, color='C0')
plt.bar('sissa_gpu', sissagpu_means[1], bottom=partial_sum(sissagpu_means, 1), color='C1')
plt.bar('sissa_gpu', sissagpu_means[2], bottom=partial_sum(sissagpu_means, 2), color='C2')
plt.bar('sissa_gpu', sissagpu_means[3], bottom=partial_sum(sissagpu_means, 3), color='C3')
plt.bar('sissa_gpu', sissagpu_means[4], bottom=partial_sum(sissagpu_means, 4), color='C4')
plt.bar('sissa_gpu', sissagpu_means[5], bottom=partial_sum(sissagpu_means, 5), color='C5')
plt.bar('sissa_gpu', sissagpu_means[6], bottom=partial_sum(sissagpu_means, 6), color='darkgreen')

plt.ylabel('Time ($\mu$s)', fontsize=12)
plt.legend(legend)
plt.title('Intel(R) Xeon(R) Gold 6130 CPU, Tesla T4 GPU', fontsize=14)
plt.savefig('sissa_tracing.png')
