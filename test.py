import h5py
import numpy as np

if __name__ == '__main__':

    with h5py.File('processed/trace_dataset.h5', 'r') as file:

        for row in file['traces']:
            print((row == -1).sum())