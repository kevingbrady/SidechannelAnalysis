import base64

import h5py
from torch.utils.data import IterableDataset, Dataset
from pathlib import Path
import torch
import os
import sys
import trsfile
import numpy as np
import h5py as h5
import json
import shutil

class TraceFileDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_dir=''):
        super().__init__()

        self.device = None
        self.free_gpu_memory = 0

        self.data_path = data_dir
        self.traces = None
        self.labels = None
        self.transform = transform
        self.total_traces = 0
        self.batch_size = 1
        self.pad_value = 255
        self.files = self.get_file_trace_lengths(data_dir)
        self.max_trace_file_length = max({y for x, y in self.files.values()})

        if os.path.exists('processed/') and \
            os.path.exists('processed/trace_data.h5'):
            print('Database found ...')

        else:
            print('Database not found. Generating new h5 files ...')
            self.build_trace_dataset()

    def __len__(self):
        return self.total_traces

    def __getitem__(self, idx):


        trace_data = h5py.File('processed/trace_data.h5', 'r', swmr=True)

        data = trace_data['traces'][idx]
        label = trace_data['labels'][idx]

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        if self.transform:
            data = self.transform(data)

        return data, label

    def trace_collate_fn(self, batch):
        data, label = zip(*batch)

        data_batch = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(list(data)),
            padding=self.pad_value,
            output_size=(self.batch_size, self.max_trace_file_length)
        )

        #data_batch = self.min_max_scaling_padding_ignore(data_batch, self.pad_value)
        #data_batch = self.z_score_normalization_padding_ignore(data_batch, self.pad_value)
        label_batch = torch.stack(label, dim=0)

        data_batch = data_batch.unsqueeze(1)

        return data_batch, label_batch


    def z_score_normalization_padding_ignore(self, data, pad_value):

        mask = (data != pad_value)

        # Compute zscore ignoring padded values
        masked_data = data[mask].float()

        mean = masked_data.mean()
        std = masked_data.std()

        normalized = torch.full_like(data, pad_value, dtype=torch.float32)
        normalized[mask] = (masked_data - mean) / (std + 1e-8)
        return normalized

    def min_max_scaling_padding_ignore(self, data, pad_value):

        mask = (data != pad_value)

        # Compute min/max ignoring padded values
        masked_data = data[mask].float()

        d_min = masked_data.min()
        d_max = masked_data.max()

        scaled_data = torch.full_like(data, pad_value, dtype=torch.float32)
        scaled_data[mask] = (masked_data - d_min) / (d_max - d_min + 1e-8)

        return scaled_data

    @staticmethod
    def check_trace_padding(trace, pad_width, pad_value=0):

        if len(trace) == pad_width:
            return trace

        return np.pad(trace, (0, pad_width - len(trace)), mode='constant', constant_values=pad_value)

    def build_trace_dataset(self):

        Path('processed/trace_data.h5').unlink(missing_ok=True)

        trace_idx = 0
        chunk_size = 100
        shard_size = 10000

        with h5py.File('processed/trace_data.h5', 'w') as F:

            var_len = h5py.special_dtype(vlen=np.float32)
            self.traces = F.create_dataset('traces', shape=(self.total_traces, ), chunks=(chunk_size, ), compression="gzip", dtype=var_len)
            self.labels = F.create_dataset('labels', shape=(self.total_traces, 16), chunks=(chunk_size, 16), compression="gzip", dtype=np.uint8)

            for file, (file_length, file_width) in self.files.items():

                if file.endswith('.h5'):
                    with h5.File(file, 'r') as trace_file:

                        if trace_file.get('traces') is None:

                            for key in trace_file.keys():
                                trace_list = trace_file[key].get('traces')
                                label_list = trace_file[key].get('metadata')['key']

                                for idx in range(0, trace_list.shape[0], shard_size):
                                    print(idx, trace_idx, shard_size)
                                    self.traces[trace_idx:trace_idx+shard_size] = trace_list[idx:idx+shard_size]
                                    self.labels[trace_idx:trace_idx+shard_size] = label_list[idx:idx+shard_size]
                                    trace_idx += shard_size

                        else:
                            trace_list = trace_file.get('traces')
                            label_list = trace_file.get('metadata')['key']

                            for idx in range(0, trace_list.shape[0], shard_size):
                                print(idx, trace_idx, shard_size)
                                self.traces[trace_idx:trace_idx + shard_size] = trace_list[idx:idx + shard_size]
                                self.labels[trace_idx:trace_idx + shard_size] = label_list[idx:idx + shard_size]
                                trace_idx += shard_size

                if file.endswith('.trs'):
                    with trsfile.open(file, 'r') as trace_file:
                        trace_list = np.array([trace.samples for trace in trace_file])
                        key = np.frombuffer(bytes.fromhex('cafebabedeadbeef0001020304050607'), dtype=np.uint8)

                        for idx in range(0, len(trace_list), shard_size):
                            print(idx, trace_idx, shard_size)
                            self.traces[trace_idx:trace_idx + shard_size] = trace_list[idx:idx + shard_size]
                            self.labels[trace_idx:trace_idx + shard_size] = key
                            trace_idx += shard_size



    def get_file_trace_lengths(self, data_dir):
        file_lengths = {}

        for file in Path(data_dir).rglob("*"):  # files:

            filepath = str(file.absolute())
            if filepath.endswith('.h5'):
                if not filepath.endswith('ASCAD_ATM-AESv1_variable_key.h5'):
                    with h5.File(file, 'r') as trace:
                        file_lengths[filepath] = self.set_file_shape(trace)

            if filepath.endswith('.trs'):
                with trsfile.open(file, 'r') as trace:
                    file_lengths[filepath] = self.set_file_shape(trace)

        self.total_traces = sum({x for x, y in file_lengths.values()})
        print(file_lengths)
        print(self.total_traces)
        return file_lengths

    def set_file_shape(self, trace):
        length, width = 0, 0

        if isinstance(trace, trsfile.TraceSet):
            length, width = (len(trace), len(trace[0]))

        if isinstance(trace, h5py.Group):

            length, width = self.get_hdf5_trace_length(trace)

            if (length, width) == (None, None):
                file_len = [self.get_hdf5_trace_length(trace[key]) for key in trace]

                width = max([y for x, y in file_len])
                length = sum([x for x, y in file_len])

        return length, width

    @staticmethod
    def get_hdf5_trace_length(dataset):
        if 'traces' in dataset.keys():
            return dataset['traces'].shape[0], dataset['traces'].shape[1]

        return None, None

    def __repr__(self):
        return f"TraceFileDataset({self.total_traces})"
