import h5py
from torch.utils.data import IterableDataset, Dataset
from pathlib import Path
import torch
import os
import trsfile
import numpy as np
import h5py as h5


class TraceFileDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_dir='', output_file_name='trace_data', rebuild_dataset=False):
        super().__init__()

        self.device = None
        self.free_gpu_memory = 0

        self.data_path = data_dir
        self.output_file_name = output_file_name
        self.traces = None
        self.labels = None
        self.transform = transform
        self.total_traces = 0
        self.batch_size = 1
        self.pad_value = 255
        self.files = self.get_file_trace_lengths(data_dir)
        self.max_trace_file_length = max({y for x, y in self.files.values()})

        if not rebuild_dataset and (os.path.exists('processed/') and os.path.exists(f'processed/{self.output_file_name}.h5')):
            print('Database found ...')

        else:
            if rebuild_dataset:
                print('Rebuilding dataset ...')
            else:
                print('Database not found. Generating new h5 files ...')
            self.build_trace_dataset()

    def __len__(self):
        return self.total_traces

    def __getitem__(self, idx):

        trace_data = h5py.File(f'processed/{self.output_file_name}.h5', 'r', swmr=True)

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

        label_batch = torch.stack(label, dim=0)
        data_batch = data_batch.unsqueeze(1)

        return data_batch, label_batch

    def build_trace_dataset(self):

        Path(f'processed/{self.output_file_name}.h5').unlink(missing_ok=True)

        chunk_size = 100
        shard_size = 10000

        with h5py.File(f'processed/{self.output_file_name}.h5', 'w') as F:

            var_len = h5py.special_dtype(vlen=np.float32)
            self.traces = F.create_dataset('traces', shape=(self.total_traces,), chunks=(chunk_size,),
                                           compression="gzip", dtype=var_len)
            self.labels = F.create_dataset('labels', shape=(self.total_traces, 16), chunks=(chunk_size, 16),
                                           compression="gzip", dtype=np.uint8)
            start = 0

            for file, (file_length, file_width) in self.files.items():

                if file.endswith('.h5'):
                    with h5.File(file, 'r') as trace_file:
                        if trace_file.get('traces') is None:

                            for key in trace_file.keys():
                                trace_list = trace_file[key].get('traces')
                                label_list = trace_file[key].get('metadata')['key']

                                for idx in range(0, trace_list.shape[0], shard_size):
                                    end = min(idx + shard_size, trace_list.shape[0])
                                    #print(idx, end, end - idx)
                                    self.traces[(start + idx):(start + end)] = trace_list[idx:end]
                                    self.labels[(start + idx):(start + end)] = label_list[idx:end]

                        else:
                            trace_list = trace_file.get('traces')
                            label_list = trace_file.get('metadata')['key']

                            for idx in range(0, trace_list.shape[0], shard_size):
                                end = min(idx + shard_size, trace_list.shape[0])
                                # print(idx, end, end - idx)
                                self.traces[(start + idx):(start + end)] = trace_list[idx:end]
                                self.labels[(start + idx):(start + end)] = label_list[idx:end]

                if file.endswith('.trs'):
                    with trsfile.open(file, 'r') as trace_file:

                        trace_list = np.array([trace.samples for trace in trace_file])
                        key = np.frombuffer(bytes.fromhex('cafebabedeadbeef0001020304050607'), dtype=np.uint8)

                        for idx in range(0, len(trace_list), shard_size):
                            end = min(idx + shard_size, len(trace_list))
                            #print(idx, end, end-idx)
                            self.traces[(start + idx):(start + end)] = trace_list[idx:end]
                            self.labels[(start + idx):(start + end)] = key

                start += file_length

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
