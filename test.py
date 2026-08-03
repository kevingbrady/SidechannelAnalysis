from src.TraceFileDataset import TraceFileDataset
from src.model.KeyExtractor import KeyExtractor
from src.utils import pretty_time_delta, setup_logger
from src.logFormatter import logFormatter
import torch
import time
import logging
import h5py
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    setup_logger()
    dataset = TraceFileDataset(data_dir='raw', output_file_name='test_data')
    device = (torch.device('cpu'), torch.device('cuda:0'))[torch.cuda.is_available()]

    dataset.batch_size = 1

    model = KeyExtractor(
        key_size=128,
        pad_value=dataset.pad_value,
        batch_size=dataset.batch_size,
        device=device
    )

    model = torch.compile(model, mode='reduce-overhead')

    model.load_state_dict(torch.load('final_model/key_extractor_model.pth', weights_only=True))
    model.eval()

    logging.info(f'{logFormatter.gold}' + str(dataset))
    logging.info(f'{logFormatter.gold}' + str(device))

    sample, label = dataset.trace_collate_fn([dataset[2600000]])
    sample = sample.to(device)

    with torch.inference_mode():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            y_hat = model(sample)
            prediction = torch.argmax(y_hat, dim=-1)
            prediction_string = "".join(f'{b:02x}' for b in prediction)
            label_string = "".join(f'{b:02x}' for b in label.view(-1))
            print(prediction_string, "      ", label_string, prediction_string == label_string)







