import base64
import os

import torch
import time
import logging

from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, random_split

from src.TraceFileDataset import TraceFileDataset
from src.model.KeyExtractor import KeyExtractor
from src.utils import pretty_time_delta, calculate_metrics, setup_logger, calculate_guessing_entropy
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from torch.utils.data import DataLoader
from itertools import chain
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':

    setup_logger()

    dataset = TraceFileDataset(data_dir='data')

    r1 = range(0, 200000)
    r2 = range(300000, dataset.total_traces)

    training = Subset(dataset, r2)
    print(dataset)

    dataset.device = (torch.device('cpu'), torch.device('cuda:0'))[torch.cuda.is_available()]
    torch.zeros(1, device=dataset.device)  # Force small allocation to get memory info
    print(dataset.device)

    dataset.batch_size = 32

    model = KeyExtractor(
        key_size=128,
        pad_value=dataset.pad_value,
        batch_size=dataset.batch_size
    ).to(dataset.device, non_blocking=True)

    #model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(0.00001), betas=(0.9, 0.999))
    optimizer.zero_grad(set_to_none=True)
    criterion = CrossEntropyLoss().to(dataset.device, non_blocking=True)

    loader = DataLoader(training, batch_size=dataset.batch_size, shuffle=True, collate_fn=dataset.trace_collate_fn, num_workers=os.cpu_count())

    epochs = 2

    accuracy_fn = MulticlassAccuracy(num_classes=256, average='micro').to(dataset.device)
    precision_fn = MulticlassPrecision(num_classes=256, average='macro').to(dataset.device)  # 'macro' or 'weighted' often better for imbalanced SCA
    recall_fn = MulticlassRecall(num_classes=256, average='macro').to(dataset.device)

    for epoch in range(epochs):

        epoch_start_time = time.time()
        trace_count = 0

        for batch, y in loader:

            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):

                batch_start_time = time.time()
                batch = batch.to(dataset.device, non_blocking=True)
                y = y.to(dataset.device, non_blocking=True)
                trace_count += batch.size(0)

                y_hat = model(batch)

                y = y.view(-1)
                loss = criterion(y_hat, y)

                predictions = torch.argmax(y_hat, dim=-1).view(-1)
                GE = calculate_guessing_entropy(y_hat, y)

                accuracy = accuracy_fn(predictions, y)
                precision = precision_fn(predictions, y)
                recall = recall_fn(predictions, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                logging.info(
                    f'Batch time [{pretty_time_delta(time.time() - batch_start_time)}] loss: {loss.item():.5f} [accuracy, precision, recall, GE]: [{accuracy.item():.3f}, {precision.item():.3f}, {recall.item():.3f}, {GE.item():.3f}] ({trace_count}/{len(training)}) epoch time: [{pretty_time_delta(time.time() - epoch_start_time)}]')
        print(f'Epoch {epoch + 1} completed in {pretty_time_delta(time.time() - epoch_start_time)}')
