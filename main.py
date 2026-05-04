import os

import torch
import time
import logging

from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, SubsetRandomSampler, random_split

from src.TraceFileDataset import TraceFileDataset
from src.model.KeyExtractor import KeyExtractor
from src.utils import pretty_time_delta, setup_logger
from src.logFormatter import logFormatter
from torch.utils.data import DataLoader
from itertools import chain
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':

    setup_logger()
    device = (torch.device('cpu'), torch.device('cuda:0'))[torch.cuda.is_available()]
    dataset = TraceFileDataset(data_dir='data')
    print(device)

    r1 = range(0, 200000)
    r2 = range(300000, dataset.total_traces)

    r3 = list(chain(r1, r2))

    subset = Subset(dataset, r2)
    dataset.batch_size = 32

    model = KeyExtractor(
        key_size=128,
        pad_value=dataset.pad_value,
        batch_size=dataset.batch_size,
        device=device
    )

    model = torch.compile(model, mode='reduce-overhead')

    optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(0.00001), betas=(0.9, 0.999))
    model.zero_gradients()
    criterion = CrossEntropyLoss().to(model.device, non_blocking=True)

    epochs = 20
    training_samples = dataset.batch_size * 200
    early_stop = 3
    total_time = time.time()
    test_performance = []

    for epoch in range(epochs):

        epoch_start_time = time.time()
        trace_count = 0

        training, validation, test = random_split(subset, [0.8, 0.1, 0.1])

        train_loader = DataLoader(
            training,
            batch_size=dataset.batch_size,
            sampler=SubsetRandomSampler(torch.randint(0, len(training), (training_samples,))),
            collate_fn=dataset.trace_collate_fn,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        val_loader = DataLoader(
            validation,
            batch_size=dataset.batch_size,
            sampler=SubsetRandomSampler(torch.randint(0, len(validation), (dataset.batch_size * 4,))),
            collate_fn=dataset.trace_collate_fn,
            num_workers=int(os.cpu_count() / 6),
            pin_memory=True
        )

        test_loader = DataLoader(
            test,
            batch_size=dataset.batch_size,
            sampler=SubsetRandomSampler(torch.randint(0, len(test), (dataset.batch_size * 30,))),
            collate_fn=dataset.trace_collate_fn,
            num_workers=int(os.cpu_count() / 6),
            pin_memory=True
        )

        model.train()

        for batch_idx, (batch, label) in enumerate(train_loader):

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):

                batch_start_time = time.time()
                trace_count += batch.size(0)

                (loss,
                 accuracy,
                 precision,
                 recall,
                 GE) = model.get_model_metrics(batch, label, criterion)

                loss.backward()
                optimizer.step()
                model.zero_gradients()

                if batch_idx % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        for v_batch, v_label in val_loader:
                            (v_loss,
                             v_accuracy,
                             v_precision,
                             v_recall,
                             v_GE) = model.get_model_metrics(v_batch, v_label, criterion)

                    model.train()

                logging.info(f'{logFormatter.gold}Epoch {epoch + 1}: [{pretty_time_delta(time.time() - epoch_start_time)}] {logFormatter.green}TRAIN [{loss.item():.5f}] [accuracy, precision, recall, GE]: [{accuracy.item():.3f}, {precision.item():.3f}, {recall.item():.3f}, {GE.item():.3f}] {logFormatter.orange}VALIDATION [{v_loss.item():.5f}] [{v_accuracy.item():.3f}, {v_precision.item():.3f}, {v_recall.item():.3f}, {v_GE.item():.3f}] {logFormatter.purple}({trace_count}/{training_samples})')

        logging.info(f'{logFormatter.gold}Epoch {epoch + 1} completed in {pretty_time_delta(time.time() - epoch_start_time)}')

        model.eval()
        with torch.no_grad():
            for t_batch, t_label in test_loader:
                (t_loss,
                 t_accuracy,
                 t_precision,
                 t_recall,
                 t_GE) = model.get_model_metrics(t_batch, t_label, criterion)

        test_performance.append(t_accuracy)
        logging.info(f'{logFormatter.blue}TEST [{t_loss.item():.5f}] [accuracy, precision, recall, GE]: [{t_accuracy.item():.3f}, {t_precision.item():.3f}, {t_recall.item():.3f}, {t_GE.item():.3f}]')

        if epoch > early_stop:
            check_early_stop = test_performance[(-1 * early_stop):]
            if t_accuracy <= min(check_early_stop):
                break

    logging.info(f'{logFormatter.gold}Total training time: {pretty_time_delta(time.time() - total_time)}')