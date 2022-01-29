import argparse
#################
### Arguments ###
#################
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=None, choices=["0", "1", "2", "3", "4", "5"])
parser.add_argument('--training_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=30, help="number of epochs")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--path_to_tensors', type=str, default='data/magnifications/tensors/',
                    help="path to dataset that contains train/test datasets saved as tensors")
parser.add_argument('--wandb_project_name', type=str, default='continual-segmentation')
parser.add_argument('--wandb_run_name', type=str, default='base-strategy-early-stopping')
parser.add_argument('--wandb_checkpoint_path', type=str,
                    default='checkpoints/magnifications/longer-training/base-strategy/wandb')
parser.add_argument("--results_path", type=str, default='checkpoints/magnifications/longer-training/base-strategy',
                    help="path to the folder where we save checkpoints")
args = parser.parse_args()
print('arguments parsed: ')
for arg in vars(args):
    print(' {} {}'.format(arg, getattr(args, arg) or ''))


import os
import torch
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f'reserved gpu {os.environ.get("CUDA_VISIBLE_DEVICES")}')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from avalanche.benchmarks.generators import tensors_benchmark, dataset_benchmark, benchmark_with_validation_stream
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.logging import WandBLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.benchmarks.utils import AvalancheTensorDataset
from avalanche.logging import InteractiveLogger
from avalanche.benchmarks.utils import AvalancheTensorDataset
from model import UNET
from avalanche.training.strategies import Naive
from torch.nn import BCELoss
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset
import time
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from uniseg.utils import imgutils
from torchvision.transforms import Compose, ToTensor, CenterCrop
from tqdm import tqdm
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins import ReplayPlugin, EWCPlugin


# loading data and creating the benchmark
start = time.time()
experience_1_train_x = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_1_train_x.pt'))
experience_1_train_y = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_1_train_y.pt'))
experience_1_test_x = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_1_test_x.pt'))
experience_1_test_y = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_1_test_y.pt'))

experience_2_train_x = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_2_train_x.pt'))
experience_2_train_y = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_2_train_y.pt'))
experience_2_test_x = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_2_test_x.pt'))
experience_2_test_y = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_2_test_y.pt'))

experience_3_train_x = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_3_train_x.pt'))
experience_3_train_y = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_3_train_y.pt'))
experience_3_test_x = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_3_test_x.pt'))
experience_3_test_y = torch.load(os.path.join(args.path_to_tensors, 'tensor_experience_3_test_y.pt'))
end = time.time()
print(f'It took {round((end - start) / 60, 2)} minutes to load all tensors.')


# building Avalanche dataset from tensors
exp_1_train_dataset = AvalancheTensorDataset(experience_1_train_x, experience_1_train_y,
                                             transform=CenterCrop(224), # pattern transformation
                                             target_transform=CenterCrop(224)) # mask transformation
exp_1_test_dataset = AvalancheTensorDataset(experience_1_test_x, experience_1_test_y)

exp_2_train_dataset = AvalancheTensorDataset(experience_2_train_x, experience_2_train_y,
                                             transform=CenterCrop(224), # pattern transformation
                                             target_transform=CenterCrop(224))
exp_2_test_dataset = AvalancheTensorDataset(experience_2_test_x, experience_2_test_y,
                                             transform=CenterCrop(224), # pattern transformation
                                             target_transform=CenterCrop(224))

exp_3_train_dataset = AvalancheTensorDataset(experience_3_train_x, experience_3_train_y,
                                             transform=CenterCrop(224), # pattern transformation
                                             target_transform=CenterCrop(224))
exp_3_test_dataset = AvalancheTensorDataset(experience_3_test_x, experience_3_test_y,
                                             transform=CenterCrop(224), # pattern transformation
                                             target_transform=CenterCrop(224))


# generates stream of data
scenario_from_datasets = dataset_benchmark(
    [exp_1_train_dataset, exp_2_train_dataset, exp_3_train_dataset],
    [exp_1_test_dataset, exp_2_test_dataset, exp_3_test_dataset],
)

# benchmark, where the validation stream, created from training stream, has been added
benchmark = benchmark_with_validation_stream(scenario_from_datasets,  input_stream = 'train',
                                     output_stream='valid', validation_size=0.25)


# defining training procedure
# loggers
loggers = []
loggers.append(InteractiveLogger())
loggers.append(WandBLogger(project_name=args.wandb_project_name,
                           run_name=args.wandb_run_name,
                           path=args.wandb_checkpoint_path
                          ))

# evaluation logging
eval_plugin = EvaluationPlugin(
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    benchmark=benchmark,
    loggers=loggers
)

model = UNET(in_channels=1, out_channels=1).to(DEVICE)
optimizer = Adam(model.parameters(), lr=args.lr)
criterion = BCELoss()
train_batch_size = args.training_batch_size
epochs = args.epochs
eval_batch_size = args.eval_batch_size

# plugins
early_stopping = EarlyStoppingPlugin(patience=10, metric_name = 'Loss_Exp', val_stream_name='valid', mode='min',
                                     peval_mode='epoch')

# defines how the training is performed
strategy = BaseStrategy(
    model,
    optimizer,
    criterion,
    train_batch_size,
    epochs,
    eval_batch_size,
    DEVICE,
    plugins=[early_stopping],
    evaluator=eval_plugin,
    eval_every=10,
    peval_mode='epoch'
)


# training
print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    res = strategy.train(experience)
    print('Training completed')

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    print("=> Saving checkpoint")
    torch.save(checkpoint, os.path.join(args.results_path, str(experience.current_experience) + ".pth.tar"))


    print('Evaluation on test set:')
    results.append(strategy.eval(benchmark.test_stream))