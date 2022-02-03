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
parser.add_argument("--results_path", type=str, default='checkpoints/magnifications/longer-training/base-strategy',
                    help="path to the folder where we save checkpoints")
parser.add_argument('--early_stopping', action='store_true')
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

from avalanche.benchmarks.generators import dataset_benchmark, benchmark_with_validation_stream
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.logging import WandBLogger
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.benchmarks.utils import AvalancheTensorDataset
from model import UNET
from torch.nn import BCELoss
from torch.optim import  Adam
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDatasetType
import time
from torchvision.transforms import  CenterCrop
from avalanche.training.strategies import BaseStrategy



import operator
import warnings
from copy import deepcopy

from avalanche.training.plugins import StrategyPlugin


# source: https://github.com/ContinualAI/avalanche/blob/e91183e6fee8ccfdcde79f674e0c427f72bb9e8c/avalanche/training/plugins/early_stopping.py
# I have added printing statements to `before_training_epoch` function to track the issue
class EarlyStoppingPlugin(StrategyPlugin):
    """Early stopping and model checkpoint plugin.
    The plugin checks a metric and stops the training loop when the accuracy
    on the metric stopped progressing for `patience` epochs.
    After training, the best model's checkpoint is loaded.
    .. warning::
        The plugin checks the metric value, which is updated by the strategy
        during the evaluation. This means that you must ensure that the
        evaluation is called frequently enough during the training loop.
        For example, if you set `patience=1`, you must also set `eval_every=1`
        in the `BaseStrategy`, otherwise the metric won't be updated after
        every epoch/iteration. Similarly, `peval_mode` must have the same
        value.
    """

    def __init__(
        self,
        patience: int,
        val_stream_name: str,
        metric_name: str = "Top1_Acc_Stream",
        mode: str = "max",
        peval_mode: str = "epoch",
    ):
        """Init.
        :param patience: Number of epochs to wait before stopping the training.
        :param val_stream_name: Name of the validation stream to search in the
        metrics. The corresponding stream will be used to keep track of the
        evolution of the performance of a model.
        :param metric_name: The name of the metric to watch as it will be
        reported in the evaluator.
        :param mode: Must be "max" or "min". max (resp. min) means that the
        given metric should me maximized (resp. minimized).
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            early stopping should happen after `patience`
            epochs or iterations (Default='epoch').
        """
        super().__init__()
        self.val_stream_name = val_stream_name
        self.patience = patience

        assert peval_mode in {"epoch", "iteration"}
        self.peval_mode = peval_mode

        self.metric_name = metric_name
        self.metric_key = (
            f"{self.metric_name}/eval_phase/" f"{self.val_stream_name}"
        )
        print(self.metric_key)
        if mode not in ("max", "min"):
            raise ValueError(f'Mode must be "max" or "min", got {mode}.')
        self.operator = operator.gt if mode == "max" else operator.lt

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_step = None

    def before_training(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None

    def before_training_iteration(self, strategy, **kwargs):
        if self.peval_mode == "iteration":
            self._update_best(strategy)
            curr_step = self._get_strategy_counter(strategy)
            if curr_step - self.best_step >= self.patience:
                strategy.model.load_state_dict(self.best_state)
                strategy.stop_training()

    def before_training_epoch(self, strategy, **kwargs):
        if self.peval_mode == "epoch":
            self._update_best(strategy)
            curr_step = self._get_strategy_counter(strategy)
            print(f'Metric name: {self.metric_name}')
            print(f'Metric key: {self.metric_key}')
            print(f'Current step: {curr_step}, current val metric: {strategy.evaluator.get_last_metrics().get(self.metric_key)}')
            print(f'Best step: {self.best_step}, best metric: {self.best_val}')
            print(f'Patience: {self.patience}')
            if curr_step - self.best_step >= self.patience:
                strategy.model.load_state_dict(self.best_state)
                strategy.stop_training()

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        val_acc = res.get(self.metric_key)
        if self.best_val is None:
            warnings.warn(
                f"Metric {self.metric_name} used by the EarlyStopping plugin "
                f"is not computed yet. EarlyStopping will not be triggered."
            )
        if self.best_val is None or self.operator(val_acc, self.best_val):
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc
            self.best_step = self._get_strategy_counter(strategy)

    def _get_strategy_counter(self, strategy):
        if self.peval_mode == "epoch":
            return strategy.clock.train_exp_epochs
        elif self.peval_mode == "iteration":
            return strategy.clock.train_exp_iterations
        else:
            raise ValueError("Invalid `peval_mode`:", self.peval_mode)


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
benchmark = dataset_benchmark(
    [exp_1_train_dataset, exp_2_train_dataset, exp_3_train_dataset],
    [exp_1_test_dataset, exp_2_test_dataset, exp_3_test_dataset],
    dataset_type=AvalancheDatasetType.SEGMENTATION
)

# # benchmark, where the validation stream, created from training stream, has been added
benchmark = benchmark_with_validation_stream(benchmark,  input_stream = 'train',
                                     output_stream='valid', validation_size=0.25)


# defining training procedure
# loggers
loggers = []
loggers.append(InteractiveLogger())
loggers.append(WandBLogger(project_name=args.wandb_project_name,
                           run_name=args.wandb_run_name,
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
plugins = []
if args.early_stopping:
    early_stopping = EarlyStoppingPlugin(patience=2, metric_name = 'Loss_Exp', val_stream_name='valid_stream', mode='min',
                                         peval_mode='epoch')
    plugins.append(early_stopping)

# defines how the training is performed
strategy = BaseStrategy(
    model,
    optimizer,
    criterion,
    train_batch_size,
    epochs,
    eval_batch_size,
    DEVICE,
    plugins=plugins,
    evaluator=eval_plugin,
    eval_every=1,
    peval_mode='epoch'
)


# training
print('Starting experiment...')
results = []
for i, experience in enumerate(benchmark.train_stream):
    print("Start of experience: ", experience.current_experience)
    res = strategy.train(experience, eval_streams = [benchmark.valid_stream[i]])
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
