import time

from avalanche.benchmarks import SplitCIFAR100, SplitCIFAR10  # , SplitCUB200  # , SplitTinyImageNet

from cl_algorithm_plugins.naive_replay import NaiveReplayPlugin
from datasets import SplitTinyImageNet, SplitCUB200

from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import BaseLogger

from models.arch_craft import arch_craft
from sampling_strategies.herding import HerdingSelectionStrategy
from sampling_strategies.rainbow_memory import RainbowMemorySelectionStrategy
from sampling_strategies.closest_to_canter import ClosestToCenterSelectionStrategy
from utils.text_logging import TextLogger
from avalanche.models import IncrementalClassifier, SlimResNet18

from models.resnet18 import resnet18
from cl_algorithm_plugins.er_ace import ER_ACE

from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from utils.storage_policy import ParametricBuffer, ClassBalancedBuffer
from cl_algorithm_plugins.strategy_wrappers import Replay, Naive
from sampling_strategies.teal import TEALExemplarsSelectionStrategy

from torch.optim import SGD


def init_dataset(args):
    if args.dataset == 'cifar10':
        dataset = SplitCIFAR10(n_experiences=args.num_experiences,
                               return_task_id=False,
                               shuffle=True, class_ids_from_zero_in_each_exp=False,
                               fixed_class_order=[i for i in range(10)] if args.seed is None else None,
                               seed=args.seed,
                               dataset_root='../data/avalanche/cifar10')
    elif args.dataset == 'cifar100':
        dataset = SplitCIFAR100(n_experiences=args.num_experiences,
                                return_task_id=False,
                                shuffle=True, class_ids_from_zero_in_each_exp=False,
                                fixed_class_order=[i for i in range(100)] if args.seed is None else None,
                                seed=args.seed)
    elif args.dataset == 'tinyimg':
        dataset = SplitTinyImageNet(n_experiences=args.num_experiences,
                                    return_task_id=False,
                                    shuffle=True, class_ids_from_zero_in_each_exp=False,
                                    fixed_class_order=[i for i in range(200)] if args.seed is None else None,
                                    seed=args.seed,
                                    dataset_root='../../tiny_imagenet/')
    elif args.dataset == 'cub200':
        dataset = SplitCUB200(n_experiences=args.num_experiences,
                              return_task_id=False,
                              shuffle=True, class_ids_from_zero_in_each_exp=False,
                              fixed_class_order=[i for i in range(200)] if args.seed is None else None,
                              seed=args.seed,
                              classes_first_batch=None,
                              dataset_root='../data/')
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
    return dataset


def init_incremental_model(inc_model_name, dataset, dataset_name):
    if inc_model_name == 'arch_craft':
        model = arch_craft(code=[10, 144, [3, 7, 8, 10, 10], [3, 8, 10, 10, 10]], dataset=dataset_name)  # for debug: [9, 12, [4, 6, 8, 8, 9], [1, 3, 5, 8, 9]]
        model.linear = IncrementalClassifier(model.linear.in_features,
                                             dataset.n_classes // dataset.n_experiences)
    elif inc_model_name == 'resnet18':
        model = resnet18(dataset.n_classes)
        model.linear = IncrementalClassifier(model.linear.in_features,
                                             dataset.n_classes // dataset.n_experiences)
    elif inc_model_name == 'slim_resnet18':
        model = SlimResNet18(nclasses=1)
        model.linear = IncrementalClassifier(model.linear.in_features,
                                             dataset.n_classes // dataset.n_experiences)
    else:
        raise ValueError(f'Unknown incremental model {inc_model_name}')

    return model


class ContinualLearningPipeline:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.dataset = init_dataset(args)
        self.inc_model = init_incremental_model(args.inc_model, self.dataset, args.dataset)
        self.buffer_size = args.buffer
        self.optimizer = self.get_optimizer()
        self.scheduler_plugin = self.get_scheduler_plugin()
        self.cl_strategy = self.init_cl_strategy()

    def get_optimizer(self):
        optimizer = SGD(self.inc_model.parameters(), momentum=self.args.momentum,
                        weight_decay=self.args.weight_decay, lr=self.args.lr)
        return optimizer

    def get_scheduler_plugin(self):
        scheduler = StepLR(self.optimizer, step_size=self.args.num_epochs // 3, gamma=0.3)
        scheduler_plugin = LRSchedulerPlugin(scheduler, step_granularity="epoch", first_exp_only=False)
        return scheduler_plugin

    def init_cl_strategy(self):
        if self.args.algorithm == 'er_ace':  # ER-ACE initialize with random exemplars
            storage_policy = ParametricBuffer(self.buffer_size, groupby='class')
        elif self.args.sel_strategy == 'herding':
            selection_strategy = HerdingSelectionStrategy()
            storage_policy = ParametricBuffer(self.buffer_size, groupby='class',
                                              selection_strategy=selection_strategy,
                                              )
        elif self.args.sel_strategy == 'teal':
            selection_strategy = TEALExemplarsSelectionStrategy(self.args, self.device)
            storage_policy = ParametricBuffer(self.buffer_size, groupby='class',
                                              selection_strategy=selection_strategy,
                                              )
        elif self.args.sel_strategy == 'rm':
            selection_strategy = RainbowMemorySelectionStrategy(self.args, self.device)
            storage_policy = ParametricBuffer(self.buffer_size, groupby='class',
                                              selection_strategy=selection_strategy,
                                              )
        elif self.args.sel_strategy == 'centered':
            selection_strategy = ClosestToCenterSelectionStrategy()
            storage_policy = ParametricBuffer(self.buffer_size, groupby='class',
                                              selection_strategy=selection_strategy,
                                              )
        else:
            storage_policy = ClassBalancedBuffer(self.buffer_size, adaptive_size=True)

        criterion = CrossEntropyLoss()
        loggers = [BaseLogger(), TextLogger()]
        evaluator = EvaluationPlugin(
            accuracy_metrics(epoch=True, trained_experience=True, experience=True, epoch_running=True),
            forgetting_metrics(experience=True),
            loggers=loggers,
        )

        if self.args.algorithm == 'er_ace':
            return ER_ACE(
                self.inc_model, self.optimizer, criterion, self.buffer_size, device=self.device,
                train_epochs=self.args.num_epochs, train_mb_size=self.args.batch_size,
                eval_mb_size=self.args.batch_size,
                batch_size_mem=self.args.batch_size,
                plugins=[self.scheduler_plugin], evaluator=evaluator,
                storage_policy=storage_policy,
                args=self.args
            )
        elif self.args.algorithm == 'er':
            return Replay(
                self.inc_model, self.optimizer, criterion, self.buffer_size, device=self.device,
                train_epochs=self.args.num_epochs, train_mb_size=self.args.batch_size,
                eval_mb_size=self.args.batch_size,
                plugins=[self.scheduler_plugin], evaluator=evaluator, storage_policy=storage_policy
            )
        elif self.args.algorithm == 'naive_replay':
            return Naive(model=self.inc_model, optimizer=self.optimizer,
                         plugins=[self.scheduler_plugin, NaiveReplayPlugin(storage_policy=storage_policy)],
                         evaluator=evaluator, device=self.device, train_epochs=self.args.num_epochs,
                         train_mb_size=self.args.batch_size, eval_mb_size=self.args.batch_size,
                         )
        elif self.args.algorithm == 'finetune':
            return Naive(
                self.inc_model, self.optimizer, criterion,
                device=self.device,
                train_epochs=self.args.num_epochs, train_mb_size=self.args.batch_size,
                eval_mb_size=self.args.batch_size,
                plugins=[self.scheduler_plugin], evaluator=evaluator,
            )
        else:
            raise ValueError(f'Unknown algorithm {self.args.algorithm}')

    def train(self):
        exp_acc_dict = {exp_id: [] for exp_id in range(self.dataset.n_experiences)}
        for t, experience in enumerate(self.dataset.train_stream):
            print(f"Start of experience: {experience.current_experience}\n"
                  f"Current Classes: {experience.classes_in_this_experience}\n"
                  f"Current Time: {time.strftime('%H:%M:%S', time.localtime())}")

            self.cl_strategy.train(
                experience,
                num_workers=1,
                drop_last=True,
            )

            exp_res = self.cl_strategy.eval(self.dataset.test_stream[: t + 1])
            for exp_id in range(t + 1):
                exp_acc_dict[exp_id].append(
                    exp_res[f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{str(exp_id).zfill(3)}'])

            print(f"\nExperience {experience.current_experience} Results:\n{exp_res}\n")

        return exp_acc_dict

