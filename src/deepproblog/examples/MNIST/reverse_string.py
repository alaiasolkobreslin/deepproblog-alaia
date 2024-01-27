from json import dumps

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from deepproblog.dataset import DataLoader, Dataset
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.embeddings import List as DPLList

from typing import Callable, List, Iterable, Tuple
from torch.utils.data import Dataset as TorchDataset
import random
import itertools
import json
from problog.logic import Term, list2term, Constant
from deepproblog.dataset import Dataset
from deepproblog.query import Query

from argparse import ArgumentParser

_DATA_ROOT = Path(__file__).parent

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

emnist_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 
                     'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
                     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

class EMNISTOperator(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l1, l2 = self.data[index]
        label = self._get_label(index)
        l1 = [self.dataset[x][0] for x in l1]
        l2 = [self.dataset[x][0] for x in l2]
        return l1, l2, label

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(EMNISTOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        emnist_indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(emnist_indices)
        dataset_iter = iter(emnist_indices)
        # Build list of examples (emnist indices)
        self.data = []
        try:
            while dataset_iter:
                self.data.append(
                    [
                        [next(dataset_iter) for _ in range(self.size)]
                        for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return f"{tuple(itertools.chain(*self.data[i]))}\t{self._get_label(i)}"

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def to_query(self, i: int) -> Query:
        """Generate queries"""
        mnist_indices = self.data[i]
        expected_result = self._get_label(i)
        # print(f"emnist indices: {mnist_indices}")
        # print(f"expected result: {expected_result}")

        # Build substitution dictionary for the arguments
        subs = dict()
        var_names = []
        for i in range(self.arity):
            inner_vars = []
            for j in range(self.size):
                t = Term(f"p{i}_{j}")
                subs[t] = Term(
                    "tensor",
                    Term(
                        self.dataset_name,
                        Constant(mnist_indices[i][j]),
                    ),
                )
                inner_vars.append(t)
            var_names.append(inner_vars)

        # Build query
        if self.size == 1:
            return Query(
                Term(
                    self.function_name,
                    *(e[0] for e in var_names),
                    Constant(tuple(expected_result)),
                ),
                subs,
            )
        else:
            return Query(
                Term(
                    self.function_name,
                    *(list2term(e) for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )
        
    def chars_to_str(self, chars: Iterable[int]) -> int:
        return ''.join(emnist_mapping[i] for i in chars)

    def _get_label(self, i: int):
        mnist_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            self.chars_to_str(self.dataset[j][1] for j in i) for i in mnist_indices
        ]
        # print(ground_truth)
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def __len__(self):
        return len(self.data)

char_to_int = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
    'a': 36, 'b': 37, 'd': 38, 'e': 39, 'f': 40, 'g': 41, 'h': 42, 'n': 43, 'q': 44, 'r': 45, 't': 46
}

def reverse_string(n: int, dataset: str, seed=None):
    return EMNISTOperator(
        dataset_name=dataset,
        function_name="reverse_string",
        #operator = lambda char_list: sum(ord(char) * (1000 ** idx) for idx, char in enumerate(char_list)),
        #operator= lambda char_list: sum(char_to_int[char] * (47 ** idx) for idx, char in enumerate(reversed(char_list))),
        # operator=lambda x: ''.join([i for i in reversed(x)]),
        operator=lambda x: tuple(i for i in reversed(x)),
        size=n,
        arity=3,
        seed=seed,
    )

datasets = {
    "train": torchvision.datasets.EMNIST(
        split='balanced',root=str(_DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.EMNIST(
        split='balanced',root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}

class EMNIST_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]

EMNIST_train = EMNIST_Images("train")
EMNIST_test = EMNIST_Images("test")

class EMNIST(Dataset):
    def __len__(self):
        return len(self.data)

    def to_query(self, i):
        l = Constant(self.data[i][1])
        return Query(
            Term("char", Term("tensor", Term(self.dataset, Term("a"))), l),
            substitution={Term("a"): Constant(i)},
        )

    def __init__(self, dataset):
        self.dataset = dataset
        self.data = datasets[dataset]

class EMNIST_Net(nn.Module):
    def __init__(self):
        super(EMNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 47)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

parser = ArgumentParser("reverse_string")
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

method = "exact"
N = 1

name = "reverse_string{}_{}_{}".format(method, N, args.seed)

train_set = reverse_string(N, "train")
test_set = reverse_string(N, "test")

train_set = train_set.subset(0, 1)
test_set = test_set.subset(0, 1)

network = EMNIST_Net()

pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "emnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("/workspace/deepproblog-alaia/src/deepproblog/examples/MNIST/models/reverse_string.pl", [net])

model.set_engine(ExactEngine(model), cache=False)


model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

loader = DataLoader(train_set, 2, False, seed=args.seed)
train = train_model(model, loader, 10, log_iter=100, profile=0)
model.save_state("snapshot/" + name + ".pth")
train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
)
train.logger.write_to_file("log/" + name)
