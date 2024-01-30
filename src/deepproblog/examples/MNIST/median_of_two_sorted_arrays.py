from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, MNISTOperator
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

from argparse import ArgumentParser

def execute_median_of_two_sorted_arrays(nums1, nums2):
    def kth(a, b, k):
        if not a:
            return b[k]
        if not b:
            return a[k]
        ia, ib = len(a) // 2, len(b) // 2
        ma, mb = a[ia], b[ib]

        if ia + ib < k:
            if ma > mb:
                return kth(a, b[ib + 1:], k - ib - 1)
            else:
                return kth(a[ia + 1:], b, k - ia - 1)
        else:
            if ma > mb:
                return kth(a[:ia], b, k)
            else:
                return kth(a, b[:ib], k)
    l = len(nums1) + len(nums2)
    if l % 2 == 1:
        return kth(nums1, nums2, l // 2)
    else:
        return (kth(nums1, nums2, l // 2) + kth(nums1, nums2, l // 2 - 1)) / 2.

def median_of_two_sorted_arrays(n: int, dataset: str, seed=None):
    return MNISTOperator(
        dataset_name=dataset,
        function_name="get_median_two_sorted",
        operator=lambda x: execute_median_of_two_sorted_arrays(x[:3], x[3:]),
        size=n,
        arity=6,
        seed=seed,
    )

parser = ArgumentParser("median_of_two_sorted_arrays")
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

method = "exact"
N = 1

name = "median_of_two_sorted_arrays{}_{}_{}".format(method, N, args.seed)

train_set = median_of_two_sorted_arrays(N, "train")
test_set = median_of_two_sorted_arrays(N, "test")

train_set = train_set.subset(0, 10000)
test_set = test_set.subset(0, 1000)

network = MNIST_Net()

pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("models/median_of_two_sorted_arrays.pl", [net])
if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "geometric_mean":
    model.set_engine(
        ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    )

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
