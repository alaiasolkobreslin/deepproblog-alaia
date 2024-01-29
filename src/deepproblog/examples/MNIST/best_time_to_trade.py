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

def best_time_to_buy_and_sell_stock(prices):
    left = 0
    right = 1
    max_profit = 0
    while right < len(prices):
        currentProfit = prices[right] - prices[left]  # our current Profit
        if prices[left] < prices[right]:
            max_profit = max(currentProfit, max_profit)
        else:
            left = right
        right += 1
    return max_profit

def best_time_to_trade(n: int, dataset: str, seed=None):
    return MNISTOperator(
        dataset_name=dataset,
        function_name="main",
        operator=lambda x: best_time_to_buy_and_sell_stock(x),
        size=n,
        arity=8,
        seed=seed,
    )

parser = ArgumentParser("best_time_to_trade")
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

method = "exact"
N = 1

name = "best_time_to_trade{}_{}_{}".format(method, N, args.seed)

train_set = best_time_to_trade(N, "train")
test_set = best_time_to_trade(N, "test")

train_set = train_set.subset(0, 10000)
test_set = test_set.subset(0, 1000)

network = MNIST_Net()

pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("models/best_time_to_trade.pl", [net])
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
