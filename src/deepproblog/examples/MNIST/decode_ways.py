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

def execute_decode_ways(s):
    if not s:
        return 0
    dp = [0 for _ in range(len(s) + 1)]
    dp[0] = 1
    dp[1] = 0 if s[0] == "0" else 1
    for i in range(2, len(s) + 1):
        if 0 < int(s[i-1:i]) <= 9:
            dp[i] += dp[i - 1]
        if 10 <= int(s[i-2:i]) <= 26:
            dp[i] += dp[i - 2]
    return dp[len(s)]

def decode_ways(n: int, dataset: str, seed=None):
    return MNISTOperator(
        dataset_name=dataset,
        function_name="decode_ways",
        operator=lambda x: execute_decode_ways(''.join(str(i) for i in x)),
        size=n,
        arity=3,
        seed=seed,
    )

parser = ArgumentParser("decode_ways")
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

method = "exact"
N = 1

name = "decode_ways{}_{}_{}".format(method, N, args.seed)

train_set = decode_ways(N, "train")
test_set = decode_ways(N, "test")

train_set = train_set.subset(0, 1000)
test_set = test_set.subset(0, 1000)

network = MNIST_Net()

pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("/workspace/deepproblog-alaia/src/deepproblog/examples/MNIST/models/decode_ways.pl", [net])
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
