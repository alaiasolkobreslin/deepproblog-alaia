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

def execute_largest_rectangle(heights):
    # problem 84: https://leetcode.com/problems/largest-rectangle-in-histogram/
    max_area, st = 0, []
    for idx, x in enumerate(heights):
        if len(st) == 0:
            st.append(idx)
        elif x >= heights[st[-1]]:
            st.append(idx)
        else:
            while st and heights[st[-1]] > x:
                min_height = heights[st.pop()]
                max_area = max(max_area, min_height*(idx-1 -
                               st[-1])) if st else max(max_area, min_height*idx)
            st.append(idx)
    while st:
        min_height = heights[st.pop()]
        max_area = max(max_area, min_height*(len(heights)-1 -
                       st[-1])) if st else max(max_area, min_height*len(heights))
    return max_area

def largest_rectangle(n: int, dataset: str, seed=None):
    return MNISTOperator(
        dataset_name=dataset,
        function_name="main",
        operator=lambda x: 1 if execute_largest_rectangle(x) else 0,
        size=n,
        arity=6,
        seed=seed,
    )

parser = ArgumentParser("largest_rectangle")
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

method = "exact"
N = 1

name = "largest_rectangle{}_{}_{}".format(method, N, args.seed)

train_set = largest_rectangle(N, "train")
test_set = largest_rectangle(N, "test")

train_set = train_set.subset(0, 10000)
test_set = test_set.subset(0, 1000)

network = MNIST_Net()

pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("models/largest_rectangle.pl", [net])
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
