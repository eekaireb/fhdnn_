import copy
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
import numpy as np
from utils import *
from sampling import *
from cifarDataModule import CifarData
from mnistDataModule import MnistData
import wandb


def average_weights(weights):
    master = copy.deepcopy(weights[0])

    for key in master.keys():
        for w in weights[1:]:
            master[key] += w[key]
        master[key] = torch.div(master[key], len(weights))

    return master


class client(object):
    def __init__(self, model, data, idx, local_epochs, local_batch_size, device, lr=0.001, debug=False):
        self.model = copy.deepcopy(model)
        self.data = data
        self.id = idx
        self.E = local_epochs
        self.B = local_batch_size
        self.device = device
        self.lr = lr
        self.debug = debug

    def train(self):
        loader = DataLoader(self.data, shuffle=True,
                            batch_size=self.B, num_workers=8)
        self.model = self.model.to(self.device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), weight_decay=1e-3, lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(self.E):
            overall_acc = 0.0
            for idx, batch in enumerate(loader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device).type(torch.long)

                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

                _, y_hat = torch.max(y_hat, dim=1)
                acc = accuracy(y_hat, y)
                overall_acc += acc

            overall_acc /= (idx + 1)

            if self.debug:
                print("\t=> client: {} loss {} acc {}".format(
                    self.id, loss.item(), overall_acc))

        self.model.cpu()

        return self.model.state_dict()

    def update_model(self, new_model):
        self.model = copy.deepcopy(new_model)


class federated_framework(object):
    def __init__(
        self, model, data_splits, test_data, nclients=10, fraction=0.5, nrounds=20,
        local_epochs=5, local_batch_size=10, cpu=False, lr=0.001, debug=False,
        decay_lr=0.99, log=False, cuda=0, iid = True, **kwargs
    ):
        self.global_model = copy.deepcopy(model)
        self.nclients = nclients
        self.device = 'cpu' if cpu else 'cuda:' + str(cuda)
        self.clients = [client(model, data_splits[idx], idx, local_epochs,
                               local_batch_size, self.device, lr, debug) for idx in range(nclients)]
        print("=> Initialized clients")
        self.test_data = test_data
        self.C = fraction
        self.nrounds = nrounds
        self.scale = decay_lr
        self.log = log
        self.iid = iid

        if log:
            wandb.init(project='federated')

    def train_clients(self, target_acc=None):
        print("=> Begining training")
        results = []
        rounds = []

        for rnd in range(self.nrounds):
            num = int(self.C * self.nclients)
            chosen = np.random.choice(self.nclients, size=num, replace=False)

            client_models = [self.clients[client].train() for client in chosen]
            updated_weights = average_weights(client_models)
            self.global_model.load_state_dict(updated_weights)
            self.broadcast()
            test_acc = self.test()
            print("\t=> Run: {} accuracy: {}".format(rnd, test_acc))

            if self.log:
                wandb.log({
                    'test_acc': test_acc,
                    'round': rnd
                })

            results.append(test_acc)
            rounds.append(rnd)
            if not self.iid:
                self.decay_lr()

            history = {
                'rounds': rounds,
                'acc': results
            }

        if self.log:
            wandb.finish()

        return history

    def broadcast(self):
        for client in self.clients:
            client.update_model(self.global_model)

    def decay_lr(self):
        for client in self.clients:
            client.lr = client.lr * self.scale

    def test(self):
        loader = DataLoader(self.test_data, batch_size=32,
                            num_workers=8, shuffle=False)
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()

        overall_acc = 0.0
        for idx, batch in enumerate(loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.global_model(x)
            _, y_hat = torch.max(y_hat, dim=1)
            acc = accuracy(y_hat, y)
            overall_acc += acc
        overall_acc /= (idx + 1)

        self.global_model = self.global_model.cpu()

        return overall_acc


if __name__ == '__main__':
    from models import CNNMnist
    from torch_gists.models import ResNet18
    parser = generate_parser()
    args = parser.parse_args()
    print(args)
    data = MnistData(batch_size=512, workers=8)
    splits = mnist_iid(args.nclients)
    model = CNNMnist()

    wandb.init(project='federated')
    fed = federated_framework(model, splits, data.valset(), **vars(args))
    fed.train_clients()
    wandb.finish()
