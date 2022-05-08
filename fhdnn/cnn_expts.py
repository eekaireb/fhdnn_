import torchvision.transforms as transforms
from cifarDataModule import CifarData
from mnistDataModule import MnistData
from models import *
from utils import *
from sampling import *
from federated_cnn import *
import matplotlib
import pytorch_lightning as pl
matplotlib.use('Agg')

parser = generate_parser()
args = parser.parse_args()
pl.seed_everything(20)

print("---------------------------------------------------------------------------------------")
print(args)
print("---------------------------------------------------------------------------------------")

if args.dataset == 'cifar10':
    data = CifarData(batch_size=512, workers=8, expt = 'cnn', mode = 10)
    test_data = data.valset()
    if args.simclr:
        model = Linear_simclr()
    else:
        model = eval('ResNet' + str(args.resnet) + '()')

    if args.iid:
        splits = cifar_iid(args.nclients, expt = 'cnn', mode = 10)
    else:
        splits = cifar_niid(args.nclients, expt = 'cnn', mode = 10)
elif args.dataset == 'mnist':
    data = MnistData(batch_size=512, workers=8)
    data.val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.03081)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    test_data = data.valset()

    if args.simclr:
        model = Linear_simclr()
    else:
        model = CNNMnist()

    if args.iid:
        splits = mnist_iid(args.nclients)
    else:
        splits = mnist_niid(args.nclients)
elif args.dataset == 'celeba':
    model = eval('ResNet' + str(args.resnet) + '()')
    test_data = jtestDataset(
        '/home/bitwiz/codeden/data/leaf/data/celeba/data/test/all_data_iid_01_0_keep_5_test_9.json')
    if args.iid:
        splits = celeba_iid(
            '/home/bitwiz/codeden/data/leaf/data/celeba/data/train/all_data_iid_01_0_keep_5_train_9.json')
    else:
        raise NotImplementedError
elif args.dataset == 'fashionmnist':
    data = FashionMnistData(batch_size=512, workers=8)

    data.train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    data.val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    test_data = data.valset()

    if args.iid:
        splits = fashionmnist_iid(args.nclients)
    else:
        splits = fashionmnist_niid(args.nclients)
else:
    print("wrong option for dataset. Please pick from cifar10, mnist, celeba")
    raise NotImplemented



print("=> Initialized data splits")
#print(model.parameters)

federated = federated_framework(model, splits, test_data, **vars(args))
history = federated.train_clients()
torch.save(history, './logs/expt_cnn_{}_{}_E-{}_B-{}_-C-{}-nclient-{}_lr-{}-ploss-{}-iid-{}.txt'.format(
    args.dataset, args.resnet, args.local_epochs, args.local_batch_size, args.fraction,
    args.nclients, args.lr, args.ploss, args.iid
))
