from federated_hd import *
from sampling import *
from utils import *
from models import *
from mnistDataModule import MnistData
from cifarDataModule import CifarData
from fashionmnistDataModule import FashionMnistData
from torch_gists.models.resnet import *

def run():
    parser = generate_parser()
    args = parser.parse_args()

    print("---------------------------------------------------------------------------------------")
    print(args)
    print("---------------------------------------------------------------------------------------")

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp12_87_52/epoch%3D960.ckpt'
    #weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
    net = SimCLR.load_from_checkpoint(
        weight_path, strict=False, dataset='imagenet', maxpool1=False, first_conv=False, input_height=28)
    net.freeze()

    encoder = nn.Sequential(
        net,
        hd.RandomProjectionEncoder(2048, args.D)
    )

    classifier = hd.HDClassifier(10, args.D)

    model = (encoder, classifier)

    print("=> Initialized model")

    if args.dataset == 'cifar10':
        data = CifarData(batch_size=512, workers=8, expt = 'hd', mode = 10)
        test_data = data.valset()

        if args.iid:
            splits = cifar_iid(args.nclients, expt = 'hd', mode = 10)
        else:
            splits = cifar_niid(args.nclients, expt = 'hd', mode = 10)
    elif args.dataset == 'mnist':
        data = MnistData(batch_size=512, workers=8, expt = 'hd')

        data.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        data.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        test_data = data.valset()
        flip_data = data.flipset()

        if args.iid:
            splits = mnist_iid(args.nclients)
        else:
            splits = mnist_niid(args.nclients)
    elif args.dataset == 'celeba':
        classifier = hd.hd_classifier(2, args.D)
        test_data = jtestDataset(
            './codeden/data/leaf/data/celeba/data/test/all_data_iid_01_0_keep_5_test_9.json')
        if args.iid:
            splits = celeba_iid(
                './codeden/data/leaf/data/celeba/data/train/all_data_iid_01_0_keep_5_train_9.json')
        else:
            raise NotImplementedError
    elif args.dataset == 'femnist':
        classifier = hd.hd_classifier(62, args.D)
        if args.iid:
            splits = femnist(
                './codeden/data/leaf/data/femnist/data/train')
            test_data = femnist_test(
                './codeden/data/leaf/data/femnist/data/test')
        else:
            raise NotImplementedError
    elif args.dataset == 'fashionmnist':
        data = FashionMnistData(batch_size=512, workers=8, expt = 'hd')

        data.train_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        data.val_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        test_data = data.valset()
        flip_data = data.flipset()

        if args.iid:
            splits = fashionmnist_iid(args.nclients)
        else:
            splits = fashionmnist_niid(args.nclients)
    else:
        print("wrong option for dataset. Please pick from cifar10, mnist, celeba")
        raise NotImplemented

    print("=> Initialized data splits")

    federated = federated_framework(model, splits, test_data, flip_data, **vars(args))
    history = federated.train()
    accs = history['acc']

    print("=================> Mac accuracy {}".format(max(accs)))

    torch.save(history, './logs/ploss_expt_fhdnn_{}_{}_E-{}_B-{}_C-{}-nclient-{}-C-{}_lr-{}-iid-{}.pt'.format(
        args.dataset, args.resnet, args.local_epochs, args.local_batch_size, args.fraction,
        args.nclients, args.fraction, args.lr, args.iid
    ))
    

if __name__=='__main__':
    run()
