import os
from collections import defaultdict
from cifarDataModule import CifarData
from mnistDataModule import MnistData
from fashionmnistDataModule import FashionMnistData
from torch.utils.data import TensorDataset, random_split, Dataset
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from multiprocessing import Pool
from PIL import Image
from functools import partial
import torchvision.transforms.functional as TF

################### Fashion MNIST ################### 

###### iid
def fashionmnist_iid(num_users):
	dm = FashionMnistData(batch_size = 512, workers = 16)
	dm.train_transform = transforms.Compose([
		transforms.ToTensor(),
		#transforms.Normalize((0.2859,), (0.3202,)),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
	])
	
	dm.val_transform = transforms.Compose([
		transforms.ToTensor(),
		#transforms.Normalize((0.2859,), (0.3202,)),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
	])


	dataset = dm.trainset()
	split_len = len(dataset) // num_users
	split_array = [split_len] * num_users
	splits = random_split(dataset, split_array)

	return splits

####### non iid
def fashionmnist_niid(num_users, separated_path = None):
	if separated_path == None:
		separated = {}
		dm = FashionMnistData(batch_size = 512, workers = 16)
		dm.train_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.2859,), (0.3202,)),
			transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
		])
		
		dm.val_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.2859,), (0.3202,)),
			transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
		])


		for label in range(10):
			separated[label] = torch.tensor([])
		
		for batch in tqdm(dm.train_dataloader()):
			x, y = batch
			for label in range(10):
				separated[label] = torch.cat([separated[label], x[y == label]], dim = 0)
		
		torch.save(separated, './codeden/data/mnist/fashionmnist_class_wise.pt')
	else:
		separated = torch.load(separated_path)
	
	nshards = 200
	shard_size = 300
	shards_per_user = nshards // num_users

	shards = []

	for label in range(10):
		for i in range(0, 6000, shard_size):
			batch_x = separated[label][i : i + shard_size]
			batch_y = [label] * batch_x.shape[0]
			batch_y = torch.tensor(batch_y)
			shards.append((batch_x, batch_y))
	
	random.shuffle(shards)

	splits = []

	for idx in tqdm(range(0, len(shards), shards_per_user)):
		batch_x = shards[idx][0]
		batch_y = shards[idx][1]

		for i in range(1, shards_per_user):
			batch_x = torch.cat([batch_x, shards[idx + i][0]], dim = 0)
			batch_y = torch.cat([batch_y, shards[idx + i][1]], dim = 0)
		
		ds = TensorDataset(batch_x, batch_y)
		splits.append(ds)
	
	length = [len(split) for split in splits]
	print(sum(length))

	return splits


################### MNIST ################### 

###### iid
def mnist_iid(num_users):
	dm = MnistData(batch_size = 512, workers = 16)
	dm.train_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
		transforms.RandomVerticalFlip(1) 
	])
	
	dm.val_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
		])

	dm.flip_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1) ), 
		transforms.RandomVerticalFlip(1) 
		])


	dataset = dm.trainset()
	split_len = len(dataset) // num_users
	split_array = [split_len] * num_users
	splits = random_split(dataset, split_array)

	return splits

def mnist_update(num_users, p):
	dm = MnistData(batch_size = 512, workers = 16)
	dm.train_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
		transforms.RandomVerticalFlip(p) 
	])
	
	dataset = dm.trainset()
	split_len = len(dataset) // num_users
	split_array = [split_len] * num_users
	splits = random_split(dataset, split_array)

	return splits



####### non iid
def mnist_niid(num_users, separated_path = None):
	if separated_path == None:
		separated = {}
		dm = MnistData(batch_size = 512, workers = 16)
		dm.train_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
			transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
		])
		
		dm.val_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),
			transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
		])


		for label in range(10):
			separated[label] = torch.tensor([])

		save = True
		for batch in tqdm(dm.train_dataloader()):
			x, y = batch
			if save:
				for i in range(len(x)):
					for j in range(len(x[i])):
						for k in range(len(x[i][j])):
							np.savetxt('x.csv', x[i][j][k], delimiter=',')
				save = False
			for label in range(10):
				separated[label] = torch.cat([separated[label], x[y == label]], dim = 0)
		        
		#for label in range(10):
			#print(len(separated[label]))
		print(len(separated[1]))
		torch.save(separated, './codeden/data/mnist/mnist_class_wise.pt')
	else:
		separated = torch.load(separated_path)
	
	nshards = 200
	#shard_size = 300
	shard_size = int(len(separated[1])/200)
	shards_per_user = nshards // num_users

	shards = []

	'''for label in range(10):
		for i in range(0, 6000, shard_size):
			batch_x = separated[label][i : i + shard_size]
			batch_y = [label] * batch_x.shape[0]
			batch_y = torch.tensor(batch_y)
			shards.append((batch_x, batch_y))
	'''
	for i in range(0, 6000, shard_size):
			batch_x = separated[1][i : i + shard_size]
			batch_y = [1] * batch_x.shape[0]
			batch_y = torch.tensor(batch_y)
			shards.append((batch_x, batch_y))
	
	random.shuffle(shards)

	splits = []

	for idx in tqdm(range(0, len(shards), shards_per_user)):
		batch_x = shards[idx][0]
		batch_y = shards[idx][1]

		for i in range(1, shards_per_user):
			batch_x = torch.cat([batch_x, shards[idx + i][0]], dim = 0)
			batch_y = torch.cat([batch_y, shards[idx + i][1]], dim = 0)
		
		ds = TensorDataset(batch_x, batch_y)
		splits.append(ds)
	
	length = [len(split) for split in splits]
	print(sum(length))

	return splits

#################### CIFAR10 #################### 

###### iid
def cifar_iid(num_users, expt, mode = 10):
	dm = CifarData(batch_size = 512, workers = 16, expt = expt, mode = mode)
	dataset = dm.trainset()
	split_len = len(dataset) // num_users
	split_array = [split_len] * num_users
	splits = random_split(dataset, split_array)

	return splits

###### non iid
def cifar_niid(num_users, separated_path = None, expt = 'cnn', mode = 10):
        dm = CifarData(batch_size = 512, workers = 16, expt = expt, mode = mode)
        separated = {}

        for i in range(mode):
                separated[i] = torch.tensor([])

        for batch in tqdm(dm.train_dataloader()):
                x, y = batch
                for label in range(mode):
                        separated[label] = torch.cat([separated[label], x[y == label]], dim = 0)

        nshards = 200
        shard_size = 250
        shards_per_user = nshards // num_users

        shards = []

        samps_per_class = 50000 // mode

        for label in range(mode):
                for i in range(0, samps_per_class, shard_size):
                        batch_x = separated[label][i : i + shard_size]
                        batch_y = [label] * batch_x.shape[0]
                        batch_y = torch.tensor(batch_y)
                        shards.append((batch_x, batch_y))

        random.shuffle(shards)

        splits = []

        for idx in tqdm(range(0, len(shards), shards_per_user)):
                batch_x = shards[idx][0]
                batch_y = shards[idx][1]

                for i in range(1, shards_per_user):
                        batch_x = torch.cat([batch_x, shards[idx + i][0]], dim = 0)
                        batch_y = torch.cat([batch_y, shards[idx + i][1]], dim = 0)

                ds = TensorDataset(batch_x, batch_y)
                splits.append(ds)

        length = [len(split) for split in splits]
        print(sum(length))

        return splits

########################## CELEBA federated ########################## 

def readimg(basepath, name):
	img = Image.open(basepath + name).convert('RGB')
	img = TF.to_tensor(np.array(img))
	img = TF.resize(img, 32)

	return img

class JsonDataset(Dataset):
	def __init__(self, jfile, idx, basepath):
		with open(jfile, 'r') as file:
			data = file.read()
		jdata = json.loads(data)
		self.images = jdata['user_data'][idx]['x']
		self.labels = jdata['user_data'][idx]['y']
		self.basepath = basepath

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		img_list = self.images[item]
		'''
		pool = Pool(32)
		img_func = partial(readimg, self.basepath)

		images = pool.map(img_func, [img_list])
		'''
		images = readimg(self.basepath, img_list)
		labels = self.labels[item]
		
		return images, torch.tensor(labels).type(torch.long)

class jtestDataset(Dataset):
	def __init__(self, jfile, basepath = './codeden/data/leaf/data/celeba/data/raw/img_align_celeba/'):
		with open(jfile, 'r') as file:
			data = file.read()
		jdata = json.loads(data)

		users = jdata['users']
		all_imgs_list = []
		all_labels = []

		for user in users:
			all_imgs_list.extend(jdata['user_data'][user]['x'])
			all_labels.extend(jdata['user_data'][user]['y'])

		self.images = all_imgs_list
		self.labels = all_labels
		self.basepath = basepath

	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		img_list = self.images[item]
		'''
		pool = Pool(32)
		img_func = partial(readimg, self.basepath)

		images = pool.map(img_func, [img_list])
		'''
		images = readimg(self.basepath, img_list)
		labels = int(self.labels[item])
		
		return images, torch.tensor(labels)


def celeba_iid(jfile, basepath = '/Users/emilyekaireb/Projects/codeden/data/leaf/data/celeba/data/raw/img_align_celeba/'):
	with open(jfile, 'r') as file:
		data = file.read()
	
	jdata = json.loads(data)
	users = jdata['users']

	splits = []

	for user in users:
		splits.append(JsonDataset(jfile, user, basepath))
	
	return splits

def read_dir(data_dir):
    '''
    code taken from TalwalkarLab/leaf and improvised
    '''
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]

    print("reading data")
    t = tqdm(range(len(files)))
    for idx, f in enumerate(files):
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata =json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])
        t.update()
    t.refresh()
    t.close()

    print("creating data splits")
    data_dict = defaultdict(lambda: None)
    t = tqdm(range(len(clients)))

    for idx, client in enumerate(clients):
        data_x = torch.tensor(data[client]['x'])
        data_x = data_x.reshape(data_x.shape[0], 28, 28).unsqueeze(dim = 1)
        data_x = data_x.repeat(1, 3, 1, 1)
        data_y = torch.tensor(data[client]['y']).type(torch.int)
        data_dict[client] = (data_x, data_y)
        t.update()
    t.refresh()
    t.close()

    return data_dict, clients

############################### FEMNIST ############################### 
def femnist(data_dir):
    data_dict, clients = read_dir(data_dir)
    splits = []
    
    for client in clients:
        data_x = data_dict[client][0]
        data_y = data_dict[client][1]
        dataset = TensorDataset(data_x, data_y)
        splits.append(dataset)
    
    return splits

def femnist_test(data_dir):
    data_dict, clients = read_dir(data_dir)
    data = torch.tensor([])
    labels = torch.tensor([])

    for client in clients:
        data_x = data_dict[client][0]
        data_y = data_dict[client][1]
        data = torch.cat([data, data_x], dim = 0)
        labels = torch.cat([labels, data_y], dim = 0).type(torch.int)

    test_dataset = TensorDataset(data, labels)

    return test_dataset

############################### CELEBA ############################### 
def read_dir_celeba(data_dir):
    '''
    code taken from TalwalkarLab/leaf and improvised
    '''
    clients = []
    data = defaultdict(lambda: None)
    bpath = './codeden/data/leaf/data/celeba/data/raw/img_align_celeba'

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]

    print("reading data")
    t = tqdm(range(len(files)))
    for idx, f in enumerate(files):
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata =json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])
        t.update()
    t.refresh()
    t.close()

    print("creating data splits")
    data_dict = defaultdict(lambda: None)
    t = tqdm(range(len(clients)))

    for idx, client in enumerate(clients):
        data_x = readimg(bpath, data[client]['x'])
        data_x = torch.tensor(data_x)
        data_x = data_x.reshape(data_x.shape[0], 28, 28).unsqueeze(dim = 1)
        data_x = data_x.repeat(1, 3, 1, 1)
        data_y = torch.tensor(data[client]['y']).type(torch.int)
        data_dict[client] = (data_x, data_y)
        t.update()
    t.refresh()
    t.close()

    return data_dict, clients

def celeba(data_dir):
    data_dict, clients = read_dir_celeba(data_dir)
    splits = []
    
    for client in clients:
        data_x = data_dict[client][0]
        data_y = data_dict[client][1]
        dataset = TensorDataset(data_x, data_y)
        splits.append(dataset)
    
    return splits




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    #splits = cifar_niid(100)
    #splits = cifar_niid(100, '/data/class_wise.pt')
    splits = celeba_iid('./codeden/data/leaf/data/celeba/data/train/all_data_iid_01_0_keep_5_train_9.json')
    dl = DataLoader(splits[0], batch_size=128, num_workers = 16)

    test_data = jtestDataset('./codeden/data/leaf/data/celeba/data/test/all_data_iid_01_0_keep_5_test_9.json')
    tdl = DataLoader(test_data, batch_size = 128, num_workers = 16)

    print("--- client train ---")
    for batch in tqdm(dl):
            x, y = batch
            print(x.shape)
            break
    
    print("--- test data ---")
    for batch in tqdm(tdl):
            x, y = batch

    '''
    base_dir = '/home/bitwiz/codeden/data/leaf/data/femnist/data/'
    train_dir = os.path.join(base_dir, 'train')
    train_splits = femnist(train_dir)
    dl = DataLoader(train_splits[0], batch_size = 128, num_workers = 16)

    print("--- client train ---")
    for batch in tqdm(dl):
        x, y = batch
        print(x.shape, y.shape)
        break

    test_dir = os.path.join(base_dir, 'test')
    test_set = femnist_test(test_dir)
    dl = DataLoader(test_set, batch_size = 128, num_workers = 8)

    print("--- test data ---")
    for batch in dl:
        x, y = batch
        print(x.shape, y.shape)
        break

    base_dir = '/home/bitwiz/codeden/data/leaf/data/celeba/data/'
    train_dir = os.path.join(base_dir, 'train')
    splits = celeba(train_dir)

    dl = DataLoader(train_splits[0], batch_size = 128, num_workers = 16)

    print("--- client train ---")
    for batch in tqdm(dl):
        x, y = batch
        print(x.shape, y.shape)
        break

    test_dir = os.path.join(base_dir, 'test')
    test_set = celeba(test_dir)
    dl = DataLoader(test_set, batch_size = 128, num_workers = 8)

    print("--- test data ---")
    for batch in dl:
        x, y = batch
        print(x.shape, y.shape)
        break

    '''
