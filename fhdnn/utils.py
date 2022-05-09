from argparse import ArgumentParser
import csv

def generate_parser():
	parser = ArgumentParser()
	parser.add_argument('--cpu', action = 'store_true')
	parser.add_argument('-E', '--local_epochs', type = int, default = 1)
	parser.add_argument('-B', '--local_batch_size', type = int, default = 10)
	parser.add_argument('-C', '--fraction', type = float, default = 0.1)
	parser.add_argument('-d', '--D', type = int, default = 10000)
	parser.add_argument('-nc', '--nclients', type = int, default = 100)
	parser.add_argument('-n', '--nrounds', type = int, default = 20)
	parser.add_argument('--iid', action = 'store_true')
	parser.add_argument('--debug', action = 'store_true')
	parser.add_argument('--lr', type = float, default = 1)
	parser.add_argument('--decay_lr', type = float, default = 0.99)
	parser.add_argument('--dataset', type = str)
	parser.add_argument('--resnet', type = int, default = 18)
	parser.add_argument('--simclr', action = 'store_true')
	parser.add_argument('--log', action = 'store_true')
	parser.add_argument('--cuda', type = int, default = 0)
	parser.add_argument('--lifelong', action = 'store_true')

	return parser

