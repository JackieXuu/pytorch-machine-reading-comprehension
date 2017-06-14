from __future__ import division
from __future__ import print_function

import argparse
import torch
import numpy as np
import time
import os
import shutil
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from utils import DataLoader_test, load_word2vec_embeddings, Data_preprocessor_test
from model_predict import GAReader
from tqdm import tqdm
from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument('--use_feat', action="store_true", default=False)
parser.add_argument('--train_emb', action="store_true", default=False)
parser.add_argument('--data_dir', type=str, default="data")
parser.add_argument('--resume_model', type=str, default="best_model_nlp.pth")
parser.add_argument('--checkpoint', type=str, default="checkpoint")
parser.add_argument('--embed_file', type=str, default="data/word2vec_glove.txt")
parser.add_argument('--gru_size', type=int, default=256)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--vocab_size', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--max_example', type=int, default=None)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--grad_clip', type=float, default=10)
parser.add_argument('--init_learning_rate', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--char_dim', type=int, default=25)
parser.add_argument('--gating_fn', type=str, default='tmul')
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--gpu', type=str, default="0")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
def init():
	if args.seed:
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)
	else:
		seed = np.random.randint(2**31)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.benchmark = True
	batch_size = args.batch_size
	use_chars = args.char_dim > 0
	data = Data_preprocessor_test()

	data = data.preprocess(data_dir=args.data_dir, max_example=args.max_example, no_training_set=True, use_chars=use_chars)
	validation_batch_loader = DataLoader_test(data.validation, batch_size, shuffle=False)
	testing_batch_loader = DataLoader_test(data.testing, batch_size, shuffle=False)
	print("loading word2vec file")
	embed_init, embed_dim = load_word2vec_embeddings(data.dictionary[0], args.embed_file)
	print("Embedding dimension: {}".format(embed_dim))
	model = GAReader(args.n_layers, data.vocab_size, data.n_chars,args.drop_out, args.gru_size, embed_init, embed_dim, \
					args.train_emb, args.char_dim, args.use_feat,args.gating_fn)
	model.cuda()
	optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_learning_rate)
	criterion = nn.CrossEntropyLoss().cuda()
	return model, optimizer, criterion, validation_batch_loader, testing_batch_loader


def variable(inputs, use_cuda=True, is_training=True):
	if use_cuda:
		if is_training:
			return [Variable(torch.from_numpy(i).cuda()) for i in inputs]
		else:
			return [Variable(torch.from_numpy(i).cuda(), volatile=True) for i in inputs]
	else:
		if is_training:
			return [Variable(torch.from_numpy(i)) for i in inputs]
		else:
			return [Variable(torch.from_numpy(i), volatile=True) for i in inputs]

def test(model, criterion, batch_loader):
        model.eval()
        acc = loss = 0
#       batch_idx = 0
        f = open("result.txt", 'w')
        for batch_idx, (dw, dt, qw, qt, m_dw, m_qw, \
                tt, tm, c, m_c, cl, num, candidate_all) in tqdm(enumerate(batch_loader), total=len(batch_loader)):
                dw, dt, qw, qt, m_dw, m_qw,\
                         tt, tm, c, m_c, cl = variable([dw, dt, qw, qt, m_dw, m_qw, tt, tm, c, m_c, cl], is_training=False)
                pred_ans = model(dw, dt, qw, qt, m_dw, m_qw, tt, tm, c, m_c, cl)

                pred = np.squeeze(pred_ans.cpu().data.numpy(), axis=1)
                for idx in range(len(candidate_all)):
                        f.write(candidate_all[idx][pred[idx]])
                        f.write('\n')


def resume(filename, optimizer, model):
	if os.path.isfile(filename):
		print('==> loading  checkpoint {}'.format(filename))
		checkpoint = torch.load(filename)
		model.load_state_dict(checkpoint['state_dict'])
	else:
		print("==> no checkpoint found at '{}'".format(filename))
	return model


def main():
	model, optimizer, criterion, validation_batch_loader, testing_batch_loader = init()
	filename = os.path.join(args.checkpoint, args.resume_model)
	model = resume(filename, optimizer, model)
	test(model, criterion, testing_batch_loader)
		
if __name__ == '__main__':
	main()








