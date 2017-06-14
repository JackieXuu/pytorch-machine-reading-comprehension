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
from utils import DataLoader, load_word2vec_embeddings, Data_preprocessor
from model import GAReader
from tqdm import tqdm
from torch.autograd import Variable
parser = argparse.ArgumentParser()

parser.add_argument('--use_feat', action="store_true", default=False)
parser.add_argument('--train_emb', action="store_true", default=True)
parser.add_argument('--data_dir', type=str, default="data")
parser.add_argument('--resume_model', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default="checkpoint")
parser.add_argument('--embed_file', type=str, default="word2vec_glove.txt")
parser.add_argument('--gru_size', type=int, default=256)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--vocab_size', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--max_example', type=int, default=None)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--grad_clip', type=float, default=10)
parser.add_argument('--init_learning_rate', type=float, default=5e-5)
parser.add_argument('--seed', type=int, default=1)
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
	data = Data_preprocessor()

	data = data.preprocess(data_dir=args.data_dir, max_example=args.max_example, no_training_set=False, use_chars=use_chars)
	training_batch_loader = DataLoader(data.training, batch_size, shuffle=True)
	validation_batch_loader = DataLoader(data.validation, batch_size, shuffle=False)
	testing_batch_loader = DataLoader(data.testing, batch_size, shuffle=False)
	print("loading word2vec file")
	embed_path = os.path.join(args.data_dir, args.embed_file)
	embed_init, embed_dim = load_word2vec_embeddings(data.dictionary[0], embed_path)
	print("Embedding dimension: {}".format(embed_dim))
	model = GAReader(args.n_layers, data.vocab_size, data.n_chars,args.drop_out, args.gru_size, embed_init, embed_dim, \
					args.train_emb, args.char_dim, args.use_feat,args.gating_fn)
	model.cuda()
	optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_learning_rate)
	criterion = nn.CrossEntropyLoss().cuda()
	return model, optimizer, criterion, training_batch_loader, validation_batch_loader, testing_batch_loader

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=5):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


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



def train(model, optimizer, criterion, training_batch_loader, epoch):
	model.train()
	acc = loss = 0
	#batch_idx = 0
	for batch_idx, (dw, dt, qw, qt, a, m_dw, m_qw, \
		tt, tm, c, m_c, cl, num, _, _) in enumerate(training_batch_loader):#, total=len(training_batch_loader)):
		dw, dt, qw, qt, a, m_dw, m_qw,\
			 tt, tm, c, m_c, cl = variable([dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl])
		loss_, acc_, _, _ = model(dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl)
		loss += loss_.cpu().data.numpy()[0]
		acc_ = acc_.cpu().data.numpy()[0] / float(dw.size(0))
		acc += acc_

		if (batch_idx+1) % args.print_every == 0:
			print('[Train] Epoch {} Iter {}/{} Accuracy {:.4f} Loss {:.7f}'.format(epoch, batch_idx+1, len(training_batch_loader), acc_, loss_.cpu().data.numpy()[0]))
		optimizer.zero_grad()
		loss_.backward()
		clip_grad_norm(parameters=filter(lambda p: p.requires_grad, model.parameters()),max_norm=args.grad_clip)
		optimizer.step()
	#	batch_idx += 1
	acc /= len(training_batch_loader)
	loss /= len(training_batch_loader)
	print('[Train] Epoch {} Average Accuracy {:.4f} Average Loss {:.7f}'.format(epoch, acc, loss))

def valid(model, criterion, validation_batch_loader, epoch):
	model.eval()
	acc = loss = 0
#	batch_idx = 0
	for batch_idx, (dw, dt, qw, qt, a, m_dw, m_qw, \
		tt, tm, c, m_c, cl, num, _, _) in tqdm(enumerate(validation_batch_loader), total=len(validation_batch_loader)):
		dw, dt, qw, qt, a, m_dw, m_qw,\
			 tt, tm, c, m_c, cl = variable([dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl], is_training=False)
		loss_, acc_, _, _ = model(dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl)

                loss += loss_.cpu().data.numpy()[0]
                acc_ = acc_.cpu().data.numpy()[0] / float(dw.size(0))
		acc += acc_
	acc /= len(validation_batch_loader)
	loss /= len(validation_batch_loader)
	print('[Valid] Epoch {} Average Accuracy {:.4f} Average Loss {:.7f}'.format(epoch, acc, loss))
	return acc


def test(model, criterion, testing_batch_loader, epoch):
        model.eval()
        acc = loss = 0
#       batch_idx = 0
        for batch_idx, (dw, dt, qw, qt, a, m_dw, m_qw, \
                tt, tm, c, m_c, cl, num, _, _) in tqdm(enumerate(testing_batch_loader), total=len(testing_batch_loader)):
                dw, dt, qw, qt, a, m_dw, m_qw,\
                         tt, tm, c, m_c, cl = variable([dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl], is_training=False)
                loss_, acc_, _, _ = model(dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl)

                loss += loss_.cpu().data.numpy()[0]
                acc_ = acc_.cpu().data.numpy()[0] / float(dw.size(0))
                acc += acc_
        acc /= len(testing_batch_loader)
        loss /= len(testing_batch_loader)
        print('[Test] Epoch {} Average Accuracy {:.4f} Average Loss {:.7f}'.format(epoch, acc, loss))
        return acc


def save_checkpoint(state, filename='checkpoint.pth'):
	torch.save(state, filename)

def resume(filename, optimizer, model):
	if os.path.isfile(filename):
		print('==> loading  checkpoint {}'.format(filename))
		checkpoint = torch.load(filename)
		start_epoch = checkpoint['epoch'] + 1
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("==> loaded checkpoint '{}' (epoch {})".format(filename, start_epoch))
	else:
		print("==> no checkpoint found at '{}'".format(filename))
	return model, optimizer, start_epoch


def main():
	if not os.path.isdir(args.checkpoint):
		os.mkdir(args.checkpoint)
	model, optimizer, criterion, training_batch_loader, validation_batch_loader, testing_batch_loader = init()
	start_epoch = 1
	best_acc = 0.0
	if args.resume_model is not None:
		filename = os.path.join(args.checkpoint, args.resume_model)
		model, optimizer, start_epoch = resume(filename, optimizer, model)
	valid(model, criterion, validation_batch_loader, 0)
	best_acc = test(model, criterion, testing_batch_loader, 0)
	for epoch in range(start_epoch, args.epoch):
		exp_lr_scheduler(optimizer, epoch)
		train(model, optimizer, criterion, training_batch_loader, epoch)
		#acc = valid(model, criterion, validation_batch_loader, epoch)
		if epoch % args.save_every == 0:
			filename = os.path.join(args.checkpoint, "checkpoint_nlp{}.pth".format(epoch))
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
			}, filename=filename)
			#if acc > best_acc:
			#	best_acc =acc
			#	shutil.copyfile(filename,  os.path.join(args.checkpoint,"best_model_nlp.pth"))
		_ = valid(model, criterion, validation_batch_loader, epoch)
		acc = test(model, criterion, testing_batch_loader, epoch)
		if epoch % args.save_every == 0:
			if acc > best_acc:
				best_acc =acc
				shutil.copyfile(filename,  os.path.join(args.checkpoint,"best_model_nlp.pth"))
		
if __name__ == '__main__':
	main()








