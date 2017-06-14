from __future__ import division, print_function

import numpy as np
import random
from builtins import range
MAX_WORD_LEN = 10

class DataLoader_test:
	def __init__(self, samples, batch_size, shuffle=True):
		self.batch_size = batch_size
		self.samples = np.array(samples)
		self.length = self.samples.shape[0]
		self.shuffle = shuffle
		if self.shuffle:
			idx = np.arange(0, self.length)
			np.random.shuffle(idx)
			self.samples = self.samples[idx]
		else:
			self.samples = self.samples
		self.max_word_len = MAX_WORD_LEN
		self.reset()
	
	def __len__(self):
		return len(self.batch_pool)
	
	def reset(self):
		self.iter = 0
		self.batch_pool = []
		idx_list = np.arange(0, self.length, self.batch_size)
		if self.shuffle:
			idx = np.arange(0, self.length)
			np.random.shuffle(idx)
			self.samples = self.samples[idx]
		for idx in idx_list:
			self.batch_pool.append(np.arange(idx, min(idx+self.batch_size, self.length)))
	
	def __iter__(self):
		return self

	def next(self):
		if self.iter == len(self.batch_pool):
			self.reset()
			raise StopIteration()
		idxs = self.batch_pool[self.iter]
		paragraph_len = [len(self.samples[idx][0]) for idx in idxs]
		question_len = [len(self.samples[idx][1]) for idx in idxs]
		candidate_len = [len(self.samples[idx][3]) for idx in idxs]
		cur_max_paragraph_len = np.max(paragraph_len)
		cur_max_question_len = np.max(question_len)
		cur_max_candidate_len = np.max(candidate_len)
		cur_batch_size = len(idxs)
	
		# document words
		dw = np.zeros((cur_batch_size, cur_max_paragraph_len), dtype='int32')
		# question words
		qw = np.zeros((cur_batch_size, cur_max_question_len), dtype='int32')
		# candidate answers
		c = np.zeros((cur_batch_size, cur_max_paragraph_len, cur_max_candidate_len), dtype='int16')
		# position of cloze in query
		cl = np.zeros((cur_batch_size, ),  dtype='int32')
		# document word mask
		m_dw = np.zeros((cur_batch_size, cur_max_paragraph_len), dtype='int32')
		# query word mask
		m_qw = np.zeros((cur_batch_size, cur_max_question_len), dtype='int32')
		# candidate mask
		m_c = np.zeros((cur_batch_size, cur_max_paragraph_len), dtype='int32')
		# num
		nums = np.zeros((cur_batch_size, ), dtype='int32')
		
		candidate_all = []
		types = {}
		for n, ix in enumerate(idxs):
			paragraph_indices, question_indices, candidate_indices, \
				paragraph_chars, question_chars, cloze, num, candidate = self.samples[ix]
			candidate_all.append(candidate)
			dw[n,:len(paragraph_indices)] = np.array(paragraph_indices)
			qw[n,:len(question_indices)] = np.array(question_indices)
			m_dw[n,:len(paragraph_indices)] = 1
			m_qw[n,:len(question_indices)] = 1
			for it, word in enumerate(paragraph_chars):
				wtuple = tuple(word)
				if wtuple not in types:
					types[wtuple] = []
				types[wtuple].append((0, n, it))
			for it, word in enumerate(question_chars):
				wtuple = tuple(word)
				if wtuple not in types:
					types[wtuple] = []
				types[wtuple].append((1, n, it))	
			for it, cc in enumerate(candidate_indices):
				index = [ii for ii in range(len(paragraph_indices)) if paragraph_indices[ii] in cc]
				m_c[n, index] = 1
				c[n, index, it] = 1
			cl[n] = cloze
		dt = np.zeros((cur_batch_size, cur_max_paragraph_len), dtype='int32')
		qt = np.zeros((cur_batch_size, cur_max_question_len), dtype='int32')
		tt = np.zeros((len(types), self.max_word_len), dtype='int32')
		tm = np.zeros((len(types), self.max_word_len), dtype='int32')
		n = 0
		for k, v in types.items():
			tt[n, :len(k)] = np.array(k)
			tm[n, :len(k)] = 1
			for (sw, bn, sn) in v:
				if sw == 0:
					dt[bn, sn] = n
				else:
					qt[bn, sn] = n
			n += 1

		self.iter += 1
		return dw, dt, qw, qt, m_dw, m_qw, tt, tm, c, m_c, cl, str(nums), candidate_all

if __name__ == '__main__':
	import numpy as np
	a = [[2],[8], [1], [1],[1],[1],[1],[1]]
	c = [a for _ in range(1000)]
	print(np.array(c).shape)
	b = DataLoader(c, 20)
#	for data in b:
#		print(data)
