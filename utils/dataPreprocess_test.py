from __future__ import division, print_function

import os
from collections import Counter
import numpy as np
from functools import partial
from tqdm import tqdm
SYMB_BEGIN = "@begin"
SYMB_END = "@end"
MAX_WORD_LEN = 10


class Data:
    def __init__(self, dictionary, training, validation, testing, n_entities):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.testing = testing
	self.vocab_size = len(dictionary[0])
        self.n_chars = len(dictionary[1])
        self.n_entities = n_entities
        self.inv_dictionary = {v:k for k,v in dictionary[0].items()}

class Data_preprocessor_test:
	def preprocess(self, data_dir, max_example=None, no_training_set=False, use_chars=True):
		vocab_file = os.path.join(data_dir, "vocab.txt")
		word_dictionary, char_dictionary, training_num, val_num, test_num = self.makeDictionary(data_dir, vocab_file)
		dictionary = (word_dictionary, char_dictionary)
		if no_training_set:
			training = None
		else:
			print("preparing training data ...")
			training_file = os.path.join(data_dir, "train.txt")
			training = self.parse_file(training_file, dictionary, training_num, max_example, use_chars)
		print("preparing validation data ...")
		val_file = os.path.join(data_dir, "dev.txt")
		validation = self.parse_file(val_file, dictionary, val_num, max_example, use_chars)
		print("preparing testing data ...")
		test_file = os.path.join(data_dir, "test.txt")
		testing = self.parse_file(test_file, dictionary, test_num, max_example, use_chars)
		data = Data(dictionary, training, validation, testing, 10)
		return data		

	def parse_file(self, filename, dictionary, number, max_example, use_chars):
		w_dict, c_dict = dictionary[0], dictionary[1]
		all_file_information = []
		with open(filename, 'r') as f:
			for num in tqdm(range(number)):
				paragraph = []
				for lineIndex in range(20):
					paragraph += self.processLine(f.readline())
				line = self.processLine(f.readline())
				f.readline()
				question = line[:-1]
				candidate_all = line[-1].split('|')
				candidate = [[c] for c in candidate_all]
				try:
					cloze = question.index('XXXXX')
				except ValueError:
					print(num, question)
				# tokens/question --> indexes
				paragraph_indices = list(map(lambda w: w_dict.get(w, 0), paragraph))
				question_indices = list(map(lambda w: w_dict.get(w, 0), question))
				if use_chars:
					paragraph_chars = list(map(lambda w: map(lambda c: c_dict.get(c, c_dict[' ']), list(w)[:MAX_WORD_LEN]), paragraph))
					question_chars = list(map(lambda w: map(lambda c: c_dict.get(c, c_dict[' ']), list(w)[:MAX_WORD_LEN]), question))
				else:
					paragraph_chars = []
					question_chars = []
				# ans/cand --> index
				candidate_indices = [list(map(lambda w: w_dict.get(w, 0), c)) for c in candidate]
				all_file_information.append((paragraph_indices, question_indices, candidate_indices, \
							paragraph_chars, question_chars, cloze, str(num), candidate_all))
		return all_file_information



	def processLine(self, line):
		line = line.replace("``", '"')
		line = line.replace("''", '"')
		line = line.replace('`', "'")
		return line.split()[1:]

	def readFile(self, filename, count):
		cnt = 0
		sample = []
		with open(filename, 'r') as f:
			while True:
				line = self.processLine(f.readline())
				if line == []:
					break
				cnt += 1
				sample += line
				for i in range(19):
					line = self.processLine(f.readline())
					sample += line
				# q & a
				line = self.processLine(f.readline())
				sample += line[:-2]
				# entities = line[-1].split('|')
				# sample += line
				for token in sample:
					count[token] += 1
				
				f.readline()
				sample = []
		return cnt
	
				
			

	def makeDictionary(self, data_dir, vocab_file):
		if os.path.exists(vocab_file):
			print("loading vocabularies from " + vocab_file)
			vocabularies = list(map(lambda x: x.strip(), open(vocab_file, 'r').readlines()))
			train_num = 108719
			val_num = 2000
			test_num = 2500
		else:
			train_file = os.path.join(data_dir, 'train.txt')#'data/train.txt'
			val_file = os.path.join(data_dir, 'dev.txt')
			test_file = os.path.join(data_dir, 'test.txt')
			word_counter = Counter()
			entities_counter = Counter()
			print("processing training files ...")
			train_num = self.readFile(train_file, word_counter)
			print("processing validation files ...")	
			val_num = self.readFile(val_file, word_counter)
			print("processing testing files ...")
			test_num = self.readFile(test_file, word_counter)
			vocabularies = list(set(word_counter.keys()))
			# vocabularies.append(SYMB_END)
			# vocabularies.append(SYMB_BEGIN)	
			with open(vocab_file, "w") as f:
				f.write('\n'.join(vocabularies))			
			
		vocabularies_size = len(vocabularies)
		word_dictionary = dict(zip(vocabularies, range(vocabularies_size)))
		char_set = set([c for w in vocabularies for c in list(w)])
		char_set.add(' ')
		char_dictionary = dict(zip(list(char_set), range(len(char_set))))
		print("vocabularies_size = %d" % vocabularies_size)
		print("num characters = %d" % len(char_set))
		return word_dictionary, char_dictionary, train_num, val_num, test_num

if __name__ == '__main__':
	data_dir = '../data'
	data = Data_preprocessor()	
	d = data.preprocess(data_dir, use_chars=True)
	training = np.array(d.training).shape
	print(training)
