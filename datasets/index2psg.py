import os
import re
import torch
import copy
import json, csv
import time
import argparse
from tqdm import tqdm, trange
import pickle


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))

pid2passage = {} # {pid: passage_text}
with open("./datasets/wikipedia/psgs_w100.tsv", "r") as f:
	first_line = True
	for line in tqdm(f):
		if first_line:
			first_line = False
			continue
		line = line.strip()
		try:
			line_arr = line.split("\t")
			pid = int(line_arr[0])
			passage = line_arr[2].rstrip() + ' ' + line_arr[1].rstrip()
			pid2passage[pid] = filtering_invalid_string(passage)
		except IndexError:
			print("bad passage")
		except ValueError:
			print("bad pid")
pstore(pid2passage, "./wikipedia/pid2psg.pkl")
