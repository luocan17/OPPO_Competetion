# -*- coding: utf-8 -*-
"""
Created on 2018-10-11
@author: luocan
"""

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from gensim.corpora import Dictionary
import codecs
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold

from metrics import gini_norm
sys.path.append("..")
import myModel
	
def _run_base_model(dTrain, dValid, dTest, model_params):
	
	print("creating graph...")
	model = myModel.toBERT(**model_params)

	if dTrain is not None :
		print("training model...")
		model.fit(dTrain, dValid)
	else:
		# load result
		ckpt_path = "model/model_epoch_new.ckpt"
		model.saver.restore(model.sess, ckpt_path)
		print("Model loading from : %s successfully" % ckpt_path)
	
	if dTest is not None :
		'''
		# 找最佳阈值
		thr, f1= 0.3, 0
		while thr < 0.8:
			a2, p2, r2, f12 = model.evaluate(dValid, thr)
			print("valid-result=[accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f]" % (a2, p2, r2, f12))
			if f1 < f12:
				f1 = f12
				best_thr = thr
			thr += 0.05
		print(best_thr,f1)
		'''
		y_pred = model.predict(dTest)
		y_pred = [str(1) if x > 0.35 else str(0) for x in y_pred]
		print(len(y_pred))
		result = '\n'.join(y_pred)
		fw = open('test_pred.csv','w')
		fw.write(result)
		fw.close()

	return 

# load data

dTrain = np.load('../data/array_ci_train.npy')
dValid = np.load('../data/array_ci_valid.npy')
dTest = np.load('../data/array_ci_test_A.npy')

# params
model_params = {
	"deep_dense": [600, 400, 400, 200, 200],
	"deep_dropout": [0.7, 0.7, 0.7, 0.7, 0.7],
	"deep_activation": ["prelu","prelu","prelu","prelu","prelu"],
	"deep_bn_decay": [0.995,0.995,0.995,0.995,0.995,0.995],
	"first_level_lstm_hs": [200], #[200, 100],
	"first_level_lstm_dropout": [0.7], #[0.8, 0.8],
	"tb_dense_hs": 200,
	"filter_sizes": [3],
	"num_filters": [100],
	"conv_pool_dropout": [0.7, 0.7],
	"second_level_lstm_hs": [200],
	"second_level_lstm_dropout": [0.8],
	"batch_size": 2000,
	"learning_rate": 0.005,
	"optimizer_type": "adam",
	"loss_type": "logloss",
	"l2_reg": 0.001,
	"verbose": 5,
	"random_seed": 20181106,
	"epoch": 50,
	"ckpt_path": "model/",
	"embedding_size": 200,
	"vocab_size": 223250,
	"field_size": 13,
	"sentence_size": 5
}
_run_base_model(dTrain, dValid, dTest, model_params)



