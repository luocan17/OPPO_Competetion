# -*- coding: utf-8 -*-
"""
Created on 2018-10-11
@author: luocan

Tensorflow implementation of DeepFM [1]
Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
	Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import numpy as np
import tensorflow as tf
import re
import jieba
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
#from yellowfin import YFOptimizer
import logging
import metrics
import random
from sklearn.model_selection import StratifiedKFold

class toBERT(BaseEstimator, TransformerMixin):
	def __init__(self, deep_dense,
			deep_dropout,
			deep_activation,
			deep_bn_decay,
			first_level_lstm_hs,
			first_level_lstm_dropout,
			tb_dense_hs,
			filter_sizes,
			num_filters,
			conv_pool_dropout,
			second_level_lstm_hs,
			second_level_lstm_dropout,
			batch_size,
			learning_rate,
			optimizer_type,
			loss_type,
			l2_reg,
			verbose,
			random_seed,
			epoch,
			ckpt_path,
			embedding_size,
			vocab_size,
			field_size,
			sentence_size,
			greater_is_better=True):
		assert field_size == 13
		assert loss_type in ["logloss", "mse"], \
			"loss_type can be either 'logloss' for classification task or 'mse' for regression task"

		self.vocab_size = vocab_size		# denote as V, size of the vocabulary
		self.field_size = field_size			# denote as F, size of the feature fields
		self.embedding_size = embedding_size	# denote as K, size of the feature embedding
		self.sentence_size = sentence_size # denote as L

		self.deep_dense = deep_dense
		self.deep_dropout = deep_dropout
		self.deep_activation = []
		self.deep_bn_decay = deep_bn_decay
		self.first_level_lstm_hs = first_level_lstm_hs
		self.first_level_lstm_dropout = first_level_lstm_dropout
		self.tb_dense_hs = tb_dense_hs
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.total_num_filters = 0
		self.conv_pool_dropout = conv_pool_dropout
		for x in self.num_filters:
			self.total_num_filters += x
		self.second_level_lstm_hs = second_level_lstm_hs
		self.second_level_lstm_dropout = second_level_lstm_dropout
		for act in deep_activation:
			if act == "relu":
				self.deep_activation.append(tf.nn.relu)
			elif act == "prelu":
				self.deep_activation.append(self.prelu_layer)

		self.l2_reg = l2_reg

		self.epoch = epoch
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.optimizer_type = optimizer_type

		self.verbose = verbose
		self.random_seed = random_seed
		self.ckpt_path = ckpt_path
		self.loss_type = loss_type
		self.greater_is_better = greater_is_better
		
		self._init_graph()
		
		# initialize logger
		logging.basicConfig(level=logging.DEBUG, # CRITICAL-ERROR-WARNING-INFO-DEBUG-NOTSET
							filename= 'record_errors.log',
							filemode='w', # 模式，有w和a，默认为a
							format='%(pathname)s[line:%(lineno)d]: %(message)s' # 日志格式
							)


	def _init_graph(self):
		self.graph = tf.Graph()
		with self.graph.as_default():

			tf.set_random_seed(self.random_seed)

			self.vocab_index = tf.placeholder(tf.int32, shape=[None, None], name="vocab_index")  # None * (F*L)
			self.props = tf.placeholder(tf.float32, shape=[None, None], name="props") # None * 12, [1.0,props,1.0]
			self.label = tf.placeholder(tf.int32, shape=[None, 1], name="label")  # None * 1
			self.first_level_lstm_dropout_p = tf.placeholder(tf.float32, shape=[None], name="first_level_lstm_dropout")
			self.deep_dropout_p = tf.placeholder(tf.float32, shape=[None], name="deep_dropout")
			self.conv_pool_dropout_p = tf.placeholder(tf.float32, shape=[None], name="conv_pool_dropout")
			self.second_level_lstm_dropout_p = tf.placeholder(tf.float32, shape=[None], name="second_level_lstm_dropout")
			self.train_phase = tf.placeholder(tf.bool, name="train_phase")

			self.weights = self._initialize_weights()

			# model
			vocab0_embedding = tf.constant(0,dtype=tf.float32,shape=[1,self.embedding_size],name="vocab0_embedding")
			vocab_embeddings = tf.concat([vocab0_embedding,self.weights["vocab_embeddings"]],axis=0)
			self.embeddings = tf.nn.embedding_lookup(vocab_embeddings, self.vocab_index)  # None * (F*L) * K
			emb_prefix = self.embeddings[:,0:self.sentence_size,:] # None * L * K
			emb_querys = self.embeddings[:,self.sentence_size:(self.field_size-2)*self.sentence_size,:] # None * 10L * K
			emb_title = self.embeddings[:,(self.field_size-2)*self.sentence_size:(self.field_size-1)*self.sentence_size,:] # None * L * K
			emb_tag = self.embeddings[:,(self.field_size-1)*self.sentence_size:,:] # None * L * K
			emb_prefix10 = tf.reshape(tf.tile(emb_prefix,[1,10,1]), shape=[-1,self.sentence_size,self.embedding_size]) # (None*10) * L * K
			emb_querys = tf.reshape(emb_querys, shape=[-1, self.sentence_size, self.embedding_size]) # (None*10) * L * K
			emb_title10 = tf.reshape(tf.tile(emb_title, [1, 10, 1]), shape=[-1, self.sentence_size, self.embedding_size])  # (None*10) * L * K
			emb_tag10 = tf.reshape(tf.tile(emb_tag, [1, 10, 1]), shape=[-1, self.sentence_size, self.embedding_size])  # (None*10) * L * K
			
			# ---------- First Level LSTM component ----------
			fir_lstm0_l0 = tf.contrib.rnn.BasicLSTMCell(self.first_level_lstm_hs[0])
			fir_lstm0_l0 = tf.contrib.rnn.DropoutWrapper(fir_lstm0_l0, output_keep_prob=self.first_level_lstm_dropout_p[0])
			cell0 = fir_lstm0_l0 # tf.nn.rnn_cell.MultiRNNCell([fir_lstm0_l0,fir_lstm0_l1])
			self.fir_lstm0_in = emb_prefix10
			state = cell0.zero_state(self.batch_size*10, tf.float32) # (None*10) * H
			with tf.variable_scope("fir_level_lstm0"):
				for time_step in range(self.sentence_size):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()
					cell_output, state = cell0(self.fir_lstm0_in[:, time_step, :], state)
			self.fir_lstm0_out = cell_output # (None*10) * H

			fir_lstm1_l0 = tf.contrib.rnn.BasicLSTMCell(self.first_level_lstm_hs[0])
			fir_lstm1_l0 = tf.contrib.rnn.DropoutWrapper(fir_lstm1_l0, output_keep_prob=self.first_level_lstm_dropout_p[0])
			# fir_lstm0_l1 = tf.contrib.rnn.BasicLSTMCell(self.lstm_first_level_hs[1])
			# fir_lstm0_l1 = tf.contrib.rnn.DropoutWrapper(fir_lstm0_l1, output_keep_prob=self.dropout_keep_lstm_first_level[1])
			cell1 = fir_lstm1_l0  # tf.nn.rnn_cell.MultiRNNCell([fir_lstm0_l0,fir_lstm0_l1])
			self.fir_lstm1_in = emb_querys
			state = cell1.zero_state(self.batch_size * 10, tf.float32)  # (None*10) * H
			with tf.variable_scope("fir_level_lstm1"):
				for time_step in range(self.sentence_size):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()
					cell_output, state = cell1(self.fir_lstm0_in[:, time_step, :], state)
			self.fir_lstm1_out = cell_output  # (None*10) * H

			fir_lstm2_l0 = tf.contrib.rnn.BasicLSTMCell(self.first_level_lstm_hs[0])
			fir_lstm2_l0 = tf.contrib.rnn.DropoutWrapper(fir_lstm2_l0, output_keep_prob=self.first_level_lstm_dropout_p[0])
			cell2 = fir_lstm2_l0  # tf.nn.rnn_cell.MultiRNNCell([fir_lstm0_l0,fir_lstm0_l1])
			self.fir_lstm2_in = emb_title10
			state = cell2.zero_state(self.batch_size * 10, tf.float32)  # (None*10) * H
			with tf.variable_scope("fir_level_lstm2"):
				for time_step in range(self.sentence_size):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()
					cell_output, state = cell2(self.fir_lstm0_in[:, time_step, :], state)
			self.fir_lstm2_out = cell_output  # (None*10) * H

			fir_lstm3_l0 = tf.contrib.rnn.BasicLSTMCell(self.first_level_lstm_hs[0])
			fir_lstm3_l0 = tf.contrib.rnn.DropoutWrapper(fir_lstm3_l0, output_keep_prob=self.first_level_lstm_dropout_p[0])
			cell3 = fir_lstm3_l0  # tf.nn.rnn_cell.MultiRNNCell([fir_lstm0_l0,fir_lstm0_l1])
			self.fir_lstm3_in = emb_tag10
			state = cell3.zero_state(self.batch_size * 10, tf.float32)  # (None*10) * H
			with tf.variable_scope("fir_level_lstm3"):
				for time_step in range(self.sentence_size):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()
					cell_output, state = cell3(self.fir_lstm0_in[:, time_step, :], state)
			self.fir_lstm3_out = cell_output  # (None*10) * H

			# -----------TimeDistributed(Dense) component------------------
			self.tb_dense0_in = tf.reshape(emb_prefix10, shape=[-1,self.embedding_size]) # (None*10*L) * K
			x0 = tf.add(tf.matmul(self.tb_dense0_in, self.weights["tb_dense_prefix_weight"]), self.weights["tb_dense_prefix_bias"])
			self.tb_dense0_out = tf.reshape(x0, shape=[-1, self.sentence_size, self.tb_dense_hs]) #(None*10) * L * self.tb_dense_hs
			self.tb_dense0_out = tf.reduce_sum(self.tb_dense0_out,axis=1) # lambda; (None*10) * self.tb_dense_hs
			self.tb_dense0_out = tf.nn.relu(self.tb_dense0_out, name="tb_dense0_relu")

			self.tb_dense1_in = tf.reshape(emb_querys, shape=[-1, self.embedding_size])  # (None*10*L) * K
			x1 = tf.add(tf.matmul(self.tb_dense1_in, self.weights["tb_dense_query_weight"]), self.weights["tb_dense_query_bias"])
			self.tb_dense1_out = tf.reshape(x1, shape=[-1, self.sentence_size, self.tb_dense_hs])  # (None*10) * L * self.tb_dense_hs
			self.tb_dense1_out = tf.reduce_sum(self.tb_dense1_out, axis=1)  # lambda; (None*10) * self.tb_dense_hs
			self.tb_dense1_out = tf.nn.relu(self.tb_dense1_out, name="tb_dense1_relu")

			self.tb_dense2_in = tf.reshape(emb_title10, shape=[-1, self.embedding_size])  # (None*10*L) * K
			x2 = tf.add(tf.matmul(self.tb_dense2_in, self.weights["tb_dense_title_weight"]), self.weights["tb_dense_title_bias"])
			self.tb_dense2_out = tf.reshape(x2, shape=[-1, self.sentence_size, self.tb_dense_hs])  # (None*10) * L * self.tb_dense_hs
			self.tb_dense2_out = tf.reduce_sum(self.tb_dense2_out, axis=1)  # lambda; (None*10) * self.tb_dense_hs
			self.tb_dense2_out = tf.nn.relu(self.tb_dense2_out, name="tb_dense2_relu")

			self.tb_dense3_in = tf.reshape(emb_tag10, shape=[-1, self.embedding_size])  # (None*10*L) * K
			x3 = tf.add(tf.matmul(self.tb_dense3_in, self.weights["tb_dense_tag_weight"]), self.weights["tb_dense_tag_bias"])
			self.tb_dense3_out = tf.reshape(x3, shape=[-1, self.sentence_size, self.tb_dense_hs])  # (None*10) * L * self.tb_dense_hs
			self.tb_dense3_out = tf.reduce_sum(self.tb_dense3_out, axis=1)  # lambda; (None*10) * self.tb_dense_hs
			self.tb_dense3_out = tf.nn.relu(self.tb_dense3_out, name="tb_dense3_relu")

			# ---------- First-Level's third component ---------------------
			querys4d = tf.expand_dims(emb_querys,axis=-1) # expand last dim; (None*10) * L * K *1
			pooled_outputs_query = []
			for i,filter_size in enumerate(self.filter_sizes):
				conv = tf.nn.conv2d(
							querys4d,
							self.weights["conv_filter_weight_%d"%i], # filter_size*K*1*64
							strides=[1,1,1,1],
							padding="VALID",
							name="conv_filter_%d"%i ) # (None*10) * (self.sentence_size-filter_size+1) * 1 * 64
				h = tf.nn.relu(tf.nn.bias_add(conv,self.weights["conv_filter_bias_%d"%i]), name="conv_relu_%d"%i)
				pooled = tf.nn.max_pool(
							h,
							ksize=[1,self.sentence_size-filter_size+1,1,1],
							strides=[1,1,1,1],
							padding="VALID",
							name="pool_filter_%d"%i ) # (None*10) * 1 * 1 * 64
				pooled_outputs_query.append(pooled)
			pooled_outputs_query = tf.concat(pooled_outputs_query,axis=3)
			self.pooled_outputs_query = tf.reshape(pooled_outputs_query, shape=[-1,self.total_num_filters]) # (None*10) * total_num_filters
			self.pooled_outputs_query = self.batch_norm_layer(self.pooled_outputs_query,0.998,self.train_phase,"conv_bn_0")
			self.pooled_outputs_query = tf.nn.dropout(self.pooled_outputs_query, self.conv_pool_dropout_p[0])

			title4d = tf.expand_dims(emb_title10, axis=-1)
			pooled_outputs_title = []
			for i, filter_size in enumerate(self.filter_sizes):
				conv = tf.nn.conv2d(
					title4d,
					self.weights["conv_filter_weight_%d" % i],  # filter_size*K*1*64
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv_filter_%d" % i)  # (None*10) * (self.sentence_size-filter_size+1) * 1 * 64
				h = tf.nn.relu(tf.nn.bias_add(conv, self.weights["conv_filter_bias_%d" % i]), name="conv_relu_%d" % i)
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, self.sentence_size - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="pool_filter_%d" % i)  # (None*10) * 1 * 1 * 64
				pooled_outputs_title.append(pooled)
			pooled_outputs_title = tf.concat(pooled_outputs_title, axis=3)
			self.pooled_outputs_title = tf.reshape(pooled_outputs_title, shape=[-1,self.total_num_filters]) # (None*10) * total_num_filters
			self.pooled_outputs_title = self.batch_norm_layer(self.pooled_outputs_title,0.998,self.train_phase,"conv_bn_1")
			self.pooled_outputs_title = tf.nn.dropout(self.pooled_outputs_title, self.conv_pool_dropout_p[1])

			# ------------ Deep component -------------------
			self.deep_in = tf.concat([self.fir_lstm0_out,self.fir_lstm1_out,self.fir_lstm2_out,self.fir_lstm3_out,
							self.tb_dense0_out,self.tb_dense1_out,self.tb_dense2_out,self.tb_dense3_out,
							self.pooled_outputs_query,self.pooled_outputs_title], axis=1) # (None*10) * (1600+128)
			xx = self.batch_norm_layer(self.deep_in, self.deep_bn_decay[0], train_phase=self.train_phase, scope_bn="deep_bn_0")
			for i in range(0, len(self.deep_dense)):
				xx = tf.add(tf.matmul(xx, self.weights["deep_dense_weight_%d" %i]), self.weights["deep_dense_bias_%d"%i])
				xx = self.deep_activation[i](xx, name="prelu%d"%i) # (None*10) * self.deep_dense[i]
				xx = tf.nn.dropout(xx, self.deep_dropout_p[i]) # dropout at each Deep layer
				xx = self.batch_norm_layer(xx, self.deep_bn_decay[i+1], train_phase=self.train_phase, scope_bn="deep_bn_%d" % (i+1))
			self.deep_out = tf.reshape(xx, shape=[-1,10,self.deep_dense[-1]]) # None * 10 * self.deep_dense[-1]
			temp = tf.reshape(self.props, shape=[-1, 10, 1])
			temp = tf.tile(temp, [1, 1, self.deep_dense[-1]])
			self.deep_out = tf.multiply(self.deep_out, temp)  # None * 10 * self.deep_dense[-1]

			# ---------- Second Level LSTM component ----------
			sec_lstm_l0 = tf.contrib.rnn.BasicLSTMCell(self.second_level_lstm_hs[0])
			sec_lstm_l0 = tf.contrib.rnn.DropoutWrapper(sec_lstm_l0, output_keep_prob=self.second_level_lstm_dropout_p[0])
			cell = sec_lstm_l0 # tf.nn.rnn_cell.MultiRNNCell([sec_lstm_l0, sec_lstm_l1])
			self.sec_lstm_in = self.deep_out
			state = cell.zero_state(self.batch_size, tf.float32)  # None * H
			with tf.variable_scope("sec_level_lstm"):
				for time_step in range(self.field_size-3):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()
					cell_output, state = cell(self.sec_lstm_in[:, time_step, :], state)
			self.sec_lstm_out = cell_output  # None * H

			# ------------ Cos Distance component ------------------------
			q_norm = tf.sqrt(tf.reduce_sum(tf.square(self.fir_lstm1_out), axis=1))  # (None*10) * 1
			t_norm = tf.sqrt(tf.reduce_sum(tf.square(self.fir_lstm2_out), axis=1))  # (None*10) * 1
			iner_pro = tf.reduce_sum(tf.multiply(self.fir_lstm1_out, self.fir_lstm2_out), axis=1)  # inner product between querys4d and title4d
			tt = tf.multiply(q_norm, t_norm)
			cosdis = tf.divide(iner_pro, tf.clip_by_value(tt, 1e-12, tf.reduce_max(tt)))  # (None*10) * 1
			cosdis = tf.reshape(cosdis, shape=[-1,10]) # None * 10

			# ----------- Fc component -----------
			self.fc_in = tf.concat([self.sec_lstm_out, cosdis],axis=1)
			self.fc_out = tf.add(tf.matmul(self.fc_in, self.weights["fc_weight"]), self.weights["fc_bias"])
			self.out = tf.nn.sigmoid(self.fc_out)
			if self.loss_type == "logloss":
				self.loss = tf.losses.log_loss(self.label, self.out)
			elif self.loss_type == "mse":
				self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
			# l2 regularization on weights
			if self.l2_reg > 0:
				self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["fc_weight"])
				for i in range(len(self.deep_dense)):
					self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["deep_dense_weight_%d"%i])

			# optimizer
			if self.optimizer_type == "adam":
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
														epsilon=1e-8).minimize(self.loss)
			elif self.optimizer_type == "adagrad":
				self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
														   initial_accumulator_value=1e-8).minimize(self.loss)
			elif self.optimizer_type == "gd":
				self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
			elif self.optimizer_type == "momentum":
				self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
					self.loss)
			#elif self.optimizer_type == "yellowfin":
			#	self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(
			#		self.loss)

			# init
			self.saver = tf.train.Saver()
			init = tf.global_variables_initializer()
			init1 = tf.local_variables_initializer()
			self.sess = self._init_session()
			self.sess.run(init)
			self.sess.run(init1)

			# number of params
			total_parameters = 0
			for variable in self.weights.values():
				shape = variable.get_shape()
				variable_parameters = 1
				for dim in shape:
					variable_parameters *= dim.value
				total_parameters += variable_parameters
			print("#params: %d" % total_parameters)


	def _init_session(self):
		config = tf.ConfigProto(device_count={"gpu": 0})
		config.gpu_options.allow_growth = True
		return tf.Session(config=config)


	def _initialize_weights(self):
		weights = dict()
		
		# embedding layers
		weights["vocab_embeddings"] = tf.Variable(
			tf.random_normal([self.vocab_size, self.embedding_size], 0.0, 0.01),
			name="vocab_embeddings")  # vocab_size * K
		weights["vocab_bias"] = tf.Variable(
			tf.random_uniform([self.vocab_size, 1], 0.0, 1.0), name="vocab_bias")  # vocab_size * 1

		# timedistributed(dense)
		glorot = np.sqrt(2.0/(self.embedding_size+self.tb_dense_hs))
		weights["tb_dense_prefix_weight"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(self.embedding_size,self.tb_dense_hs)), dtype=np.float32 )
		weights["tb_dense_prefix_bias"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(1, self.tb_dense_hs)), dtype = np.float32 )

		weights["tb_dense_query_weight"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.tb_dense_hs)), dtype=np.float32)
		weights["tb_dense_query_bias"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(1, self.tb_dense_hs)), dtype=np.float32)

		weights["tb_dense_title_weight"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.tb_dense_hs)), dtype=np.float32)
		weights["tb_dense_title_bias"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(1, self.tb_dense_hs)), dtype=np.float32)

		weights["tb_dense_tag_weight"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.tb_dense_hs)), dtype=np.float32)
		weights["tb_dense_tag_bias"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(1, self.tb_dense_hs)), dtype=np.float32)

		# conv component
		for i,filter_size in enumerate(self.filter_sizes):
			filter_shape = [filter_size, self.embedding_size, 1, self.num_filters[i]]
			weights["conv_filter_weight_%d"%i] = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1), name="conv_filter_weight_%d"%i)
			weights["conv_filter_bias_%d"%i] = tf.Variable(tf.constant(0.1, shape=[self.num_filters[i]]), name="conv_filter_bias_%d"%i)

		# deep component
		input_size = 4*self.first_level_lstm_hs[-1] + 4*self.tb_dense_hs + 2*self.total_num_filters
		glorot = np.sqrt(2.0 / (input_size + self.deep_dense[0]))
		weights["deep_dense_weight_0"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_dense[0])), dtype=np.float32)
		weights["deep_dense_bias_0"] = tf.Variable(
			np.random.normal(loc=0, scale=glorot, size=(1, self.deep_dense[0])), dtype=np.float32)
		for i in range(1, len(self.deep_dense)):
			glorot = np.sqrt(2.0 / (self.deep_dense[i-1] + self.deep_dense[i]))
			weights["deep_dense_weight_%d" % i] = tf.Variable(
				np.random.normal(loc=0, scale=glorot, size=(self.deep_dense[i-1], self.deep_dense[i])),
				dtype=np.float32)  # layers[i-1] * layers[i]
			weights["deep_dense_bias_%d" % i] = tf.Variable(
				np.random.normal(loc=0, scale=glorot, size=(1, self.deep_dense[i])),
				dtype=np.float32)  # 1 * layer[i]

		# final concat projection layer
		input_size = self.second_level_lstm_hs[-1] + 10
		glorot = np.sqrt(2.0 / (input_size + 1))
		weights["fc_weight"] = tf.Variable(
						np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
						dtype=np.float32)  # input_size*1
		weights["fc_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

		return weights

	def prelu_layer(self, x, name="prelu"):
		with tf.variable_scope(name):
			alphas = tf.get_variable("alpha", x.shape[-1], initializer=tf.constant_initializer(0.25),
									regularizer=tf.contrib.layers.l2_regularizer(1.0), dtype=tf.float32)
			pos = tf.nn.relu(x)
			neg = tf.multiply(alphas,(x-abs(x))*0.5)
			return pos + neg

	def batch_norm_layer(self, x, bn_decay, train_phase, scope_bn):
		bn_train = batch_norm(x, decay=bn_decay, center=True, scale=True, updates_collections=None,
							  is_training=True, reuse=None, trainable=True, scope=scope_bn)
		bn_inference = batch_norm(x, decay=bn_decay, center=True, scale=True, updates_collections=None,
								  is_training=False, reuse=True, trainable=True, scope=scope_bn)
		z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
		return z

	def get_batch(self, darray, batch_size, index, get_label = True):
		start = index * batch_size
		end = (index+1) * batch_size
		end = end if end < len(darray) else len(darray)
		if get_label :
			X = darray[start:end, :-11] # None * (field_size*sen_size)
			y = darray[start:end, -11] # None
			P = darray[start:end, -11:] # None * 10
			return X, P, np.reshape(y,[-1,1]) # return must be 2-d matrix
		else:
			X = darray[start:end, :-10]
			P = parray[start:end, -10:]
			return X, P, None


	# shuffle three lists simutaneously
	def shuffle_in_unison_scary(self, a):
		rng_state = np.random.get_state()
		np.random.shuffle(a)


	def fit(self, dTrain, dValid, early_stopping=False, refit=False):
		
		has_valid = False if dValid is None else True
		try:
			for epoch in range(1,self.epoch+1):
				t1 = time()
				self.shuffle_in_unison_scary(dTrain)
				total_batch = int(len(dTrain) / self.batch_size)
				with self.graph.as_default(): # 学习率每两个epoch衰减一次
					self.learning_rate = tf.train.exponential_decay(self.learning_rate, epoch, 2, decay_rate=0.95, staircase=True)
				for i in range(total_batch):
					X_batch, P_batch, y_batch = self.get_batch(dTrain, self.batch_size, i)
					feed_dict = {self.vocab_index: X_batch,
								 self.props: P_batch,
								 self.label: y_batch,
								 self.first_level_lstm_dropout_p: self.first_level_lstm_dropout,
								 self.deep_dropout_p: self.deep_dropout,
								 self.conv_pool_dropout_p: self.conv_pool_dropout,
								 self.second_level_lstm_dropout_p: self.second_level_lstm_dropout,
								 self.train_phase: True}
					learning_rate, loss, _ = self.sess.run((self.learning_rate, self.loss, self.optimizer), feed_dict=feed_dict)
					print("learning_rate:{}\tepoch:{}\tbatch:{}\tbatch_size:{}\tloss:{}".format(learning_rate,epoch,i,len(y_batch),loss))
				'''
				tlist_tra = random.sample(range(len(dTrain)),min(len(dTrain),50000))
				a1, p1, r1, f11 = self.evaluate(dTrain[tlist_tra], pTrain[tlist_tra])
				if has_valid:
					a2, p2, r2, f12 = self.evaluate(dValid, pValid)
					logging.warning("epoch[%d] time=[%.1f s]" % (epoch, time()-t1))
					logging.warning("train50k-result=[accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f]" % (a1, p1, r1, f11))
					logging.warning("valid-result=[accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f]" % (a2, p2, r2, f12))
				else:
					logging.warning("epoch[%d] time=[%.1f s]" % (epoch, time()-t1))
					logging.warning("train50k-result=[accuracy:%.4f precision:%.4f recall:%.4f f1:%.4f]" % (a1, p1, r1, f11))
				'''
				# save result
				if (self.verbose > 0 and epoch % self.verbose == 0) or epoch==self.epoch:
					ckpt_path = self.ckpt_path + "model_epoch_" + str(epoch) + ".ckpt"
					save_path = self.saver.save(self.sess, ckpt_path)
					logging.warning("Model saved in file: %s" % save_path)
				#if has_valid and early_stopping and self.training_termination(self.valid_result):
				#	break
		except tf.errors.ResourceExhaustedError:
			print("ResourceExhaustedError")
		finally:
			ckpt_path = self.ckpt_path + "model_epoch_new.ckpt"
			save_path = self.saver.save(self.sess, ckpt_path)
			logging.warning("Model saved in file: %s" % save_path)
		

	def training_termination(self, valid_result):
		if len(valid_result) > 5:
			if self.greater_is_better:
				if valid_result[-1] < valid_result[-2] and \
					valid_result[-2] < valid_result[-3] and \
					valid_result[-3] < valid_result[-4] and \
					valid_result[-4] < valid_result[-5]:
					return True
			else:
				if valid_result[-1] > valid_result[-2] and \
					valid_result[-2] > valid_result[-3] and \
					valid_result[-3] > valid_result[-4] and \
					valid_result[-4] > valid_result[-5]:
					return True
		return False


	def predict(self,darray):
		
		batch_index = 0
		X_batch, P_batch,  _ = self.get_batch(darray, self.batch_size, batch_index, get_label=False)
		y_pred = None
		while len(X_batch) > 0:
			num_batch = len(X_batch)
			feed_dict = {self.vocab_index: X_batch,
						 self.props: P_batch,
						 self.label: np.zeros([num_batch,1], dtype=float),
						 self.first_level_lstm_dropout_p: [1.0] * len(self.first_level_lstm_dropout),
						 self.deep_dropout_p: [1.0] * len(self.deep_dropout),
						 self.conv_pool_dropout_p: [1.0] * len(self.conv_pool_dropout),
						 self.second_level_lstm_dropout_p: [1.0] * len(self.second_level_lstm_dropout),
						 self.train_phase: False}
			batch_out = self.sess.run(self.out, feed_dict=feed_dict)

			if batch_index == 0:
				y_pred = np.reshape(batch_out, (num_batch,))
			else:
				y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

			batch_index += 1
			X_batch, P_batch, _ = self.get_batch(darray, self.batch_size, batch_index, get_label=False)

		return y_pred


	def evaluate(self, darray, thr):
		
		batch_index = 0
		X_batch, P_batch, y_batch = self.get_batch(darray, self.batch_size, batch_index)
		y_pred = None
		y_label = None
		while len(X_batch) > 0:
			num_batch = len(y_batch)
			feed_dict = {self.vocab_index: X_batch,
						 self.props: P_batch,
						 self.label: y_batch,
						 self.first_level_lstm_dropout_p: [1.0] * len(self.first_level_lstm_dropout),
						 self.deep_dropout_p: [1.0] * len(self.deep_dropout),
						 self.conv_pool_dropout_p: [1.0] * len(self.conv_pool_dropout),
						 self.second_level_lstm_dropout_p: [1.0] * len(self.second_level_lstm_dropout),
						 self.train_phase: False}
			batch_out = self.sess.run(self.out, feed_dict=feed_dict)

			if batch_index == 0:
				y_pred = np.reshape(batch_out, (num_batch,))
				y_label = np.reshape(y_batch, (num_batch,))
			else:
				y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
				y_label = np.concatenate((y_label, np.reshape(y_batch, (num_batch,))))

			batch_index += 1
			X_batch, P_batch, y_batch = self.get_batch(darray, self.batch_size, batch_index)

		pred = [1 if y_pred[i] > thr else 0 for i in range(len(y_pred))]
		accuracy = metrics.accuracy_score(y_label, pred)
		precision = metrics.precision_score(y_label, pred)
		recall = metrics.recall_score(y_label, pred)
		f1 = metrics.f1_score(y_label, pred)

		return accuracy,precision,recall,f1

