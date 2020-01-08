# -*- coding:utf-8 -*-

## 分词+生成词典
import codecs
import numpy as np
import jieba
from gensim.corpora import Dictionary

# 取出原始txt中的prefix, query_predicts, title, tag, label
def str2feat(line, has_label=True):
	field = line.split('\t')
	field_size = 5 if has_label else 4

	assert (len(field) == field_size)
	try:
		prefix = field[0]
		title = field[2]
		if has_label:
			label = int(field[4][0])
			tag = field[3]
		else:
			label = None
			tag = field[3][:-2]
		query_predicts = field[1]
		query_predicts = [] if query_predicts == '{}' else query_predicts[2:-2].split('", "')
		fullquerys, props = [], [] # 完整查询，统计概率
		for qp in query_predicts:
			t = qp.split('": "')
			fullquerys.append(t[0])
			props.append(float(t[1]))
		return prefix, np.array(fullquerys), np.array(props), title, tag, label
	except IndexError:
		return None, None, None, None, None, None

# 对一行数据进行分词
# 返回分词后的行数据和该行中的词列表
def segline(line, has_label, op_type="ci"):
	assert op_type in ["ci", "zi"], \
			"op_type can be either 'ci' or 'zi'"
			
	prefix, fullquerys, props, title, tag, label = str2feat(line, has_label)
	if prefix:
		n_line = ''
		if op_type == "ci":
			words_prefix = list(jieba.cut(prefix, cut_all=False))
			words_title = list(jieba.cut(title, cut_all=False))
			words_tag = list(jieba.cut(tag, cut_all=False))
			words_in_a_line = set(words_prefix)
			words_in_a_line.update(words_title) # 用add方法不行
			words_in_a_line.update(words_tag)
			words_fqs = []
			for fq in fullquerys:
				words = list(jieba.cut(fq, cut_all=False))
				words_fqs.append(words)
				words_in_a_line.update(words)
		else:
			words_prefix = list(prefix)
			words_title = list(title)
			words_tag = list(tag)
			words_in_a_line = set(words_prefix)
			words_in_a_line.update(words_title) # 用add方法不行
			words_in_a_line.update(words_tag)
			words_fqs = []
			for fq in fullquerys:
				words = list(fq)
				words_fqs.append(words)
				words_in_a_line.update(words)
		
		# 得到加入空格的新行，空格将每个字/词分开
		n_line += ' '.join(words_prefix) + '\t{"'
		for i in range(len(words_fqs)):
			words = words_fqs[i]
			n_line += ' '.join(words) + '": "' + str(props[i])+'", "'
		if len(fullquerys) > 0: # 如果fullquerys不为空，去掉最后一个逗号、空格和双引号
			n_line = n_line[:-3] + '}\t'
		else: # 否则，直接加'}\t'
			n_line += '}\t'
		n_line += ' '.join(words_title) + '\t'
		n_line += ' '.join(words_tag)
		if has_label:
			n_line += '\t' + str(label) + '\r\n'
		else:
			n_line += '\r\n'

		return n_line, words_in_a_line
	else:
		return None, None

# 数据预处理：分词、创建词典，将分词结果写入新文件
def rawtxt_to_segedtxt(rawFile,segedFile,has_label,op_type,update=True,dct=None):
	if update: # 一般地，对于有标签的数据，要把词加入词典，因为我想要词典里的词向量是能够训练的
		assert (dct is not None)
	fw = codecs.open(segedFile, 'w+', 'utf-8')
	with codecs.open(rawFile, 'r', 'utf-8') as fr:
		for line in fr:
			n_line, words_in_a_line  = segline(line, has_label, op_type)
			try:
				fw.write(n_line)
				if has_label:
					dct.add_documents([list(words_in_a_line)]) # 参数必须为[['我的','你'],...] double list类型
			except TypeError: # n_line... is None
				print("error in dealing the record:%s" % line)
				break
	fw.close()
	return

# 把词所在词典当中的序号代替词
# 词从1开始编号（即词典中的序号加1)
# 将每一行原始数据转化为相同维度的向量
# query_predict按统计概率降序排列
def line2array(line, cols, sen_size, DCT, has_label):
	if has_label:
		assert (cols==13*sen_size+1+10)
	else:
		assert (cols==13*sen_size+10)
	try:
		prefix, fullquerys, props, title, tag, label = str2feat(line, has_label=has_label)
		asc_idx = np.argsort(-props)
		fullquerys = fullquerys[asc_idx] # fullquerys and asc_idx must be np.array
		props = props[asc_idx]
		prefix, title, tag = prefix.split(' '), title.split(' '), tag.split(' ')
		fullquerys_ = []
		for i in range(len(fullquerys)):
			fullquerys_.append(fullquerys[i].split(' '))

		arra = np.zeros([cols], dtype=np.float32)
		t = min(len(prefix), sen_size)
		for i in range(t):
			try:
				arra[i] = DCT.token2id[prefix[i]] + 1 # 词典中的序号加1
			except KeyError: # 词典中没有的词用0代替
				continue

		t1 = min(len(fullquerys_), 10)
		for j in range(t1):
			fq = fullquerys_[j]
			t = min(len(fq), sen_size)
			for i in range(t):
				try:
					arra[(j + 1) * sen_size + i] = DCT.token2id[fq[i]] + 1
				except KeyError:
					continue

		t = min(len(title), sen_size)
		for i in range(t):
			try:
				arra[11 * sen_size + i] = DCT.token2id[title[i]] + 1
			except KeyError:
				continue

		t = min(len(tag), sen_size)
		for i in range(t):
			try:
				arra[12 * sen_size + i] = DCT.token2id[tag[i]] + 1
			except KeyError:
				continue

		arra[13*sen_size] = label
		
		t = min(len(props), 10)
		for i in range(t):
			arra[13*sen_size+i] = props[i]

		return arra

	except IndexError:
		print("IndexError:%s" % line)
		return None


def txt2array(segedFile, has_label, num_line, cols, sen_size, DCT, arrayPath):
	fr = codecs.open(segedFile, 'r', 'utf-8')
	c = 0
	i = 0
	arras = np.empty([num_line, cols], dtype=np.float32)
	for line in fr:
		arra = line2array(line, cols, sen_size, DCT, has_label)
		if arra is not None:
			arras[i] = arra
			c += 1
		else:
			print(i)
		i += 1
	fr.close()
	np.save(arrayPath, arras)
	print("i = {:d}\tc = {:d}".format(i, c))
	return

def dataPre(op_type="ci"):
	segedFiles = ['seg_'+op_type+'_train.txt','seg_'+op_type+'_valid.txt','seg_'+op_type+'_test_A.txt']
	arrayFiles = ['array_'+op_type+'_train.npy','array_'+op_type+'_valid.npy','array_'+op_type+'_test_A.npy']
	# 第一阶段：使用jieba分词并构造词典，为了避免第二阶段再次使用jieba进行分词，这里将分词后的数据保存为一个新的文件
	dct = Dictionary()
	rawtxt_to_segedtxt('oppo_round1_train_20180929.txt',
						segedFiles[0],
						True,op_type,True,dct)
	rawtxt_to_segedtxt('oppo_round1_vali_20180929.txt',
						segedFiles[1],
						True,op_type,True,dct)
	rawtxt_to_segedtxt('oppo_round1_test_A_20180929.txt',
						segedFiles[2],
						False,op_type,False,dct)
	dct.save('dct_ci.dict')
	print("分词完成、词典创建成功！")
	
	# 第二阶段：将txt文本数据转化为数据npy矩阵文件
	if op_type=="ci":
		length = 5 # 每个field的一般词长
	else:
		length = 8 # 每个field的一般字长
	txt2array(segedFile=segedFiles[0],
			  has_label=True,
			  num_line=2000000,
			  cols = 13*length+11,
			  sen_size=length,
			  DCT = dct,
			  arrayPath=arrayFiles[0])
	txt2array(segedFile=segedFiles[1],
			  has_label=True,
			  num_line=50000,
			  cols = 13*length+11,
			  sen_size=length,
			  DCT = dct,
			  arrayPath=arrayFiles[1])
	txt2array(segedFile=segedFiles[2],
			  has_label=False,
			  num_line=50000,
			  cols = 13*length+10,
			  sen_size=length,
			  DCT = dct,
			  arrayPath=arrayFiles[2])
			  
	return

if __name__ == '__main__':
	
	dataPre()

'''
## 生成词向量
from gensim.models import word2vec

fileSegWordDonePath = 'oppo_round1_segword.txt'
fileVectorsPath = 'wordvec.bin'

sentence = word2vec.Text8Corpus(fileSegWordDonePath)
model = word2vec.Word2Vec(sentence) # Word2Vec() -> Word2Vec().load_vab() -> Word2Vec().train()
model.save(fileVectorsPath
'''

