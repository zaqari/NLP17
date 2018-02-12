import ast
import re
import numpy as np
import pandas as pd
from nltk.parse.stanford import StanfordDependencyParser as sparse
pathmodelsjar = '~/stanford-english-corenlp-2016-01-10-models.jar'
pathjar = '~/stanford-parser/stanford-parser.jar'
depparse = sparse(path_to_jar=pathjar, path_to_models_jar=pathmodelsjar)

sent_corpus='~/NLP/Homework4/sents_corpus.txt'
df_in = pd.read_table(sent_corpus, sep='\t', names=['sent', 'check'], skipinitialspace=True, skiprows=0)

#####
##FUNCTIONS
#####

def dep_test(sentence):
	res = depparse.raw_parse(sentence)
	dep = res.__next__()
	ventral_stream = list(dep.triples())
	for tuple in ventral_stream:
		print(tuple)


def occipital(sentence, outlist, idx):
	v1=[]
	try:
		res = depparse.raw_parse(sentence)
		dep = res.__next__()
		ventral_stream = list(dep.triples())
		for tuple in ventral_stream:
			v1.append(tuple[0])
		for array in set(v1):
			if 'NN' in array[1]:
				a=[]
				a.append(array[0])
				for tuple in ventral_stream:
					if tuple[0]==array:
						a.append(tuple[2][0])
				a.append(idx)
				#print('=====ARRAY=====')
				#print(sentence)
				#print(a)
				#print('=====+++++=====')
				outlist.append(a)
	except OSError:
		print('Bloody hell, you dolt-minded crayon!')
	except AssertionError:
		print('Is it that hard to learn how to write a .csv file???')
	except UnicodeEncodeError:
		print('A pu ouela-ba angre?')
	except IndexError:
		print('THAT WAS NOT USEFUL!!!!')
	except UnicodeDecodeError:
		print('Confusing letters??')


def build_persist_data(inlist, outlist):
	for it in inlist:
		a=[]
		if len(it[:len(it)-3])<10:
			for word in it[:len(it)-1]:
				a.append(str(word))
			for k in range(len(a), 10):
				a.append('0')
			a.append(it[len(it)-1])
		elif len(it[:len(it)-3])>10:
			for word in it[:10]:
				a.append(str(word))
			a.append(it[len(it)-1])
		outlist.append(a) if len(a) == 11 else 0


def check_entity(array, checklist, outlist):
	listed=checklist[int(array[len(array)-1])-1]
	deps=[]
	BII=[]
	for itm in listed:
		deps.append(str(itm[0]))
		BII.append(itm[1])
	if array[0] in deps:
		if BII[deps.index(array[0])]!='O':
			array.append(1)
		else:
			array.append(0)
	outlist.append(array)
		


def build_token_checklist(inlist, outlist):
	a=[]
	for it in inlist:
		if it[0] in ['.', '?', '!']:
			a.append(it)
			outlist.append(a)
			a=[]
		else:
			a.append(it)

def build_data():
	for itm in sent_corpus:
		num_out=len(DNN_data)
		index=sent_corpus.index(itm)
		occipital(itm, nn_deps, index)
		build_persist_data(nn_deps, DNN_data)
		df_out=pd.DataFrame(np.array(DNN_data[num_out:]).reshape(-1, 11), columns=pred_feature_cols)
		df_out.to_csv(pred_deps, sep='\t', encoding='utf-8')
		print()
		print(str(sent_corpus.index(itm)/len(sent_corpus)*100), '% complete!')



def fix_missing_data(indeces):
	for k in redux:
		num_out=len(DNN_data)
		index=k
		occipital(sent_corpus[k], nn_deps, index)
		build_persist_data(nn_deps, DNN_data)
		df_out=pd.DataFrame(np.array(DNN_data[num_out:]).reshape(-1, 11), columns=pred_feature_cols)
		df_out.to_csv(redux_data, sep='\t', encoding='utf-8')
		print()
		print(str(redux.index(k)/len(redux)*100), '% complete!')



###############IMPORTS###################

#####
##IMPORT INPUT DOCUMENTS--BUILD DEPENDENCY DATA##
#####
sent_inputs='~/NLP/Homework4/pred_sentences.txt'
df_in = pd.read_table(sent_inputs, sep='\t', names=['sent', #check
                                                    ], skipinitialspace=True, skiprows=1)


#####
##IMPORT BII LABELING DOCUMENT##
#####
#text_f = '/Users/ZaqRosen/Documents/Corpora/NLP17/gene-trainF17.txt'
#COLUMNS=['#', 'word', 'ID']
#df_in1 = pd.read_table(text_f, sep='\t', names=COLUMNS, skipinitialspace=True, skiprows=0)




###############IMPLEMENTATION#############


#####
##IMPLEMENTATION##
#####
sent_corpus=df_in['sent'].values.tolist()
check_list=[]
#aba=list(zip(df_in1['word'].values.tolist(), df_in1['ID'].values.tolist()))
#build_token_checklist(aba, check_list)

###################BUILDS NO BII DATA ARRAYS#######################
#no_BII_label_data='~/NLP/Homework4/train_data_deps.txt'
#NOBII_COLS=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'SentID']
#out_feature_cols=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'SentID', 'BII']
pred_feature_cols=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'SentID']
pred_deps='~/NLP/Homework4/pred_data_deps.txt'
redux_data='~/NLP/Homework4/pred_data_deps_fuckup.txt'



nn_deps=[]
DNN_data=[]
print('STARTING ANALYSIS')
#print(str(len(sent_corpus)), ' to analyze.')
#build_data()

redux=[18, 72, 73, 111, 177, 178, 198, 219, 233, 251, 264, 334, 337, 380, 411, 514, 516, 522, 533, 536, 555, 557]
#fix_missing_data(redux)

print('FINISHED!')

#####################BUILDS IN BII LABELING##################
#df_in = pd.read_table(no_BII_label_data, sep='\t', names=NOBII_COLS, skipinitialspace=True, skiprows=0)

#no_zeros_data=[]
#for it in df_in.values.tolist()[1:]:
#	a=[]
#	for word in it:
#		if str(word)!='0':
#			if '-' in str(word) and '-' != str(word)[0] and '-' !=str(word)[len(str(word))-1]:
#				abacus=str(word).replace('-', ' - ').split()
#				for itm in abacus:
#					a.append(itm)
#			else:
#				a.append(str(word))
#	no_zeros_data.append(a)

#data_final=[]
#for itm in no_zeros_data:
#        check_entity(itm, check_list, data_final)

#data_true_final=[]
#for it in data_final:
#	a=[]
#	if len(it[:len(it)-3]) < 10:
#		for word in it[:len(it)-3]:
#			a.append(word)
#		for k in range(len(a), 10):
#			a.append('0')
#	elif len(it[:len(it)-3]) > 10:
#		for word in it[:10]:
#			a.append(word)
#	else:
#		for word in it[:len(it)-3]:
#			a.append(word)
#	a.append(it[len(it)-2]) 
#	a.append(it[len(it)-1])
#	data_true_final.append(a)

#final_outlist=[]
#counter=[]
#for it in data_true_final:
#        if it[len(it)-1] in [0, 1]:
#                final_outlist.append(it)
#        else:
#                counter.append(it)

#df_out=pd.DataFrame(np.array(final_outlist).reshape(-1, 12), columns=out_feature_cols)
#df_out.to_csv('~/NLP/Homework4/train_data.txt', sep='\t')

#test_train=[]
#for k in set(df_out['BII'].values.tolist()):
#	a=[]
#	for itm in df_out.values.tolist():
#		if itm[len(itm)-1]==k:
#			a.append(itm)
#	test_train.append(a)

#SUPER_TRAIN=[]
#SUPER_TEST=[]
#for a in test_train:
#	cut_off=int(len(a)*.85)
#	for it in a[:cut_off]:
#		SUPER_TRAIN.append(it)
#	for it in a[cut_off:]:
#		SUPER_TEST.append(it)

#df_out_train=pd.DataFrame(np.array(SUPER_TRAIN).reshape(-1, 12), columns=out_feature_cols)
#df_out_train.to_csv('~/NLP/Homework4/train_lil_data.txt', sep='\t')
#df_out_test=pd.DataFrame(np.array(SUPER_TEST).reshape(-1, 12), columns=out_feature_cols)
#df_out_test.to_csv('~/NLP/Homework4/test_lil_data.txt', sep='\t')
