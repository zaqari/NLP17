

###A PRIORI
#The current system is incomplete. Start by collecting negative and positive
# examples from the texts the last two functions!


import numpy as np
import pandas as pd

###INPUT FILES
#The first file, corpus_file refers to training data directory being used by
# the system, while the second, test_file, refers to the test data file
# directory
pos_f = '~/Corpora/NLP17/hotelPosT-train.txt'
neg_f = '~/Corpora/NLP17/hotelNegT-train.txt'

INCOLUMNS = ['ID', 'text']

#The following create two pandas data frames for both the pos and neg data.
df_pos = pd.read_table(pos_f, sep='\t', names=INCOLUMNS, skipinitialspace=True, skiprows=0)
df_neg = pd.read_table(neg_f, sep='\t', names=INCOLUMNS, skipinitialspace=True, skiprows=0)

neg_only = []
pos_only = []


#.replace('\,'.replace('\.', '').replace('\?', '').replace('\"', '').replace('\'', '').replace('\!', '').replace('\:', '').replace('\;', '')
get_rid=['\'', '\"', '\.', '\!', '\,', '\?', '\:', '\;', '\)', '\(', '\{', '\}', '\[', '\]']

def lexeme_dif():
    neg_lexeme=[]
    pos_lexeme=[]
    for itm in df_pos['text'].values.tolist():
        for ida in get_rid:
            a = itm.replace(ida, '')
            b = a.split()
            for its in b:
                pos_lexeme.append(its)
    for itm in df_neg['text'].values.tolist():
        for ida in get_rid:
            a = itm.replace(ida, '')
            b = a.split()
            for its in b:
                neg_lexeme.append(its)
    odontoceti = set(pos_lexeme)
    baleinoceti = set(neg_lexeme)
    for itm in list(odontoceti-baleinoceti):
        pos_only.append(itm)
    for itm in list(baleinoceti-odontoceti):
        neg_only.append(itm)

lexeme_dif()


###
######
###DEPENDENCY/CONSTRUCTION BUILDER
######
###

#imports Stanford’s Dependency Parser and sets up environment.
from nltk.parse.stanford import StanfordDependencyParser as sparse
pathmodelsjar = '/Users/ZaqRosen/nltk_data/stanford-english-corenlp-2016-01-10-models.jar'
pathjar = '/Users/ZaqRosen/nltk_data/stanford-parser/stanford-parser.jar'
depparse = sparse(path_to_jar=pathjar, path_to_models_jar=pathmodelsjar)

#This will be autodefined to 'nsubj' or 'dobj', but the point is to find
# any local deps that correlate to the sentence headers, essentially--those
# bits that more or less establish the relationship of the verb. The POS#
# .append protocol is effectively a fail-safe in case you end up analyzing
# a sub-structure lacking the 'nsubj' and 'dobj' components.
def litc(POS1, POS2, listy):
	for tuple in listy:
		POS1.append(tuple[2][0]) if 'nsubj' in tuple[1] else 0
		POS2.append(tuple[2][0]) if 'dobj' in tuple[1] else 0
	POS1.append('0')
	POS2.append('0')

#We know that WHAT the dep tree looks like is important--is it just an ADJ.P?
# or is the search item part of the entire clausal unit? This'll get to the
# bottom of these questions.
def sylvan(lmtc, pfc):
	for tuple in lmtc:
		pfc.append(tuple[1])

#Oblique elements appear to be vitally important in metaphor and constructional
# analyses. This block will thus derive the oblique elements' structure in
# simplified terms, or establish an adjective relationship of some sort to
# analyze.
def obl(oblique, lmtc, ventral_stream):
	oblique.append((0,0))
	for tuple in lmtc:
		oblique.append((tuple[2][0], tuple[0][0])) if 'JJ' in tuple[2][1] else 0
		if 'mod' in tuple[1]:
			ph_head = tuple[2]
			for tuple in ventral_stream:
				oblique.append((tuple[2][0], ph_head[0])) if tuple[1]=='case' and tuple[0]==ph_head else 0

def print_protocol( a, b, c, d):
	e = [a, b, c, d]
	for item in e:
		print('')
		print(item)
		print('=========')
			
#Where the magic happens, this links everything up into a coherent chunk that
# can then be passed to another function later in order to utilize the const.
# components, or print everything to a .csv via brocas().
###
#When running it for Thesis data collection, kill LmTarget by noting it out, as
# in corpus_callosum1 = [media, #LmTarget . . .
def corpus_callosum(ventral_stream, v1, lmtc, sentence, media, LmTarget, TEST, outl):
	oblique=[]
	pfc=[]
	NSUBJ=[]
	DOBJ=[]
	sylvan(lmtc, pfc)
	litc(NSUBJ, DOBJ, lmtc)
	obl(oblique, lmtc, ventral_stream)
	#print_protocol(oblique, pfc, NSUBJ, DOBJ)
	for item in oblique:
		corpus_callosum1 = [#media, #LmTarget,
			NSUBJ[0], DOBJ[0], str(pfc).replace(',', ' '), v1[0], item[0], item[1],
			#sentence
			]
		brocas(sentence, pfc, lmtc, corpus_callosum1, outl, TEST)
		corpus_callosum1 = []
		
#This little function simply packages and presents all the data collected in
# an easily interpretable chunk. If TEST=='build', it'll generate a .csv of
# the data the rest of the script collects.
def brocas(sentence, pfc, lmtc, array, out, TEST='non'):
	print('==============')
	if TEST == 'non':
		print(sentence)
		print('')
	if TEST == 'test':
		print('Grammatical Structure:')
		print(pfc)
		print(' ')
		print('Lexical Items:')
		print(lmtc)
		print(' ')
	#Builds the training data sheet for you, if selected.
	if TEST == 'build':
		out.append(array)
	print('Array: ')
	print(array)
	print('==============')

#Occipital(), named after the occipital lobe, is the integrative tissue
# that triggers the whole process, and links all the components together.
def occipital(sentence, search1, outlist, TEST='non', mediaitem='non'):
	#Resets triggers and data failsafes
	v1 = ''
	lmtc = []
	#components from Stanford's Dependency parser to create dep. tree.
	try:
		res = depparse.raw_parse(sentence)
		dep = res.__next__()
		ventral_stream = list(dep.triples())
		for tuple in ventral_stream:
			if search1 in tuple[2][0]:
				v1=tuple[0]
			elif search1 in tuple[0][0] and 'cop' in tuple[1]:
				v1=tuple[0]
			elif search1 in tuple[0][0] and 'neg' in tuple[1]:
				v1=tuple[0]
		for tuple in ventral_stream:
			if tuple[0]==v1:
				lmtc.append(tuple)
		corpus_callosum(ventral_stream, v1, lmtc, sentence, mediaitem, search1, TEST, outlist) if len(lmtc)>0 else print('Bloody hell, you dolt-minded crayon!')
	except OSError:
		print('Bloody hell, you dolt-minded crayon!')
	except AssertionError:
		print('Is it that hard to learn how to write a .csv file???')
	except UnicodeEncodeError:
		print('A pu ouela-ba n angre?')

OUTLIST=[]
def Training_Data_Builder(array):
    OUTLIST.append(array)
    

###
#####
###TRANSLATION INTO USABLE DISTINCTIONS
#####
###
REFs=['service', 'Service', 'hotel', 'Hotel', 'staff', 'Staff', 'sheets', 'Sheets', 'food', 'Food', 'room', 'Room', 'bed', 'Bed', 'expensive', 'Expensive', 'rate', 'Rate', 'breakfast', 'Breakfast', 'clean', 'Clean', 'dirty', 'Dirty', 'close']
#############################################

def splitty(sents, list_out):
        a = sents.split('. ')
        b = []
        for it in a:
                abba=it.split('! ')
                for itm in abba:
                        b.append(itm)
        c = []
        for it in b:
                amia=it.split('? ')
                for itm in amia:
                        c.append(itm)
        for it in c:
                list_out.append(it)

pos_reviews = list(zip(df_pos['ID'].values.tolist(), df_pos['text'].values.tolist()))
neg_reviews = list(zip(df_neg['ID'].values.tolist(), df_neg['text'].values.tolist()))

def data_constr(lista, listb):
        for it in lista:
                fracas = it[1].replace('Dr.', 'Dr').replace('Mrs.', 'Mrs').replace('Ms.', 'Ms').replace('St.', 'St')
                az=[]
                splitty(fracas, az)
                listb.append((it[0], az))

poss=[]
negs=[]
data_constr(pos_reviews, poss)
data_constr(neg_reviews, negs)

def pvc(inlist, freelist):
	memory=[]
	for uberitm in inlist:
		for sentence in uberitm[1]:
			for lexeme in REFs:
				if lexeme in sentence:
					occipital(sentence, lexeme, memory, 'build', uberitm[0])
		freelist.append((uberitm[0], memory))
		memory=[]

pos_c=[]
neg_c=[]
#pvc(poss, pos_c)
#pvc(negs, neg_c)

#############################################
def retain_data(listin, export_f):
        f = open(export_f, 'a')
        for it in listin:
                for array in it[1]:
                        freaky = it[0] + '\t' + str(array) +'\n'
                        f.write(freaky)
        f.close()

pos_export = '~/Corpora/NLP17/sentiRetain_pos.txt'
neg_export = '~/Corpora/NLP17/sentiRetain_neg.txt'

def data_packager(listin, listout):
        for it in listin:
                for array in it[1]:
                        array.insert(0, it[0])
                        listout.append(array)

####
##IF WORKING ON RETAINED DATA
####

#Use the following to rebuild the previous session via some training data

cols = ['ID', 'array']

df_pos2 = pd.read_table(pos_export, sep='\t', names=cols, skipinitialspace=False, skiprows=0)
df_neg2 = pd.read_table(neg_export, sep='\t', names=cols, skipinitialspace=False, skiprows=0)

#df_pos2 = pd.concat([df_EXpos['ID'], df_EXpos['array'].astype(list)], axis=1, join='inner')
#df_neg2 = pd.concat([df_EXneg['ID'], df_EXneg['array'].astype(list)], axis=1, join='inner')

def rebuild_old_session(dfi, outlist):
	abba=set(dfi['ID'].values.tolist())
	crunch = list(zip(dfi['ID'].values.tolist(), dfi['array'].values.tolist()))
	for it in abba:
		rebuilt_list=[]
		for itm in crunch:
			if it == itm[0]:
				rebuilt_list.append(itm[1])
		outlist.append((it, rebuilt_list))

pos_sess=[]
neg_sess=[]
rebuild_old_session(df_pos2, pos_sess)
rebuild_old_session(df_neg2, neg_sess)

print('all data previously held in pos_c and neg_c are now fully integrated in df_pos and df_neg.')

def rebuild_train_data(dfk, outlist):
	dragma=[]
	for it in dfk['array'].values.tolist():
		dragma.append(it)
	for it in dragma:
		a=it.replace('[', '').replace(']', '').replace('\'', '').replace('\"', '')
		b=a.split(', ')
		outlist.append(b)

posFrame_array=[]
negFrame_array=[]
rebuild_train_data(df_pos2, posFrame_array)
rebuild_train_data(df_neg2, negFrame_array)

COLUMNS=['nsubj', 'dobj', 'syn', 'verb', 'obl1', 'obl2']
df_pos3 = pd.DataFrame(np.array(posFrame_array).reshape(-1, 6), columns=COLUMNS)
df_neg3 = pd.DataFrame(np.array(negFrame_array).reshape(-1, 6), columns=COLUMNS)

df_negL=[]
df_posL=[]

def build_labels(dfi, outlist, num=int(0)):
	for it in range(0, int(len(dfi)-1)):
		outlist.append(num)

build_labels(df_pos3, df_posL, 0)
build_labels(df_neg3, df_negL, 1)

df_pos = pd.concat([df_pos2['ID'], df_pos3, pd.DataFrame(np.array(df_posL).reshape(-1, 1), columns=['label'])], axis=1, join='inner')
df_neg = pd.concat([df_neg2['ID'], df_neg3, pd.DataFrame(np.array(df_negL).reshape(-1, 1), columns=['label'])], axis=1, join='inner')

####
##TRAINING DATA EXPORT
####

final_export_f = '~/NLP/Homework3/train_data.csv'

import csv
import codecs

def training_data_builder(array, fileOut):
	with codecs.open(fileOut, 'a', 'utf-8') as csvfile:
		databuilder = csv.writer(csvfile, delimiter=',',
				quotechar='|',
				quoting=csv.QUOTE_MINIMAL)
		databuilder.writerow(array)
	csvfile.close()

def build_train_data():	
        for it in df_pos.values.tolist():
                training_data_builder(it, final_export_f)
        for it in df_neg.values.tolist():
                training_data_builder(it, final_export_f)

#######OVERFN

def build_pred_data(dfp):
        outCOLUMNS=['ID', 'nsubj', 'dobj', 'syn', 'verb', 'obl1', 'obl2']
        final_predExport_f = '~/NLP/Homework3/pred_data.csv'
        pos_reviews = pos_reviews = list(zip(dfp['ID'].values.tolist(), dfp['text'].values.tolist()))
        poss=[]
        data_constr(pos_reviews, poss)
        pos_c=[]
        pvc(poss, pos_c)
        arraylist=[]
        data_packager(pos_c, arraylist)
        df_out = pd.DataFrame(np.array(arraylist).reshape(-1, 7), columns=outCOLUMNS)
        for it in df_out.values.tolist():
            training_data_builder(it, final_predExport_f)

####
##BUILD PREDICTION DATA
####
#All items in this must be unstarred to create prediction data!!!

