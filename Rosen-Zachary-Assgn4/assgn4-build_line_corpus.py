import numpy as np
import pandas as pd

def build_sents(df, outlista):
	a=[]
	for word in df['word'].values.tolist():
		if word in ['.', '?', '!']:
			a.append(word)
			outlista.append(a)
			a=[]
		else:
			a.append(word)
		

def build_checks(inlist, outlist):
	a=[]
	for it in inlist:
		if it[0] in ['.', '?', '!']:
			a.append((it[0], it[1]))
			outlist.append(a)
			a=[]
		else:
			a.append((it[0], it[1]))

def build_token_checklist(inlist, outlist):
	a=[]
	for it in inlist:
		if it[0] in ['.', '?', '!']:
			a.append(it)
			outlist.append(a)
			a=[]
		else:
			a.append(it)

def BoW_2_string(inlist, outlist):
	for sent in inlist:
		a=''
		for word in sent:
			if word in ['.', '?', '!']:
				a+=str(word)
			else:
				a+=str(word)
				a+=' '
		b=a.replace('( ', '(').replace(' )', ')').replace('[ ', '[').replace(' ]',']').replace(' - ', '-').replace('{ ', '{').replace(' }', '}').replace(' .', '.')
		outlist.append(b)


#####
##IMPORT FILE VIA PANDAS
#####
text_f =  '/Users/ZaqRosen/Documents/Corpora/NLP17/F17-assgn4-test.txt'
COLUMNS=['#', 'word', 'ID']
df_in1 = pd.read_table(text_f, sep='\t', names=COLUMNS, skipinitialspace=True, skiprows=0)



#####
##IMPLEMENTATION
#####
tok_corpus=[]
build_sents(df_in1, tok_corpus)

sent_corpus=[]
BoW_2_string(tok_corpus, sent_corpus)

sentsss=pd.DataFrame(np.array(sent_corpus).reshape(-1, 1), columns=['sent'])
sentsss.to_csv('~/NLP/Homework4/pred_sentences.txt', sep='\t', encoding='utf-8')

############BLOCK OUT BELOW THIS LINE IF BUILDING PREDICTION CORPUS#############
#aba=list(zip(df_in1['word'].values.tolist(), df_in1['ID'].values.tolist()))
#tok_checklist=[]
#build_token_checklist(aba, tok_checklist)

#checksss=pd.DataFrame(np.array(tok_checklist).reshape(-1, 1).astype(list), columns=['checklist'])

#df_persist_sent_corpus=pd.concat([sentsss, checksss], axis=1, join='inner')
#df_persist_sent_corpus.to_csv('/Users/ZaqRosen/Documents/computational_ling/class_work/NLP/Homework4/sents_corpus.txt', sep='\t', encoding='utf-8')
