import pandas as pd
import numpy as np

#####
##COLUMNS FOR ALL WORK##
#####
orig_COLUMNS=['IDX','word']
pred_entity_COLUMNS=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'SentID','label']
pred_deps_COLUMNS=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'SentID']
feature_COLUMNS=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9']


#####
##IMPORTS##
#####
orig_doc='~Corpora/NLP17/F17-assgn4-test.txt'
pred_entity_doc='~/NLP17/Homework4/pred_plus.txt'
pred_dep_doc = '~NLP17/Homework4/pred_data.txt'

df_orig = pd.read_table(orig_doc, sep='\t', names=orig_COLUMNS, skipinitialspace=True, skiprows=0)
df_entity = pd.read_table(pred_entity_doc, sep='\t', names=pred_entity_COLUMNS, skipinitialspace=True, skiprows=1)
df_deps = pd.read_table(pred_dep_doc, sep='\t', names=pred_deps_COLUMNS, skipinitialspace=True, skiprows=1)

#####
##FUNCTIONS##
#####

def build_sents(df, outlista):
	a=[]
	for word in df['word'].values.tolist():
		if word in ['.', '?', '!']:
			a.append(word)
			outlista.append(a)
			a=[]
		else:
			a.append(word)



def build_initial_entries(outlist):
        for sent in sent_corpus:
                d=[]
                b_word=[]
                b_mark=[]
                a=[]
                if sent_corpus.index(sent) in set(df_entity['SentID'].values.tolist()):
                        for it in df_entity.values.tolist():
                                if it[10]==sent_corpus.index(sent):
                                        d.append(it)
                elif sent_corpus.index(sent) not in set(df_entity['SentID'].values.tolist()):
                        fig=[]
                        for k in range(11):
                                fig.append('NULLRESULT')
                        fig.append(0)
                        d.append(fig)
                for itm in d:
                        for word in itm[:9]:
                                if word!='0':
                                        b_word.append(word)
                                        b_mark.append(itm[11])
                for word in sent:
                        if word in b_word:
                                a.append([word, 'I', sent_corpus.index(sent)]) if b_mark[b_word.index(word)]!=0 else a.append([word, 'O', sent_corpus.index(sent)])
                                b_word.remove(word)
                        else:
                                a.append([word, 'O', sent_corpus.index(sent)])
                for itm in a:
                        outlist.append(itm)


def build_final_outputs(inlist, outlist):
        for it in inlist:
                if it[1]=='I':
                        if inlist[inlist.index(it)-1][1]=='O':
                                outlist.append([it[0], 'B'])
                        else:
                                outlist.append([it[0], it[1]])
                else:
                        outlist.append([it[0], it[1]])


def fix_outputs(inlist, outlist):
        for it in inlist:
                if it not in ['O', 'B', 'I']:
                        outlist.append('O')
                else:
                        outlist.append(it)
#####
##IMPLEMENTATION##
#####
sent_corpus=[]
build_sents(df_orig, sent_corpus)

bubba=[]
build_initial_entries(bubba)

finns=[]
build_final_outputs(bubba, finns)

df_label=pd.DataFrame(np.array(finns).reshape(-1, 1), columns=['label'])
df_finns=pd.concat([df_orig['IDX'], df_orig['word'], df_label], axis=1, join='inner')

final_data=[]
fix_outputs(df_finns['label'].values.tolist(), final_data)
df_labels_finns=pd.DataFrame(np.array(final_data).reshape(-1, 1), columns=['label'])

df_final=pd.concat([df_orig['IDX'], df_orig['word'], df_labels_finns], axis=1, join='inner')
df_final.to_csv('~/NLP17/Rosen-Zachary-Assgn4-answers.txt', sep='\t', encoding='utf-8', header=False, index=False)
