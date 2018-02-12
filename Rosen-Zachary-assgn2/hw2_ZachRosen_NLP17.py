import pandas as pd
import numpy as np
import codecs

COLUMNS = ['line', 'word', 'pos']

###INPUT FILES
#The first file, corpus_file refers to training data directory being used by
# the system, while the second, test_file, refers to the test data file
# directory
corpus_file = '~/berp-POS-training.txt'
test_file = '~/assgn2-test-set.txt'

#The following create two pandas data frames for both the training and test data.
df_corpus = pd.read_table(corpus_file, sep='\t', names=COLUMNS, skipinitialspace=True, skiprows=0)
df_test = pd.read_table(test_file, sep='\t', names=COLUMNS, skipinitialspace=True, skiprows=0)

###TRAINING DATA SET-UP
#The components here create both the tokenized, intersectional corpus being used
# as well as establishing the functions that will be used to process this corpus
# build the probabilistic functions powering the POS tagger. These are all inte-
# grated into the over_fn3 function.

#Builds the tokenized, intersectional corpus.
tok_corpus=[]
def build_structured_corpora():
	out_list=[]
	words = df_corpus['word'].values.tolist()
	POS = df_corpus['pos'].values.tolist()
	for it in words:
		if it != '.':
			out_list.append((
				(it, words[words.index(it)-1]) if words.index(it) != 0 else (it, '$'),
				POS[words.index(it)]
				))
		if it == '.':
			out_list.append((
				(it, words[words.index(it)-1]),
				POS[words.index(it)]
				))
			tok_corpus.append(out_list)
			out_list=[]

#Appends final POS decision to the running list of POS tags.
def print_protocol(pos_decision):
    priors.append(pos_decision)

#This function collects all collocates where the word from the test data occurs,
# including the POS tag associated with test data word in that collocation.
def list_builder(wi, list1):
	struc_data=[]
	for sent in tok_corpus:
		for itm in sent:
			if wi == itm[0][0]:
				struc_data.append(itm)
	for it in set(struc_data):
		list1.append(it)

#This function takes in data from list_builder, and generates a minimal matrix
# of probabilities for the POS tag associated with the input word.
def p_posXwi(bigr, list1):
	local_pos_p=[]
	out_list=[]
	local_p_matrix=[]
	for it in list1:
		if bigr == it[0]:
			out_list.append(it[1])
	if len(out_list) == 0:
		win_p=float(0)
		win_pos=''
		#Step 1: CREATE LOCALIZED POS TAGS
		for it in list1:
			local_pos_p.append(it[1])
		#Step 2: CALCULATE PROPABILITY
		for itm in local_pos_p:
			numerator = local_pos_p.count(itm)
			denominator = len(local_pos_p)
			local_p_matrix.append( (itm, numerator/denominator) )
		for itm in local_p_matrix:
			if itm[1] > win_p:
				win_p = itm[1]
				win_pos=itm[0]
		out_list.append(win_pos)
	print_protocol(out_list[0])

#This function integrates the list_builder function with the p_posXwi function
def over_fn3(wordM, bigram):
	bigrm = bigram
	if wordM in df_corpus['word'].values.tolist():
		local_corpus=[]
		list_builder(wordM, local_corpus)
		p_posXwi(bigrm, local_corpus)
	elif wordM not in df_corpus['word'].values.tolist():
		print_protocol('NUNCALOS')

###INPUT FROM TEST_DATA
#The following create a corpus of sentences from the test data data-frame, and
# then allow us to feed those examples--token by token--into the over_fn3 fun-
# ction above.

#Generates a tokenized corpus, organized into sentences.
test_corpus=[]
def build_test_corpus():
    alpha = df_test['word'].values.tolist()
    alist_word=[]
    for itm in alpha:
        if itm != '.':
            alist_word.append(itm)
        if itm == '.':
            alist_word.append(itm)
            test_corpus.append(alist_word)
            alist_word=[]

#Feeds examples from the tokenized test corpus through the over_fn3 function.
def test_input():
        for itm in test_corpus:
                for it in itm:
                        if itm.index(it) ==0:
                                over_fn3(it, (it, '$'))
                        else:
                                over_fn3(it, (it, itm[itm.index(it)-1]))


###MANIPULATORY FUNCTIONS
#Fun fact, this system is not currently equipped to handle situations in which
# a word is not in the corpus dictionary . . . that's bad news bears. What we
# should do is create a sub-function for running probability if the lexical item
# ain't in there. Which we did, and named is catch_missing_pos.
def catch_missing_pos(pos1, pos2):
	POSs = df_corpus['pos'].values.tolist()
	previous_POS=[]
	following_POS=[]
	final_POS = ''
	fin_p = float(0)
	for itm in POSs:
		if itm == pos1:
			following_POS.append(POSs[POSs.index(itm)+1])
		elif itm == pos2:
			previous_POS.append(POSs[POSs.index(itm)-1])
	likelies = previous_POS + following_POS
	for its in set(likelies):
		numerator = likelies.count(its)
		if numerator/len(likelies) > fin_p:
			fin_p = numerator/len(likelies)
			final_POS = its
	final_print.append(final_POS)
	

###SUPER SIMPLIFIED INITIAL CALC
#I felt like I would be cheating if I didn't include these. These three combine
# all the prior elements to create a true, pure viterbi algorithm.
def quickie(wordf):
	listh=[]
	list_builder(wordf, listh)
	p_posXwi((wordf, '$'), listh)

def quick_input():
        for it in df_test['word'].values.tolist():
                if it in df_corpus['word'].values.tolist():
                        quickie(it)
                elif it not in df_corpus['word'].values.tolist():
                        print_protocol('NUNCALOS')

wallaby=[]
def quick_total():
        quick_input()
        final_scrub()
        quick_clean(wallaby)


###OUTPUT LISTS & FINAL SCRUBBING
#Priors is in fact a list of all examples the system is sure of, as well as a
# placeholder for those examples which it was not sure of. To solve the mystery
# of the missing tags, we'll run catch_missing_pos to fill in what gaps remain.
# the final product will be appended to final_print which can then be used to
# generate a new data frame.
priors = []
final_print=[]
        
def final_scrub():
    for itm in priors:
        if itm == 'NUNCALOS':
            POS_prior = priors[priors.index(itm)-1]
            POS_suiv = priors[priors.index(itm)+1]        
            catch_missing_pos(POS_prior, POS_suiv)
        elif itm != 'NUNCALOS':
            final_print.append(itm)

def quick_clean(outbound):
        a = df_test['line'].values.tolist()
        b = df_test['word'].values.tolist()
        alist=[]
        blist=[]
        clist=[]
        for itm in a:
                if itm == '1' and a.index(itm) != 0:
                        alist.append(' ')
                        alist.append(itm)
                else:
                        alist.append(itm)
        for its in b:
                if its != '.':
                        blist.append(its)
                elif its =='.':
                        blist.append(its)
                        blist.append(' ')
        for whale in final_print:
                if whale != '.':
                        clist.append(whale)
                elif whale =='.':
                        clist.append(whale)
                        clist.append(' ')
        odontoceti = list(zip(alist, blist, clist))
        for orca in odontoceti:
                outbound.append(orca)

def out_print():
	f = open('~/Desktop/zachrosen_assgn2.txt', 'a')
	for beluga in final_data:
		freaky = str(beluga[0]) + '\t' + str(beluga[1]) + '\t' + str(beluga[2] + '\n')
		f.write(freaky)
	f.close()

###RUNNING THE PROGRAM
#We start with running the data constructor, ranging from building our training
# and test inputs, prior to generating the POS tags with test_input, and finally
# running a final scrubbing of the data with final_scrub.
#Everything then gets passed to quick_clean to create a tab delimited set of
# outputs before printing the whole thing to a document.
build_structured_corpora()
build_test_corpus()
test_input()
final_scrub()

final_data=[]
quick_clean(final_data)
out_print()

#df_finalPOS = pd.DataFrame(np.array(final_print).reshape(-1, 1), columns=['pos'])

