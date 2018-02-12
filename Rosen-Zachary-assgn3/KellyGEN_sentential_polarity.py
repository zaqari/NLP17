import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np

print('TO DO: ')
print('(1) Create input fns for m2')
print('(2) Label inputs for m2 with POS or NEG')


###LOGGING SET-UP###
tf.logging.set_verbosity(tf.logging.INFO)

###SMALL DATASETS, NO INCLUSION OF ARG=XREF###
Train_Data2 = '~/NLP/Homework3/train_data.csv'
Test_Data2 = '~/NLP/Homework3/df_test_data.csv'
pred_data = '~/NLP/Homework3/pred_data.csv'

feature_columns = ['subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2']
DNN_COLUMNS = ['ID', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'Labels']
PREDCOLUMNS = ['ID', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'Labels']

###DATA INPUT BUILDER###
df_train1 = pd.read_csv(Train_Data2, names=DNN_COLUMNS, skipinitialspace=True)
df_test1 = pd.read_csv(Test_Data2, names=DNN_COLUMNS, skipinitialspace=True)

df_train = pd.concat([df_train1[feature_columns], df_train1['Labels'].astype(int)], axis=1, join='inner')
df_test = pd.concat([df_test1[feature_columns], df_test1['Labels'].astype(int)], axis=1, join='inner')
df_pred = pd.read_csv(pred_data, names=PREDCOLUMNS, skipinitialspace=False)

###SETTING CLASSES AND THE NUMBER OF DNN VARIABLES###
#It's not good enough to simply set the number of classes you might have. We
# really want to have some sort of automated and accurate way of rendering the
# number of classes we're looking at so we can avoid the mistake earlier of
# SAYING we have x-classes, and really having less than that (and thus having
# a loss-value of +5.4 . . . ).
##NOTE: the number of classes must always be n+1
nClasses= int(len(set(df_train['Labels'].values.tolist())))
#all_classes = list(set(df_train['Labels'].values.tolist()))
#S_V_hash_size = int(len(df_train))

###DNN SETUP###
#Herein lies the powerhouse of this system. The following establishes a DNN
# in which the edited information above is passed into a network classifier
# and is then classified to what its 'LmSource' value ought to be.

##NOTE ON GENERATING SOURCE DOMAIN CLASSES (9.19.17)
#It would be worthwhile to /group/ the source domain values in some meaningful
# way. I recommend the following technique to do such:
# (1) Generate a list of all the values for source tags via: df_in['LmSource']
#     .values.tolist() and print said list to a different document.
# (2) Using their vector representations to group them geometrically, or
#     manually grouping them all together.
# (3) Using these as a list to replace the values in the column with a numerical
#     token/classifier.

##NOTE: CHANGING CATEGORICAL_COLUMNS
#As of right now, the CATEGORICAL_COLUMNS will only call out to columns in df[].
# The problem is, the data is now in df_XREF. The solution, then, might lie in
# putting everything into df_XREF except the label column, and then running
# /those/ values through the feed function. The only difference, then, is
# including 'verb' and 'syn' in df_XREF.

CATEGORICAL_COLUMNS = ['subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2']
LABELS_COLUMN = ['Labels']

def input_fn(df):
        # Creates a dictionary mapping from each continuous feature column name (k) to
        # the values of that column stored in a constant Tensor.
        #continuous_cols = {k: tf.constant(df[k].values)
                                #  for k in CONTINUOUS_COLUMNS}
        # Creates a dictionary mapping from each categorical feature column name (k)
        # to the values of that column stored in a tf.SparseTensor.
        categorical_cols = {k: tf.SparseTensor(
                indices=[[i, 0] for i in range(df[k].size)],
                values=df[k].values,
                dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
        # Merges the two dictionaries into one.
        feature_cols = dict(categorical_cols.items())
        # Converts the label column into a constant Tensor.
        label = tf.constant(df['Labels'].values.astype(int))
        # Returns the feature columns and the label.
        return feature_cols, label

def train_input_fn():
        return input_fn(df_train)

def eval_input_fn():
        return input_fn(df_test)

def pred_input_fn():
        return input_fn(df_pred)

subj = tf.contrib.layers.sparse_column_with_hash_bucket("subj", hash_bucket_size=int(15000))

syn = tf.contrib.layers.sparse_column_with_hash_bucket("syn", hash_bucket_size=int(15000))

verb = tf.contrib.layers.sparse_column_with_hash_bucket("verb", hash_bucket_size=int(15000))

obl1 = tf.contrib.layers.sparse_column_with_hash_bucket("obl1", hash_bucket_size=int(15000))

obl2 = tf.contrib.layers.sparse_column_with_hash_bucket("obl2", hash_bucket_size=int(15000))

dobj = tf.contrib.layers.sparse_column_with_hash_bucket("dobj", hash_bucket_size=int(15000))

#This is set up to be realized as either the tempfile doc, or
# /tmp/KellyGEN
model_dir = '/tmp/KellyGEN'  #tempfile.mkdtemp()

subjxverb = tf.contrib.layers.crossed_column(
	[subj, verb],
	hash_bucket_size=int(1e6),
	combiner='sum')

subjxdobj = tf.contrib.layers.crossed_column(
	[subj, dobj],
	hash_bucket_size=int(1e6),
	combiner='sum')

dobjxverb = tf.contrib.layers.crossed_column(
	[dobj, verb],
	hash_bucket_size=int(1e6),
	combiner='sum')

dobjxobl1xobl2 = tf.contrib.layers.crossed_column(
	[dobj, obl1, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

###MOVEMENT FROM VERB FOCAL TO THETA-ROLE FOCAL MODEL
#The following are new features that could be used in the DNN in order to make
# decisions pertaining to categorization of examples. The percentage above each
# one indicates the total variance across the entiretyof the text. Ideally, DNN
# featues will be around 40-50%, with wide features being low (~10%). To date,
# subj and obj are the only features with that low a variance. The real paradigm
# shift lies in the realization that verb semantics are in fact less useful for
# classification than previoulsy thought with the temporal, toy-data set. From
# here on, we'll be focusing on what would traditionally be the theta-roles
# for categorical differentiation.

obl1xobl2 = tf.contrib.layers.crossed_column(
	[obl1, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

synxsubj = tf.contrib.layers.crossed_column(
	[syn, subj],
	hash_bucket_size=int(1e6),
	combiner='sum')

synxdobj = tf.contrib.layers.crossed_column(
	[syn, dobj],
	hash_bucket_size=int(1e6),
	combiner='sum')

synxobl1 = tf.contrib.layers.crossed_column(
	[syn, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

subjxobl1 = tf.contrib.layers.crossed_column(
	[subj, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')

subjxobl2 = tf.contrib.layers.crossed_column(
	[subj, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')

dobjxobl1 = tf.contrib.layers.crossed_column(
	[dobj, obl1],
	hash_bucket_size=int(1e6),
	combiner='sum')


dobjxobl2 = tf.contrib.layers.crossed_column(
	[dobj, obl2],
	hash_bucket_size=int(1e6),
	combiner='sum')



###VALIDATION MONITORING:
#The following are the metrics and set-up for the validation monitor such that
# we can track the progress of the system overtime using Tensorboard.

wide_collumns = []

deep_columns = [
        tf.contrib.layers.embedding_column(dobjxobl1xobl2, dimension=5),
        tf.contrib.layers.embedding_column(verb, dimension=5),
        tf.contrib.layers.embedding_column(subjxverb, dimension=5),
        tf.contrib.layers.embedding_column(dobjxverb, dimension=5),
        tf.contrib.layers.embedding_column(dobjxobl1, dimension=5),
        tf.contrib.layers.embedding_column(subjxobl1, dimension=5),
        tf.contrib.layers.embedding_column(obl1xobl2, dimension=5),
        ]


validation_metrics = {
        #The below is the best bet to run accuracy in here, but we need to
        # somehow run labels as a full-blown tensor of some sort.
#        'acc': m.evaluate(input_fn=eval_input_fn, steps=2)
#        'accuracy': tf.contrib.learn.MetricSpec(
#                metric_fn=tf.contrib.metrics.streaming_accuracy,
#                prediction_key=tf.contrib.learn.PredictionKey.
#                CLASSES),
#        'precision':tf.contrib.learn.MetricSpec(
#                metric_fn=tf.contrib.metrics.streaming_precision,
#                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
#        'recall': tf.contrib.learn.MetricSpec(
#                metric_fn=tf.contrib.metrics.streaming_recall,
#                prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
        }

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        df_test[feature_columns].values,
        df_test['Labels'].values,
        every_n_steps=50,
        #metrics=validation_metrics
        )

m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_collumns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[7, 10],
        n_classes=nClasses,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10),
        fix_global_step_increment_bug=True
        )

#results = m.evaluate(input_fn=eval_input_fn, steps=200)

#print(results)

####
##MAKING M2 DATA STRUCTURES
####

def pred_simple(listin, outlist):
	for it in listin:
		pos=it.count('pos')
		neg=it.count('neg')
		if pos > neg:
			outlist.append('POS')
		elif pos < neg:
			outlist.append('NEG')
		elif pos == neg:
			outlist.append('POS')

def sort_train_doc(listin, outlist):
        for ID in set(df_train1['ID'].values.tolist()):
                m2_features=[]
                label=''
                for pred in listin:
                        if ID==pred[0]:
                                m2_features.append(pred[1])
                                if label == '':
                                        label=pred[2]
                outlist.append([m2_features, label])

def m2_training_data(outlist):
        predictions = m.predict_classes(input_fn=train_input_fn)
        for pred in predictions:
                outlist.append(pred)

def paddington(listin, outlist):
        for FS in listin:
                out_stack=[]
                fin_list=[]
                if len(FS[0]) < 40:
                        for k in range(len(FS[0]), 40):
                                FS[0].append(int(-1))                      
                for x in FS[0]:
                        out_stack.append(x)
                for it in out_stack:
                        if it==0:
                                fin_list.append(str('pos'))
                        elif it==1:
                                fin_list.append(str('neg'))
                        elif it==int(-1):
                                fin_list.append(str('null'))
                fin_list.append(FS[1])
                outlist.append(fin_list)


m2_inputs=[]
def m2tbuild(ouy):
        m2_train=[]
        m2_training_data(m2_train)
        trainxdoc=list(zip(df_train1['ID'].values.tolist(), m2_train, df_train['Labels'].values.tolist()))
        m2_trainFS=[]
        sort_train_doc(trainxdoc, m2_trainFS)
        paddington(m2_trainFS, ouy)

####
##MAKING PREDICTIONS
####

def predictions_collection(outlist):
        predictions = m.predict_classes(input_fn=pred_input_fn)
        for pred in predictions:
                outlist.append(pred)

def sort_pred_doc(listin, outlist):
        for ID in set(df_pred['ID'].values.tolist()):
                m2_features=[]
                for pred in listin:
                        if ID==pred[0]:
                                m2_features.append(pred[1])
                outlist.append(m2_features)

def pred_paddington(listin, outlist):
        for FS in listin:
                out_stack=[]
                fin_list=[]
                if len(FS) < 40:
                        for k in range(len(FS), 40):
                                FS.append(int(-1))                      
                for x in FS:
                        out_stack.append(x)
                for it in out_stack:
                        if it==0:
                                fin_list.append(str('pos'))
                        elif it==1:
                                fin_list.append(str('neg'))
                        elif it==int(-1):
                                fin_list.append(str('null'))
                outlist.append(fin_list)

###F(X) REALIZATION
def predbuild(ouk):
        pred_set=[]
        predictions_collection(pred_set)
        predxdoc=list(zip(df_pred['ID'].values.tolist(), pred_set))
        m2FS=[]
        sort_pred_doc(predxdoc, m2FS)
        pred_paddington(m2FS, ouk)

predictions_x=[]

############DNN2


#########

####
##CREATE M2 COLUMNS
####

m2cols=[]
for it in range(40):
        m2cols.append(str(it))

m2cols_labels=[]
for it in m2cols:
        m2cols_labels.append(it)
m2cols_labels.append('Labels')

##########


m2_CATEGORICAL_COLUMNS = []
for it in m2cols:
        m2_CATEGORICAL_COLUMNS.append(str(it))

LABELS_COLUMN = ['Labels']

def m2_input_fn(df):
        # Creates a dictionary mapping from each continuous feature column name (k) to
        # the values of that column stored in a constant Tensor.
        #continuous_cols = {k: tf.constant(df[k].values)
        #                          for k in CONTINUOUS_COLUMNS}
        # Creates a dictionary mapping from each categorical feature column name (k)
        # to the values of that column stored in a tf.SparseTensor.
        categorical_cols = {k: tf.SparseTensor(
                indices=[[i, 0] for i in range(df[k].size)],
                values=df[k].values,
                dense_shape=[df[k].size, 1])
                        for k in m2_CATEGORICAL_COLUMNS}
        # Merges the two dictionaries into one.
        feature_cols = dict(categorical_cols.items())
        # Converts the label column into a constant Tensor.
        label = tf.constant(df['Labels'].values.astype(int))
        # Returns the feature columns and the label.
        return feature_cols, label

def m2_train_input_fn():
        return m2_input_fn(m2_train)

def m2_eval_input_fn():
        return m2_input_fn(df_test)

def m2_pred_input_fn():
        return m2_input_fn(m2_pred)

m1=tf.contrib.layers.sparse_column_with_hash_bucket('0', hash_bucket_size=int(3))
m2=tf.contrib.layers.sparse_column_with_hash_bucket('1', hash_bucket_size=int(3))
m3=tf.contrib.layers.sparse_column_with_hash_bucket('2', hash_bucket_size=int(3))
m4=tf.contrib.layers.sparse_column_with_hash_bucket('3', hash_bucket_size=int(3))
m5=tf.contrib.layers.sparse_column_with_hash_bucket('4', hash_bucket_size=int(3))
m6=tf.contrib.layers.sparse_column_with_hash_bucket('5', hash_bucket_size=int(3))
m7=tf.contrib.layers.sparse_column_with_hash_bucket('6', hash_bucket_size=int(3))
m8=tf.contrib.layers.sparse_column_with_hash_bucket('7', hash_bucket_size=int(3))
m9=tf.contrib.layers.sparse_column_with_hash_bucket('8', hash_bucket_size=int(3))
m10=tf.contrib.layers.sparse_column_with_hash_bucket('9', hash_bucket_size=int(3))
m11=tf.contrib.layers.sparse_column_with_hash_bucket('10', hash_bucket_size=int(3))
m12=tf.contrib.layers.sparse_column_with_hash_bucket('11', hash_bucket_size=int(3))
m13=tf.contrib.layers.sparse_column_with_hash_bucket('12', hash_bucket_size=int(3))
m14=tf.contrib.layers.sparse_column_with_hash_bucket('13', hash_bucket_size=int(3))
m15=tf.contrib.layers.sparse_column_with_hash_bucket('14', hash_bucket_size=int(3))
m16=tf.contrib.layers.sparse_column_with_hash_bucket('15', hash_bucket_size=int(3))
m17=tf.contrib.layers.sparse_column_with_hash_bucket('16', hash_bucket_size=int(3))
m18=tf.contrib.layers.sparse_column_with_hash_bucket('17', hash_bucket_size=int(3))
m19=tf.contrib.layers.sparse_column_with_hash_bucket('18', hash_bucket_size=int(3))
m20=tf.contrib.layers.sparse_column_with_hash_bucket('19', hash_bucket_size=int(3))
m21=tf.contrib.layers.sparse_column_with_hash_bucket('20', hash_bucket_size=int(3))
m22=tf.contrib.layers.sparse_column_with_hash_bucket('21', hash_bucket_size=int(3))
m23=tf.contrib.layers.sparse_column_with_hash_bucket('22', hash_bucket_size=int(3))
m24=tf.contrib.layers.sparse_column_with_hash_bucket('23', hash_bucket_size=int(3))
m25=tf.contrib.layers.sparse_column_with_hash_bucket('24', hash_bucket_size=int(3))
m26=tf.contrib.layers.sparse_column_with_hash_bucket('25', hash_bucket_size=int(3))
m27=tf.contrib.layers.sparse_column_with_hash_bucket('26', hash_bucket_size=int(3))
m28=tf.contrib.layers.sparse_column_with_hash_bucket('27', hash_bucket_size=int(3))
m29=tf.contrib.layers.sparse_column_with_hash_bucket('28', hash_bucket_size=int(3))
m30=tf.contrib.layers.sparse_column_with_hash_bucket('29', hash_bucket_size=int(3))
m31=tf.contrib.layers.sparse_column_with_hash_bucket('30', hash_bucket_size=int(3))
m32=tf.contrib.layers.sparse_column_with_hash_bucket('31', hash_bucket_size=int(3))
m33=tf.contrib.layers.sparse_column_with_hash_bucket('32', hash_bucket_size=int(3))
m34=tf.contrib.layers.sparse_column_with_hash_bucket('33', hash_bucket_size=int(3))
m35=tf.contrib.layers.sparse_column_with_hash_bucket('34', hash_bucket_size=int(3))
m36=tf.contrib.layers.sparse_column_with_hash_bucket('35', hash_bucket_size=int(3))
m37=tf.contrib.layers.sparse_column_with_hash_bucket('36', hash_bucket_size=int(3))
m38=tf.contrib.layers.sparse_column_with_hash_bucket('37', hash_bucket_size=int(3))
m39=tf.contrib.layers.sparse_column_with_hash_bucket('38', hash_bucket_size=int(3))
m40=tf.contrib.layers.sparse_column_with_hash_bucket('39', hash_bucket_size=int(3))


wide_columns2=[]

deep_columns2=[
        tf.contrib.layers.embedding_column(m1, dimension=5),
        tf.contrib.layers.embedding_column(m2, dimension=5),
        tf.contrib.layers.embedding_column(m3, dimension=5),
        tf.contrib.layers.embedding_column(m4, dimension=5),
        tf.contrib.layers.embedding_column(m5, dimension=5),
        tf.contrib.layers.embedding_column(m6, dimension=5),
        tf.contrib.layers.embedding_column(m7, dimension=5),
        tf.contrib.layers.embedding_column(m8, dimension=5),
        tf.contrib.layers.embedding_column(m9, dimension=5),
        tf.contrib.layers.embedding_column(m10, dimension=5),
        tf.contrib.layers.embedding_column(m11, dimension=5),
        tf.contrib.layers.embedding_column(m12, dimension=5),
        tf.contrib.layers.embedding_column(m13, dimension=5),
        tf.contrib.layers.embedding_column(m14, dimension=5),
        tf.contrib.layers.embedding_column(m15, dimension=5),
        tf.contrib.layers.embedding_column(m16, dimension=5),
        tf.contrib.layers.embedding_column(m17, dimension=5),
        tf.contrib.layers.embedding_column(m18, dimension=5),
        tf.contrib.layers.embedding_column(m19, dimension=5),
        tf.contrib.layers.embedding_column(m20, dimension=5),
        tf.contrib.layers.embedding_column(m21, dimension=5),
        tf.contrib.layers.embedding_column(m22, dimension=5),
        tf.contrib.layers.embedding_column(m23, dimension=5),
        tf.contrib.layers.embedding_column(m24, dimension=5),
        tf.contrib.layers.embedding_column(m25, dimension=5),
        tf.contrib.layers.embedding_column(m26, dimension=5),
        tf.contrib.layers.embedding_column(m27, dimension=5),
        tf.contrib.layers.embedding_column(m28, dimension=5),
        tf.contrib.layers.embedding_column(m29, dimension=5),
        tf.contrib.layers.embedding_column(m30, dimension=5),
        tf.contrib.layers.embedding_column(m31, dimension=5),
        tf.contrib.layers.embedding_column(m32, dimension=5),
        tf.contrib.layers.embedding_column(m33, dimension=5),
        tf.contrib.layers.embedding_column(m34, dimension=5),
        tf.contrib.layers.embedding_column(m35, dimension=5),
        tf.contrib.layers.embedding_column(m36, dimension=5),
        tf.contrib.layers.embedding_column(m37, dimension=5),
        tf.contrib.layers.embedding_column(m38, dimension=5),
        tf.contrib.layers.embedding_column(m39, dimension=5),
        tf.contrib.layers.embedding_column(m40, dimension=5),
        ]

model_m2_dir = '/tmp/KellyGEN_RNN'

m2 = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_m2_dir,
        linear_feature_columns=wide_columns2,
        dnn_feature_columns=deep_columns2,
        dnn_hidden_units=[11, 20],
        n_classes=nClasses,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10),
        fix_global_step_increment_bug=True
        )



##############FINAL REALIZTION!!!!!!
m.fit(input_fn=train_input_fn, steps=int(len(df_train)*4))
m2tbuild(m2_inputs)
m2_train = pd.DataFrame(np.array(m2_inputs).reshape(-1, 41), columns=m2cols_labels)
m2.fit(input_fn=m2_train_input_fn, steps=int(len(m2_train)*4))
predbuild(predictions_x)       
#pred_pred(pred_inputs, predictions_x)
m2_pred = pd.DataFrame(np.array(predictions_x).reshape(-1, 40), columns=m2cols)
m2_pred['Labels']=np.nan
final_answers = m2.predict_classes(input_fn=m2_pred_input_fn)
fuckimdone=[]
for pred in final_answers:
	fuckimdone.append(pred)
dgaf = pd.read_table('~/Documents/Corpora/NLP17/hw3_testset.txt', sep='\t', names=['ID', 'text'], skipinitialspace=True, skiprows=0)
final_output = list(zip(dgaf['ID'].values.tolist(), fuckimdone))


def all_done(listin, export_f):
	f = open(export_f, 'a')
	for it in listin:
		if it[1]==0:
			freaky = it[0] + '\t' + 'POS' + '\n'
			f.write(freaky)
		if it[1]==1:
			freaky = it[0] + '\t' + 'NEG' +'\n'
			f.write(freaky)
	f.close()

PRAGGLEJAZ = '~/Documents/HW3.txt'
all_done(final_output, PRAGGLEJAZ)
