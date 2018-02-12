import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np

###LOGGING SET-UP###
tf.logging.set_verbosity(tf.logging.INFO)

###SMALL DATASETS, NO INCLUSION OF ARG=XREF###
train_data = '~/NLP/Homework4/train_data.txt'
test_data = '~/NLP/Homework4/test_lil_data.txt'
pred_data = '~/NLP/Homework4/pred_data.txt'

DNN_COLUMNS=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'SentID', 'label']
PRED_COLUMNS=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'SentID']
feature_COLUMNS=['head', '1', '2', '3', '4', '5', '6', '7', '8', '9']

df_train=pd.read_table(train_data, sep='\t', names=DNN_COLUMNS, skipinitialspace=True, skiprows=2)
df_test=pd.read_table(test_data, sep='\t', names=DNN_COLUMNS, skipinitialspace=True, skiprows=2)
df_pred=pd.read_table(pred_data, sep='\t', names=PRED_COLUMNS, skipinitialspace=True, skiprows=1)
df_pred['label'] = np.nan

nClasses= int(len(set(df_train['label'].values.tolist())))

CATEGORICAL_COLUMNS=feature_COLUMNS
LABELS_COLUMN=['label']

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
        label = tf.constant(df[LABELS_COLUMN].values.astype(int))
        # Returns the feature columns and the label.
        return feature_cols, label

def train_input_fn():
        return input_fn(df_train)

def eval_input_fn():
        return input_fn(df_test)

def pred_input_fn():
        return input_fn(df_pred)

head = tf.contrib.layers.sparse_column_with_hash_bucket("head", hash_bucket_size=int(15000))

w1 = tf.contrib.layers.sparse_column_with_hash_bucket("1", hash_bucket_size=int(15000))

w2 = tf.contrib.layers.sparse_column_with_hash_bucket("2", hash_bucket_size=int(15000))

w3 = tf.contrib.layers.sparse_column_with_hash_bucket("3", hash_bucket_size=int(15000))

w4 = tf.contrib.layers.sparse_column_with_hash_bucket("4", hash_bucket_size=int(15000))

w5 = tf.contrib.layers.sparse_column_with_hash_bucket("5", hash_bucket_size=int(15000))

w6 = tf.contrib.layers.sparse_column_with_hash_bucket("6", hash_bucket_size=int(15000))

w7 = tf.contrib.layers.sparse_column_with_hash_bucket("7", hash_bucket_size=int(15000))

w8 = tf.contrib.layers.sparse_column_with_hash_bucket("8", hash_bucket_size=int(15000))

w9 = tf.contrib.layers.sparse_column_with_hash_bucket("9", hash_bucket_size=int(15000))

#This is set up to be realized as either the tempfile doc, or /tmp/KellyGEN
model_dir = '/tmp/KellyENT'  #tempfile.mkdtemp()

headx1 = tf.contrib.layers.crossed_column(
	[head, w1],
	hash_bucket_size=int(1e6),
	combiner='sum')

headx2 = tf.contrib.layers.crossed_column(
	[head, w2],
	hash_bucket_size=int(1e6),
	combiner='sum')

headx3 = tf.contrib.layers.crossed_column(
	[head, w3],
	hash_bucket_size=int(1e6),
	combiner='sum')

headx4 = tf.contrib.layers.crossed_column(
	[head, w4],
	hash_bucket_size=int(1e6),
	combiner='sum')

headx5 = tf.contrib.layers.crossed_column(
	[head, w5],
	hash_bucket_size=int(1e6),
	combiner='sum')

headx6 = tf.contrib.layers.crossed_column(
	[head, w6],
	hash_bucket_size=int(1e6),
	combiner='sum')

headx7 = tf.contrib.layers.crossed_column(
	[head, w7],
	hash_bucket_size=int(1e6),
	combiner='sum')

headx8 = tf.contrib.layers.crossed_column(
	[head, w8],
	hash_bucket_size=int(1e6),
	combiner='sum')

headx9 = tf.contrib.layers.crossed_column(
	[head, w9],
	hash_bucket_size=int(1e6),
	combiner='sum')

###VALIDATION MONITORING:
#The following are the metrics and set-up for the validation monitor such that
# we can track the progress of the system overtime using Tensorboard.

wide_collumns = []

deep_columns = [
        tf.contrib.layers.embedding_column(headx1, dimension=5),
        tf.contrib.layers.embedding_column(headx2, dimension=5),
        tf.contrib.layers.embedding_column(headx3, dimension=5),
        tf.contrib.layers.embedding_column(headx4, dimension=5),
        tf.contrib.layers.embedding_column(headx5, dimension=5),
        tf.contrib.layers.embedding_column(headx6, dimension=5),
        tf.contrib.layers.embedding_column(headx7, dimension=5),
        tf.contrib.layers.embedding_column(headx8, dimension=5),
        tf.contrib.layers.embedding_column(headx9, dimension=5),
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
        df_test[feature_COLUMNS].values,
        df_test['label'].values,
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

#m.fit(input_fn=train_input_fn, steps=int(len(df_train)*4))

#results = m.evaluate(input_fn=eval_input_fn, steps=200)

#print(results)

#################### MAKE PREDICTIONS ########################
def predictions_collection(outlist):
        predictions = m.predict_classes(input_fn=pred_input_fn)
        for pred in predictions:
                outlist.append(pred)

predictions=[]
predictions_collection(predictions)

out_data=pd.DataFrame(np.array(predictions).reshape(-1, 1), columns=['label'])
outbound_bus=pd.concat([df_pred[PRED_COLUMNS], out_data], axis=1, join='inner')
pred_entity_doc='~/NLP/Homework4/pred_plus.txt'
outbound_bus.to_csv(pred_entity_doc, sep='\t', encoding='utf-8')
