from collections import namedtuple
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

def split_valid_test_data(train, fraction=(1 - 0.8)):
    y_train = train['target']
    lb = preprocessing.LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    x_train = train.drop(['id', 'target'], axis=1)

    x_train, valid_x, y_train, valid_y = train_test_split(x_train, y_train, test_size=fraction)

    return x_train.values, y_train, valid_x, valid_y

def build_neural_network(X):
    hidden_units    =   10
    tf.reset_default_graph()
    inputs          =   tf.placeholder(tf.float32, shape=[None, X.shape[1]])
    labels          =   tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate   =   tf.placeholder(tf.float32)
    is_training     =   tf.Variable(True,dtype=tf.bool)
    
    initializer     =   tf.contrib.layers.xavier_initializer()
    
    fc              =   tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc              =   tf.layers.batch_normalization(fc, training=is_training)
    fc              =   tf.nn.relu(fc)
    
    logits          =   tf.layers.dense(fc, 1, activation=None)
    cross_entropy   =   tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost            =   tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer   =   tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted       =   tf.nn.sigmoid(logits)
    correct_pred    =   tf.equal(tf.round(predicted), labels)
    accuracy        =   tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes    =   ['inputs', 'labels', 'learning_rate','is_training', 'logits', 'cost', 'optimizer', 'predicted', 'accuracy']
    Graph           =   namedtuple('Graph', export_nodes)
    local_dict      =   locals()
    graph           =   Graph(*[local_dict[each] for each in export_nodes])

    return graph

def get_batch(data_x, data_y, batch_size=32):
    batch_n         =   len(data_x) //  batch_size
    
    for i in range(batch_n):
        batch_x     =   data_x[i * batch_size :(i + 1) * batch_size]
        batch_y     =   data_y[i * batch_size :(i + 1) * batch_size]
        yield batch_x,batch_y
