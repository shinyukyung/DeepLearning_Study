import tensorflow as tf

def mlp_model(features, labels, mode, params):

    config = params

    # Dense Layers
    hidden1     =   tf.layers.dense(inputs=features, units=config['n_hidden1'], activation=tf.nn.relu)
    training    =   tf.placeholder_with_default(False, shape=(), name='training')
    bn1         =   tf.layers.batch_normalization(hidden1, momentum = 0.9)
    drop_h1     =   tf.layers.dropout(inputs=bn1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    hidden2     =   tf.layers.dense(inputs=drop_h1, units=config['n_hidden2'], activation=tf.nn.relu)
    drop_h2     =   tf.layers.dropout(inputs=hidden2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    hidden3     =   tf.layers.dense(inputs=drop_h2, units=config['n_hidden3'], activation=tf.nn.relu)
    drop_h3     =   tf.layers.dropout(inputs=hidden3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits      =   tf.layers.dense(inputs=drop_h3, units=config['nclasses'], activation=tf.nn.sigmoid)

    predictions = {"classes": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=config['nclasses'])
    print(onehot_labels)
    print(logits)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)