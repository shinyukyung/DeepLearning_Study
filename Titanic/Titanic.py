import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data
import utils
import tensorflow as tf
from sklearn.metrics import accuracy_score

epochs              =   200
train_collect       =   50
train_print         =   train_collect*2
learning_rate_value =   0.0005
batch_size          =   8
x_collect           =   []
train_loss_collect  =   []
train_acc_collect   =   []
valid_loss_collect  =   []
valid_acc_collect   =   []

nan_columns         =   ["Age", "SibSp", "Parch"]
not_concerned_col   =   ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]
dummy_columns       =   ["Pclass"]

#data
train_data          =   pd.read_csv("input/train.csv")
test_data           =   pd.read_csv("input/test.csv")

train_data          =   data.nan_padding(train_data, nan_columns)
test_data           =   data.nan_padding(test_data, nan_columns)

test_passenger_id   =   test_data["PassengerId"]

train_data          =   data.drop_not_concerned(train_data, not_concerned_col)
test_data           =   data.drop_not_concerned(test_data, not_concerned_col)

train_data          =   data.dummy_data(train_data, dummy_columns)
test_data           =   data.dummy_data(test_data, dummy_columns)

train_data          =   data.sex_to_int(train_data)
test_data           =   data.sex_to_int(test_data)

train_data          =   data.normalize_age(train_data)
test_data           =   data.normalize_age(test_data)

gender_submission   =   pd.read_csv("input/gender_submission.csv")
test_label          =   gender_submission["Survived"]

train_x, train_y, valid_x, valid_y = data.split_valid_test_data(train_data)

#train
model = utils.build_neural_network(train_x)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration=100
    for e in range(epochs):
        for batch_x,batch_y in utils.get_batch(train_x,train_y,batch_size):
            iteration+=1
            feed = {model.inputs: train_x, model.labels: train_y, model.learning_rate: learning_rate_value, model.is_training: True}

            train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
            
            if iteration % train_collect == 0:
                x_collect.append(e)
                train_loss_collect.append(train_loss)
                train_acc_collect.append(train_acc)

                if iteration % train_print==0:
                     print("Epoch: {}/{}".format(e + 1, epochs), "Train Loss: {:.4f}".format(train_loss), "Train Acc: {:.4f}".format(train_acc))
                        
                feed = {model.inputs: valid_x, model.labels: valid_y, model.is_training: False}
                val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                valid_loss_collect.append(val_loss)
                valid_acc_collect.append(val_acc)
                
                if iteration % train_print==0:
                    print("Epoch: {}/{}".format(e + 1, epochs), "Validation Loss: {:.4f}".format(val_loss), "Validation Acc: {:.4f}".format(val_acc))
                
    saver.save(sess, "checkpoint/titanic.ckpt")

#test
model   = utils.build_neural_network(test_data)
restorer=tf.train.Saver()

with tf.Session() as sess:
    restorer.restore(sess, "checkpoint/titanic.ckpt")
    feed={
        model.inputs: test_data, model.is_training: False}
    test_predict=sess.run(model.predicted,feed_dict=feed)
    
from sklearn.preprocessing import Binarizer

binarizer           =   Binarizer(0.5)
test_predict_result =   binarizer.fit_transform(test_predict)
test_predict_result  =   test_predict_result.astype(np.int32)

passenger_id        =   test_passenger_id.copy()
evaluation          =   passenger_id.to_frame()
evaluation["Survived"]=test_predict_result
evaluation.to_csv("input/evaluation_submission.csv",index=False)


print("\n\n\n\n\n***************************\nAccuary: " + 
      str(accuracy_score(test_label, test_predict_result)) + 
      "\n***************************\n\n\n\n\n")