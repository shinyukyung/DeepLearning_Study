import tensorflow as tf
import utils

class ModelTraining(object):
    def __init__(self, Model, Input, Label, Val_Input, Val_Label, Epoch, Train_Collect, Train_Print, Learning_Rate, Batch_Size):
        self.Model              =   Model
        self.X_train            =   Input
        self.Label              =   Label
        self.Val_Input          =   Val_Input
        self.Val_Label          =   Val_Label
        self.Epoch              =   Epoch
        self.Train_Collect      =   Train_Collect
        self.Train_Print        =   Train_Print
        self.Learning_Rate      =   Learning_Rate
        self.Batch_Size         =   Batch_Size
     
    def train(self):
        x_collect               =   []
        train_loss_collect      =   []
        train_acc_collect       =   []
        valid_loss_collect      =   []
        valid_acc_collect       =   []

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 100
            for e in range(self.Epoch):
                for batch_x, batch_y in utils.get_batch(self.X_train, self.Label, self.Batch_Size):
                    iteration += 1
                    feed = {self.Model.inputs: self.X_train, self.Model.labels: self.Label, 
                            self.Model.learning_rate: self.Learning_Rate, self.Model.is_training: True}

                    train_loss, _, train_acc = sess.run([self.Model.cost, self.Model.optimizer, self.Model.accuracy], feed_dict=feed)
            
                    if iteration % self.Train_Collect == 0:
                        x_collect.append(e)
                        train_loss_collect.append(train_loss)
                        train_acc_collect.append(train_acc)

                        if iteration % self.Train_Print == 0:
                            print("epoch: {}/{}".format(e + 1, self.Epoch), 
                                  "train loss: {:.4f}".format(train_loss), 
                                  "train acc: {:.4f}".format(train_acc))
                        
                        feed = {self.Model.inputs: self.Val_Input, self.Model.labels: self.Val_Label, self.Model.is_training: False}
                        val_loss, val_acc = sess.run([self.Model.cost, self.Model.accuracy], feed_dict=feed)
                        valid_loss_collect.append(val_loss)
                        valid_acc_collect.append(val_acc)
                
                        if iteration % self.Train_Print == 0:
                            print("epoch: {}/{}".format(e + 1, self.Epoch), "validation loss: {:.4f}".format(val_loss), "validation acc: {:.4f}".format(val_acc))    
            saver.save(sess, "checkpoint/porto_pilsa.ckpt")


class Inference(object):
    def __init__(self, Model, Test):
        self.Model              =   Model
        self.Test               =   Test

    def test(self):
        restorer = tf.train.Saver()
        with tf.Session() as sess:
            restorer.restore(sess, "checkpoint/porto_pilsa.ckpt")
            feed={self.Model.inputs: self.Test, self.Model.is_training: False}
            test_predict=sess.run(self.Model.predicted, feed_dict=feed)
        return test_predict
