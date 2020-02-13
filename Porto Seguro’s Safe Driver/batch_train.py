from sklearn import metrics
import tensorflow as tf
import numpy as np
import data as dataset
import utils
import math

class ModelTraining(object):
    def __init__(self, MODEL, INPUT, LABEL, VAL_INPUT, VAL_LABEL, EPOCH, LEARNING_RATE, BATCH_SIZE):
        self.MODEL              =   MODEL
        self.INPUT              =   INPUT
        self.LABEL              =   LABEL
        self.VAL_INPUT          =   VAL_INPUT
        self.VAL_LABEL          =   VAL_LABEL
        self.EPOCH              =   EPOCH
        self.LEARNING_RATE      =   LEARNING_RATE
        self.BATCH_SIZE         =   BATCH_SIZE

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            BEST_LOSS = 999999.0
            sess.run(tf.global_variables_initializer())

            for e in range(self.EPOCH):
                if len(self.INPUT) < self.BATCH_SIZE:
                    self.BATCH_SIZE = len(self.INPUT)

                TRAIN_DATA  = dataset.DataSet(self.INPUT, self.LABEL)
                BATCH_TOTAL = math.ceil(len(self.INPUT) / self.BATCH_SIZE)
                print('Total Batch = {}, Batch Size = {}'.format(BATCH_TOTAL, self.BATCH_SIZE))
                
                for b in range(BATCH_TOTAL):
                    feed = {self.MODEL.inputs: self.INPUT, self.MODEL.labels: self.LABEL, 
                            self.MODEL.learning_rate: self.LEARNING_RATE, self.MODEL.is_training: True}
                    TRAIN_LOSS, _, TRAIN_ACC = sess.run([self.MODEL.cost, self.MODEL.optimizer, self.MODEL.accuracy], feed_dict = feed)
                    
                    if BEST_LOSS > TRAIN_LOSS:
                        BEST_LOSS = TRAIN_LOSS

                    if b % (BATCH_TOTAL // 5) == 0 :
                        print('Epoch [{0:2d}/{1:2d}] ---------------------------------- \n'.format(e + 1, self.EPOCH) +
                              'Batch [{0:4d}/{1:4d}]: Loss = {2:4.2f}, best Loss = {3:4.2f}'.format((b + 1), BATCH_TOTAL, TRAIN_LOSS, BEST_LOSS))
                        
                # VALIDATION
                valid_acc_list  = []
                pred_class_all  = []
                pred_class_all  = []
                total_vali_loss = []
                target_class_all= []

                VALID_DATA = dataset.DataSet(self.VAL_INPUT, self.VAL_LABEL)
                BATCH_VALID_TOTAL = math.ceil(len(self.VAL_INPUT) / self.BATCH_SIZE)
                
                for v in range(BATCH_VALID_TOTAL):
                    BATCH_VALID_IMAGES, BATCH_VALID_LABELS = VALID_DATA.next_batch(self.BATCH_SIZE)
                    feed = {self.MODEL.inputs: self.VAL_INPUT, self.MODEL.labels: self.VAL_LABEL, self.MODEL.is_training: False}
                    VAL_LOSS, VAL_ACC, VALID_CLASS = sess.run([self.MODEL.cost, self.MODEL.accuracy, self.MODEL.predicted], feed_dict=feed) 

                    valid_acc_list.append(VAL_ACC)
                    pred_class_all.extend(VALID_CLASS) 
                    total_vali_loss.append(VAL_LOSS)
                    target_class_all.extend(BATCH_VALID_LABELS)
                    
                    if v == BATCH_VALID_TOTAL - 1:
                        valid_acc_avg = np.mean(valid_acc_list)
                        print("Valid Accuracy Average: ", valid_acc_avg)                     
                        print('[Weight] Update.')
                                                
            saver.save(sess, "checkpoint/porto_pilsa.ckpt")

class Inference(object):
    def __init__(self, MODEL, TEST):
        self.MODEL              =   MODEL
        self.TEST               =   TEST

    def test(self):
        _restore = tf.train.Saver()
        with tf.Session() as sess:
            _restore.restore(sess, "checkpoint/porto_pilsa.ckpt")

            feed={self.MODEL.inputs: self.TEST, self.MODEL.is_training: False}
            TEST_PREDICT = sess.run(self.MODEL.predicted, feed_dict = feed)
        return TEST_PREDICT
