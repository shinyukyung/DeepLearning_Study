import tensorflow as tf
import utils

class ModelTraining(object):
    def __init__(self, MODEL, INPUT, LABEL, VAL_INPUT, VAL_LABEL, EPOCH, LEARNING_RATE, BATCH_SIZE):
        self.MODEL          =   MODEL
        self.INPUT          =   INPUT
        self.LABEL          =   LABEL
        self.VAL_INPUT      =   VAL_INPUT
        self.VAL_LABEL      =   VAL_LABEL
        self.EPOCH          =   EPOCH
        self.LEARNING_RATE  =   LEARNING_RATE
        self.BATCH_SIZE     =   BATCH_SIZE
     
    def train(self):
        NUM_EPOCH           =   []
        self.TRAIN_COLLECT  =   50
        self.TRAIN_PRINT    =   self.TRAIN_COLLECT * 2

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iter = 100
            for e in range(self.EPOCH):
                for batch_x, batch_y in utils.get_batch(self.INPUT,  self.LABEL, self.BATCH_SIZE):
                    iter += 1
                    feed = {self.MODEL.inputs: self.INPUT, self.MODEL.labels: self.LABEL, 
                            self.MODEL.learning_rate: self.LEARNING_RATE, self.MODEL.is_training: True}
                    TRAIN_LOSS, _, TRAIN_ACC = sess.run([self.MODEL.cost, self.MODEL.optimizer, self.MODEL.accuracy], feed_dict = feed)
            
                    if iter % self.TRAIN_COLLECT == 0:
                        NUM_EPOCH.append(e)
                        
                        if iter % self.TRAIN_PRINT == 0:
                            print("Epoch: {}/{}".format(e + 1, self.EPOCH), 
                                    "Train Loss: {:.4f}".format(TRAIN_LOSS), 
                                    "Train Accuracy: {:.4f}".format(TRAIN_ACC))
                        feed = {self.MODEL.inputs: self.VAL_INPUT, self.MODEL.labels: self.VAL_LABEL, 
                                self.MODEL.is_training: False}
                        VAL_LOSS, VAL_ACC = sess.run([self.MODEL.cost, self.MODEL.accuracy], feed_dict = feed)
                        
                        if iter % self.TRAIN_PRINT == 0:
                            print("Epoch: {}/{}".format(e + 1, self.EPOCH),
                                  "Validation Loss: {:.4f}".format(VAL_LOSS),
                                  "Validation Accuracy: {:.4f}".format(VAL_ACC))    

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
