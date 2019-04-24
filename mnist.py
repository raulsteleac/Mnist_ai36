# %% Change working directory from the workspace root to the ipynb file location. 
# import os
# try:
#     os.chdir(os.path.join(
#         os.getcwd(), 'mnist_ann'))
#     print(os.getcwd())
# except:
#     pass

import random
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

class Data_Loader(object):
    def __init__(self, batch_size, x_train, y_train, x_test, y_test):
        self._batch_size = batch_size
        self._batch_nr_train = len(x_train) // batch_size
        self._batch_nr_test = len(x_test)

        print("Numbers of Batches for Training : %d" % self._batch_nr_train)
        print("Numbers of Batches for Testing : %d" % self._batch_nr_test)

        y_train = np.array([self.one_hot_encoder(y) for y in y_train])
        y_test = np.array([self.one_hot_encoder(y) for y in y_test])

        self._x_train = tf.cast(tf.convert_to_tensor(self.compute_batches(x_train,
                                                                self._batch_nr_train, 
                                                                self._batch_size,
                                                                784)), tf.float32)
        self._y_train = tf.cast(tf.convert_to_tensor(self.compute_batches(y_train,
                                                                self._batch_nr_train, 
                                                                self._batch_size,
                                                                10)), tf.float32)

        self._x_test = tf.cast(tf.convert_to_tensor(self.compute_batches(x_test, 
                                                                1, 
                                                                self._batch_nr_test, 
                                                                784)), tf.float32)

        self._y_test = tf.cast(tf.convert_to_tensor(self.compute_batches(y_test,
                                                                1,
                                                                self._batch_nr_test,
                                                                10)), tf.float32)
    def one_hot_encoder(self, val):
        one_hot = np.zeros(10)
        one_hot[val] = 1
        return one_hot

    def compute_batches(self, data, batch_nr, batch_size, dimension):
        return data[0: self._batch_nr_train *
                    self._batch_size].reshape((batch_nr, batch_size, dimension))

    def create_queue(self, session, X, Y,  batch_nr):
        with tf.name_scope("Input_Queues"):
            queue = tf.FIFOQueue(capacity=batch_nr / 2, dtypes=[tf.int32])
            enqueue_op = queue.enqueue_many([[j for j in range(batch_nr)]])
            i = queue.dequeue()
            qr = tf.train.QueueRunner(queue=queue, enqueue_ops=[enqueue_op] * 2)
        return X[i], Y[i], qr

    def get_data(self, session):
        with tf.name_scope("Mnist_Inputs_TRAIN_Generator"):
            x_train, y_train, qr_train = self.create_queue(session, self._x_train, self._y_train, self._batch_nr_train)

        with tf.name_scope("Mnist_Inputs_TEST_Generator"):
            x_test, y_test = (self._x_test, self._y_test)
                
            self._coord_train= tf.train.Coordinator()
            self._threads_train = qr_train.create_threads(session, self._coord_train, start=True)

        return x_train, y_train, x_test[0], y_test[0]
    
    def get_batch_numbers(self):
        return self._batch_nr_train, self._batch_nr_test

    def free_threads(self):
        self._coord_train.request_stop()
        self._coord_train.join(self._threads_train)

class Mnist_ANN(object):
    def __init__(self, config, session):
        #Initialise variables with config values
        self._batch_size = config.batch_size
        self._learning_rate = config.learning_rate
        self._pkeep = config.pkeep
        self._train_slice = config.training_slice
        self._test_slice = config.test_slice
        self._validation_slice = config.validation_slice

        #Get data batches
        self._data_loader = Data_Loader(self._batch_size, config.x_train, config.y_train,
                                                            config.x_test, config.y_test)
        self.x_train, self.y_train, self.x_test, self.y_test = self._data_loader.get_data(session)
        self._batch_nr_train, self._batch_nr_test = self._data_loader.get_batch_numbers()
        self.init = tf.random_uniform_initializer(-0.1, 0.1)

        #Initialise variables with training values
        self._data_slice = self._train_slice
        self._batch_nr = self._batch_nr_train
        self.input = self.x_train
        self.output = self.y_train
        self._is_training = True

    def initialize_variables(self, session):
        session.run(tf.global_variables_initializer())

    def reinit_optimizer(self, session):
        var_list = self.opt.variables()
        session.run(tf.variables_initializer(var_list))

    def model(self):
        with tf.variable_scope("Mnist_ANN", reuse=tf.AUTO_REUSE, initializer=self.init):            
            with tf.name_scope("Layer_1"):
                Y1 = tf.nn.relu(self.fully_connected(self.input, 784, 400, "FC_Layer_1"))

            with tf.name_scope("Layer_2"):  
                Y2 = tf.nn.relu(self.fully_connected(Y1, 400, 200, "FC_Layer_2"))

            with tf.name_scope("Layer_3"):
                Y3 = tf.nn.relu(self.fully_connected(Y2, 200, 100, "FC_Layer_3"))

            with tf.name_scope("Layer_4"):
                Y4=tf.nn.relu(self.fully_connected(Y3, 100, 60, "FC_Layer_4"))

            with tf.name_scope("Layer_5"):
                Y5 = tf.nn.relu(self.fully_connected(Y4, 60, 30, "FC_Layer_5"))

            with tf.name_scope("Out_layer"):
                logits = self.fully_connected(Y5, 30, 10, "OutPut_Layer")
                Y_ = tf.nn.softmax(logits)
            
            self.cross_en = tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=logits)
            self.is_correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(Y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))
            tf.summary.scalar('Accuracy', self.accuracy)

            print("IS the model Training : %r" %self._is_training)
            if not self._is_training:
                return

            self.opt = tf.train.AdamOptimizer(self._learning_rate)
            self.optimizer = self.opt.minimize(self.cross_en)
            #self.optimizer = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self.cross_en)
        return

    def fully_connected(self, input_data, channels_in, neurons_per_layer, name = "Fully_Connected_Layer"):
        #print(input_data.shape)
        W = tf.get_variable(name + "_Weights", dtype=tf.float32, shape=[channels_in, neurons_per_layer])
        b = tf.get_variable(name + "_Biases",  dtype=tf.float32, shape=[neurons_per_layer])
        Y = tf.matmul(input_data, W) + b
        if self._is_training:
            Y = tf.nn.dropout(Y ,self._pkeep)
        return Y

    def config_training(self):
        self._data_slice = self._train_slice
        self._batch_nr = self._batch_nr_train
        self.input = self.x_train
        self.output = self.y_train
        self.string = "Train"
        self._is_training = True

    def config_validation(self):
        self._data_slice = self._validation_slice
        self._batch_nr = self._batch_nr_train
        self.input = self.x_train
        self.output = self.y_train
        self.string = "Validation"
        self._is_training = False

    def config_testing(self):
        self._data_slice = self._test_slice
        self._batch_nr = 1
        self.input = self.x_test
        self.output = self.y_test
        self.string = "Test"
        self._is_training = False

    def train(self):
        return {
            "accuracy": self.accuracy,
            "optimizer": self.optimizer
        }

    def valdiate(self):
        return {
            "accuracy": self.accuracy
        }

    def test(self):        
        return {
            "accuracy": self.accuracy
        }

    def run_model(self, operation, session):
        print("\n=============== %s" %self.string)
        accuracy = 0.0
        real_batch_nr = ((int)(self._batch_nr * self._data_slice))
        percentage = real_batch_nr // 10
        for step in range(real_batch_nr):
            vals, summary = session.run([operation, self.merged])
            accuracy += vals["accuracy"]    
            if not self._is_training:
                if step != 0 and step % percentage == 0:
                    print("-----------> Batch number : %d Current Accuracy : %f" % ((step / percentage),  vals["accuracy"]))
        self.writer.add_summary(summary)
        print("############### %s Accuracy = %lf \n" % (self.string, (accuracy / real_batch_nr)))

    def tensordboard_write(self, session):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./graphs', session.graph)

    def close_model(self):
        self._data_loader.free_threads()

class NormalConfig(object):
    batch_size = 64
    learning_rate = 0.001
    pkeep = 0.85
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    
    training_slice = 0.95
    validation_slice = 0.05
    test_slice = 1

#%%
def main():
    tf.reset_default_graph()
    epochs = 10
    with tf.Session() as ses:
        mn = Mnist_ANN(NormalConfig(), ses)
        mn.model()
        mn.initialize_variables(ses)
        mn.tensordboard_write(ses)

        for i in range(epochs):
            print("-------- Reached epoch : %d \n" %i)
            mn.config_training()
            mn.model()
            mn.reinit_optimizer(ses) # When this code is run using Gradient Decent this line should be commented. It's here because of the parameters, that the AdamOptimizer uses, which also need to be initialized whenever the model is instantiated (beta1 and beta2). 
            mn.run_model(mn.train(), ses)

            mn.config_validation()
            mn.model()
            mn.run_model(mn.valdiate(), ses)

        mn.config_testing()
        mn.model()
        mn.run_model(mn.test(), ses)
        mn.tensordboard_write(ses)

        mn.close_model()
        return

if __name__ == "__main__":
    main()