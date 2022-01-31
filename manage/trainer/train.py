import keras
import tensorflow as tf
import os
from keras.regularizers import l1,l2
from keras import layers
from keras import activations
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from math import ceil
from argparse import ArgumentParser
from time import sleep

class DatasetCreator(object):
    def ImportDataset(self):
        #loading dataset from tensorflow repository
        return tf.keras.datasets.cifar10.load_data()
    
    def PreprocessData(self,x_tr,y_tr,x_tst,y_tst):
        #simple dataset preprocessing
        assert x_tr.shape == (50000, 32, 32,3)
        assert x_tst.shape == (10000, 32, 32,3)
        assert y_tr.shape == (50000,1)
        assert y_tst.shape == (10000,1)
        
        x_tr = x_tr/255.0
        x_tst = x_tst/255.0
        
        return x_tr,y_tr,x_tst,y_tst
    
    def PrepareSets(self):
        #creating training and testing sets
        (x_train, y_train), (x_test, y_test) = self.ImportDataset()
        x_train, y_train, x_test, y_test = self.PreprocessData(x_train, y_train, x_test, y_test)
        
        #splitting dataset accordingly
        x, x_val, y, y_val = train_test_split(x_train,y_train,test_size=0.2, random_state=42)
        
        return x, y, x_val, y_val, x_test, y_test

class Model(object):
    def __init__(self):
        #network parameters
        self.loss_fn  = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = Adam(learning_rate=0.01)
        self.kernel_regularizer = l2(1e-3)
        self.bias_regularizer = l1(1e-3)
        
        #training parameters
        self.batch_numb = 128
        self.epochs = 1
        self.metrics = ['accuracy']
        
    def nn(self):
        #model architecture that is being copied to every worker
        model = tf.keras.Sequential([
            layers.Flatten(input_shape=(32,32,3)),
            layers.Dense(300,activation='relu',kernel_regularizer=self.kernel_regularizer,bias_regularizer=self.bias_regularizer),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(500,activation='relu',kernel_regularizer=self.kernel_regularizer,bias_regularizer=self.bias_regularizer),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(700,activation='relu',kernel_regularizer=self.kernel_regularizer,bias_regularizer=self.bias_regularizer),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(30,activation='relu',kernel_regularizer=self.kernel_regularizer,bias_regularizer=self.bias_regularizer),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(10,activation='softmax',kernel_regularizer=self.kernel_regularizer,bias_regularizer=self.bias_regularizer),
        ])
        
        return model
    
    def _is_chief(self,cluster_resolver):
        task_type = cluster_resolver.task_type
        return task_type is None or task_type == 'chief'

    def _get_temp_dir(self,model_path, cluster_resolver):
        worker_temp = f'worker{cluster_resolver.task_id}_temp'
        return os.path.join(model_path, worker_temp)

    def save_model(self,model_path, model):
        
        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        is_chief = self._is_chief(cluster_resolver)

        if not is_chief:
            model_path = self._get_temp_dir(model_path, cluster_resolver)

        model.save(model_path)

        if is_chief:
            # wait for workers to delete; check every 100ms
            # if chief is finished, the training is done
            while tf.io.gfile.glob(os.path.join(model_path, "worker*")):
                sleep(0.1)

        if not is_chief:
            tf.io.gfile.rmtree(model_path)
    
    def train_function(self,x_train,y_train,x_val,y_val,architecture,spe):
        model = architecture
        
        #model compilation
        model.compile(loss=self.loss_fn,optimizer=self.optimizer,metrics=self.metrics)
        
        #reducing learning rate if training slows down
        reduce_learning_rt = ReduceLROnPlateau(monitor='val_loss',factor=0.001,patience=5,min_lr=0.001)
        
        history = model.fit(x_train,y_train,batch_size=self.batch_numb,epochs=self.epochs,steps_per_epoch=spe,callbacks=[reduce_learning_rt],validation_data=(x_val,y_val))
        
        return model, history.epoch, history.history

class DistributedTraining(Model):
    def __init__(self, path2save):
        super().__init__()
        self.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        self.number_of_replicas = self.strategy.num_replicas_in_sync
        
        self.global_batch = self.batch_numb * self.number_of_replicas
        
        self.model_path = path2save
        
    def multi_worker_training(self,x_training, y_training, x_validation, y_validation) :
        with self.strategy.scope():
            model = self.nn()
            
        spe = ceil(len(x_training) / self.global_batch) #steps per epoch
        trained_model, _,history  = self.train_function(x_training, y_training, x_validation, y_validation,model,spe)
        #trained_model.save(self.model_path+'/manage')
        #saving the model- task performed by "chief" machine
        self.save_model(self.model_path+"/saved_model",trained_model)

#main function

def main(args):
    trainer = DistributedTraining(args.job_dir)
    dataset = DatasetCreator()
    x_training, y_training, x_validation, y_validation, x_testing, y_testing = dataset.PrepareSets()
    print('Dataset is ready!')
    
    trainer.multi_worker_training(x_training, y_training, x_validation, y_validation)
    print('Training complete!')

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--job-dir", required=True)
    
    main(parser.parse_args())