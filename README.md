# CloudComputingCourse
Graduation project for Cloud Computing Course
## Goal of the project
The main aim of this project was to perform distributed training of a neural network on a dataset consisitng of images in a cloud environment. The dataset choosen for this task was [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
## Cloud environment
I chose [Google Cloud](https://cloud.google.com/gcp?utm_source=google&utm_medium=cpc&utm_campaign=emea-pl-all-pl-bkws-all-all-trial-e-gcp-1011340&utm_content=text-ad-none-any-DEV_c-CRE_529432261649-ADGP_Hybrid%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20~%20GCP%20~%20General%23v2-KWID_43700060393215920-aud-412600777667%3Akwd-6458750523-userloc_9061066&utm_term=KW_google%20cloud-NET_g-PLAC_&gclid=Cj0KCQiAi9mPBhCJARIsAHchl1wGwqnnQ86dPcwqLEG7G9rLxNoifF-RLNMnJSX9FgDrRgRvNQE9_8EaAo6TEALw_wcB&gclsrc=aw.ds) with 300$ budget as a free trial. To train the model I utilized Google AI Platform, which is the feature of Google Cloud.
## Neural Network model architecture
For the sole purpose of training I created simple feed forward network of several fully-connected layers- each with **ReLu** as an activation function. Every network output feeds Batch Normalization layer to standarize every output of every neuron. Moreover the dropout layers are used to avoid overfitting. As this dataset contains 10 classes the model was meant to perform classification task and thus the last layer has **Softmax** activation.
```python
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
```
Other network's parameters are listed below:
```python
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
```
This model can also perform learning rate reduction in case the validation score gets stucked.

The key part is to understand that when the model is unitialized and compiled, a copy of the network goes to every worker and each worker perform training on a sub-sample of a global batch parameter. Every worker also needs to receive steps-per-epoch parameter calculated as the ratio of training set size and global batch. 
### Remarks on model saving
To properly save trained model this task has to be adressed to the master-chief machine. However as each machine(workers and chief) receives the same code they require a method to help them distinguishe which one is the chief. Therefore, I created set of methods to achieve this aim.
```python
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
```
When the training is ready ```save_model``` method is called and with the help of method ```_is_chief``` it gets an information if that machine is chief or not. After successful discrimination the model is being saved and chief waits fo workers to finish their jobs.
## Training environment
The framework used for this training is the [Tensorflow 2.6](https://www.tensorflow.org/) with [Keras 2.6](https://keras.io/). However to make this training scaleable for cloud computing i put this environment and code into [Docker](https://circleci.com/docker/?utm_source=google&utm_medium=sem&utm_campaign=sem-google-dg--emea-en-nbAuth-maxConv-auth-nb&utm_term=g_e-docker_c__rsa2_20210709&utm_content=sem-google-dg--emea-en-nbAuth-maxConv-auth-nb_keyword-text_rsa-docker_exact-&gclid=Cj0KCQiAi9mPBhCJARIsAHchl1yrK5JrteN-quD7u02l9fWlqOviW5VlvMmfsUnKVUqv8njzlMLhES4aAt02EALw_wcB) container. After the code was ready and trained well on my local machine I uploaded it to Vertx AI Jupyter Notebook where I created ```home/manage/trainer``` directory where the code was stored. To create a Docker image with the given version of Tensorflow I created a Dockerfile:
```docker
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-6

WORKDIR /

COPY trainer /trainer

ENTRYPOINT ["python","-m","trainer.train"]
```
And after Dockerfile was ready I used command:
```bash 
docker build ./ -t gcr.io/cloudecomputingcourse/manage:v10
``` 
to create Docker image and: 
```bash 
docker push gcr.io/cloudecomputingcourse/manage:v10
``` 
to push it into Google Cloud Container Repository.
## Testing on a local machine
As the Google Cloud charges even for failed trainings it is recommended to test the model on a local scale before deployment on the cloud. To perform local training after building Docker image I used command:
```bash
docker run --rm gcr.io/cloudecomputingcourse/manage:v10 --job-dir gs://cloudecomputingcourse/manage
```
the last part of the command in the argument that tells my model where to save trained model.
## Distributed training on a Google AI Platform
Finally, when model was training well locally, the Docker image was on Container Repository it was time to submit training job to the cluster. With the pipeline given below I was able to tell how many worker I would like to have and what type of machine I wanted to use. For such a light network I decided to use the most standard  one node machine with 4 vCPUs and 15Gb RAM memory.
Pipeline for job submission:
```bash
gcloud ai-platform jobs submit training multi_worker_training4 --job-dir=gs://network_bucket/manage \
--master-image-uri=gcr.io/cloudecomputingcourse/manage:v10 --master-machine-type=n1-standard-4 \
--scale-tier=CUSTOM --use-chief-in-tf-config=true --worker-image-uri=gcr.io/cloudecomputingcourse/manage:v10 \
--worker-count=2 --worker-machine-type=n1-standard-4
```
### Results
When the training was done in the console log I received this information which explicitly says that 2 workers with one master machine were deployed for this job and that they all successfuly finished their tasks.
[![final-log.png](https://i.postimg.cc/FKTbJBcT/final-log.png)](https://postimg.cc/KKg36Q0T)
