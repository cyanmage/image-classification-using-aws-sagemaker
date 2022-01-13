# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling


### Results
Here is the repartition of the different rules tested.
![img/Rules_and_results.png](img/Rules_and_results.png)

"GPU" has not been used, so LowGPUUtilization is not relevant. 
What would be interesting to investigate are the "explodingTensor" and "Overfit" errors. It seems that coefficients of tensors are getting bigger and bigger. 
Moreover, with overfitting, the model adapt to training data, but not so well for validation data...
And some issues are found for initialization of weights and maybe for CPU bottlenecks.

#### Report
In the "System usage statistics", it is said : "The 95th percentile of the total CPU utilization is only 60%". So most of the time, the CPU seems to be underutilized (even if it seems there are some CPU bottlenecks). So a smaller instance shoud be recommanded (I used a "ml.p3.2xlarge" one)
In the "Overview: CPU operators" subpart, it is interesting to see that different tasks of convolutional layers are equally reparted, so there isn't a bottleneck at a specific layer.
Recommandations are given but they may contradict other choices (batch size should be augmented, but it was a choice of hyperparameter tuning process).
In the "Dataloading analysis", it is said "Your training instance provided 8 CPU cores, however your training job only ran on average 1 dataloader workers in parallel". So if I woul have to reload a work on this instance, I would augment the number of dataworkers in parallel to take maximum profit of the throughput.
For the "CPU bottlenecks", the rule is to compare CPU and GPU usages. As GPU was not used, it considers CPU is overused, and that a part of the job should have been dedicated to GPU.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
The model is adapted from a resnet50 pretrained model. So I took profit of transfer learning and already pretuned parameters. Dense layers were added on top to perform classification (my needs).

The model was recovered from the location it was saved in S3 (with argument model_data in the PyTorch constructor class). 
Then it was deployed and I got the endpoint_name (attribute), and created an instance of a Predictor class with this name.
I adapted the input format of the endpoint (predictor.serializer = IdentitySerializer("image/png")), to take into account images.
And then I launched a prediction with an image in "local" instance. Unfortunately, I got the following error : "An error occurred (InternalFailure) when calling the InvokeEndpoint operation (reached max retries: 4)". I will investigate furthermore if there is still time.

Remark : I had to get an entry point with a python file I recovered from the Course 5 ("inference.py" in the "Operationalizing Machine Learning in Sagemaker"), as I could not use the "train_model.py" one(It is a separated instance for inference estimating and smdebug was not installed on this instance, so I got an error as "train_model.py" uses smdebug).

Here is a screenshot of the deployed active endpoint :
[!img/endpoint_screenshot_in_sagemaker.png](img/endpoint_screenshot_in_sagemaker.png)



## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
# Hyperparameter-tuning-in-sagemaker
# image-classification-using-aws-sagemaker
# test image-classification-using-aws-sagemaker
# test image-classification-using-aws-sagemaker
