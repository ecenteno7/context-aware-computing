# Context-Aware Computing: Human Activity Recognition and Ambient Soundscape Classification using Python 
This is an attempt to setup a few experiments surrounding context-aware computing using smartphones. Specifically, the included experiments start out with the premise of the original codebase using a popular dataset, *Human Activity Recognition Dataset*, and extending the idea of using a CNN architecture to identify ambient soundscapes with audio recordings. The ambient soundscape classification uses the popular dataset *UrbanSound8K*. More information about the datasets can be found in the Resources section of this ReadMe.  

The ambient sound classifier can be trained using the available dataset and is based off of identifying features from a Mel Spectrogram. ChatGPT was used as a resource to identify a good starting architecture for this neural network, but I have extended that original code suggested from the LLM quite a bit to what you see here in this repo.  

This project is a fork from the excellent codebase developed by GitHub User *tzamalisp*: [Original Project](https://github.com/tzamalisp/Human-Activity-Recognition-with-Tensorflow2-and-Keras). This project would not have been possible without their efforts. The original repository focused on developing a 2D CNN with Tensorflow and Keras (Sequential Model) for Human Activity Recognition (HAR). 

## Resources
### Human Activity Recognition
The project uses the famous *Human Activity Recognition Dataset*. Please, read the information link provided below to have a good indication of the dataset format.

* [Information on the Human Activity Recognition Using Smartphones DataSet](http://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
* [Download the UCI HAR Dataset Here](http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip)

The dataset is used to train using a classical 90/10 train/test split. Acceptable accuracy was achieved as shown from the original forked repository, and the confusion matrix from the original author seems to match what I have produced locally as well. 

### Ambient Soundscape Classification
For UrbanSound8K dataset info, please visit their website:

* [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html#10foldCV)

The dataset is used for a 10-fold cross-validation on the predefined folds in the dataset, as described in their original paper. I was able to achieve over 90% accuracy in this cross validation, and the model seems to work quite well on audio clips recorded on a smartphone. 
