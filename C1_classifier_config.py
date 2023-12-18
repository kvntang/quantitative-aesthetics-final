#########################################################Configuration
#1 give a name to your model for saving purposes
modelname = "classifier_dresscode_2"

#2 select the model type to be used from the list below
#pick model type  [0:"resnet", 1:"alexnet", 2:"vgg", 3:"squeezenet", 4:"densenet", 5:"vit_b_32"]
modelType = 5

#3 set the folder where the training images are located organized in subfolders
#each subfolder name is the name of the corresponding class
imageFolder = 'data/dataset'
#########################################################End of configuration

classifierModels = ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "vit_b_32"]
classifierModel = classifierModels[modelType]
