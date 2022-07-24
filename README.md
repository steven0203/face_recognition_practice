# Face recognition practice with pytorch
Use pytorch to train a CNN model for face recognition. 

## Data
Use VGG face2 dataset for training and evaluating.  
Use lfw dataset for evaluating.

## Training
Use training_model.ipynb to train CNN model.
### Setting
Epoch:100  
Batch size:64  
Loss: cross entropy  
Optimizer: SGD(momentum=0.9)  
Learning rate:   
At first:0.01    
epoch 40: 0.001  
epoch 70: 0.0001  

## Evaluate
Use evaluate.ipynb to evaluate.  
While evaluating with lfw dataset ,first use opencv with Haar Cascades to detect the faces of images.Then use the output of the CNN model's last layer as the feature of face image to caculate the feature distances of image pairs.

## Demo
Test with some other images.
First use opencv with Haar Cascades to detect the faces of images.Then ,use the output of the CNN model's last layer as the feature of face image to caculate the feature distance of two images. If the distance of two images is over threshold ,these two faces aren't the same person. Otherwise ,they are the same person.