# Machine Learning Project 2
## Road Segmentation Challenge

Authors: Gianni Lodetti, Luca Bracone, Omid Karimi

In this project, the goal is to create a machine learning model for the classification of pixels in an image. More precisely, we perform road segmentation on a set of aerial images to classify road and background pixels. Some more information and dataset available on [AICrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation), and in our **Project_report**.

## Installation
Here are the dependencies needed to run the program with Python:

- [Pytorch](https://pytorch.org/): ```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```
- [Scipy](https://scipy.org/install/): ```pip install scipy```
- [Sklearn](https://scikit-learn.org/stable/install.html): ```pip install -U scikit-learn```
- [NetworkX](https://networkx.org/documentation/stable/install.html): ```pip install networkx[default]``` 
- For Topological Loss: ```pip install git+https://github.com/bruel-gabrielsson/TopologyLayer.git``` 
    (Not necessary, since not used for best model) CPP build tools might be needed for this as well: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## How to run
`cd project_road_segmentation` and then ```python run.py``` to train the best model and obtain ```submission.csv``` uploaded to AICrowd. </br >
Parameters can be change in the ```run.py``` file.
Please note that without using CUDA the training could take several hours, otherwise it takes around 5 minutes with this specifications: </br >
- CPU: Intel(R) Core(TM) i7-10700 @ 2.90GHz
- GPU: Nvidia RTX 3070, 8GiB

## Data and Checkpoint
You can find a copy of the data and of a checkpoint for our model on the following google drive:
https://drive.google.com/drive/folders/1_CPfQmbCfZK3uDIjoUtynXLijFRH1WHi?usp=sharing

## Loading a model
You can run `python run.py --load <path_to_model.checkpoint>` to load a model and obtain the submissions instead of having to train it yourself.

## Repository
- ```data/``` contains the training and test sets. Download (and unzip) them from [AICrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files).
- ```project_road_segmentation/``` folder contains all the code:
  - ```run.py``` execute training and model evaluation
  - ```train.py``` code for training a model
  - ```evaluation.py``` code for evaluation of given model
  - ```helper.py``` provided helper functions for image processing
  - ```losses.py``` implementation of loss modules not present in Pytorch
  - ```models.py``` code for the different models
  - ```post_processing.py``` code for image smoothing
  - ```mask_to_submission.py``` and ```submission_to_mask.py``` provided methods for submissions and visualization
- ```provided_examples/``` provided example to get started
- ```project2_description.pdf``` instruction for the project
- ```report_teamsalt.pdf``` final report 
