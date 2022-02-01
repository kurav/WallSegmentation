# Wall Segmentation

Implementation of a wall segmentation algorithm in PyTorch. Implementation is based on the [paper](https://arxiv.org/abs/1612.01105).<br/> 

The used database is [MIT ADE20K Scene parsing dataset](http://sceneparsing.csail.mit.edu/), where 150 different categories are labeled.

Because for solving the problem of wall segmentation, we do not need all the images inside the ADE20K database
(we need only indoor images), a subset of the database is used for training the segmentation module.

This is a simplified version of the code found [here](https://github.com/bjekic/WallSegmentation).
You can download the [pretrained weights](https://drive.google.com/drive/folders/1CmY8nunLORWvx9VT51ZUgI9E3zC6i0-b?usp=sharing) and place them in [Model weights](https://github.com/kurav/WallSegmentation/tree/main/Model_weights).

For testing, run [main.py](https://github.com/kurav/WallSegmentation/tree/main/main.py).
