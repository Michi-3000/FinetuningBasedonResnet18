# FinetuningBasedonResnet18
Validation of finetuning with Resnet18 using Pytorch on Caltech101.
<br>
 **Pretrained.py** is the net fintuned based on Resnet18 and **scratch.py** serves as control.
<br>
Before model training, you have to download [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) and run **data.py** for data preprocessing first.
<br>
**CAM.py** is added for visualization of experiment results. The algorithm is based on [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf).
