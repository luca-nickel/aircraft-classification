* Fast-r-cnn, there is a fast r-cnn with COCO dataset, cocodata set include airplane (index)
* fine tune by remove all other classes
* fine tune only ROI (region of interest) layer

Notes:
For the bounding box, the default Dataset can not be used as the annotions for the Bounding Boxes are not included,
a modified copy of the dataset will be used. It is stored in the bounding_boxes folder, with following adaption:
bounding_box_train: 0 - 8000 = 8000 Datapoints
bounding_box_test: 8001 - 9900 = 1900 Datapoints
bouding_box_val: 9901 - 10000 = 99 Datapoints

Logging Aufruf über:
 tensorboard --logdir=C:\\Projekte\\LearningSoftcomputing\\aircraft-classification\\src\\bounding_boxes\\logs\\2024-05-12_11_00_09_lr_0.001_batch_size_1_l2_0\\

Data Augmentation