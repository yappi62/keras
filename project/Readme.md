# VQA project

SNU-CVLAB graduation project for visual question answering problem.

## Settings

- VGGNet parameters caffemodel for keras (vgg16_weights.h5)
  - should be downloaded in the current directory 
  - can be downloaded **[here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)**
  - needed for running files with *_cnn*
- VGGNet fc7 features (vgg_feats.mat)
  - can be downloaded **[here](http://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip)**
  - should be located in '../../VQA/coco/vgg_feats.mat' --> loading it should result in (4096, 123287) numpy ndarray
  - needed for running *jzs1_vgg_feats.py*
  - `coco_vgg_IDMap.txt` file should be moved to '../../VQA/coco/' (needed for mapping VGG feature indices to actual coco images)
