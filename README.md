# face-sketch-interactivate
This project is utilzed to change the synthesized sketch interactivitly with face pasring, it means that you can make the face components (eyes, nose et al.) have more texture while denoise the other face regions.
# Requirement
* Ubuntu or Windows
* Python3
* NVIDIA GPU + CUDA CuDNN
* Pytorch 0.3
* [Caffe](http://cs.nott.ac.uk/~psxasj/download.php?file=caffe-future.tar.gz)(for face parsing)
# Getting Started
To use the face sketch synthesis interactivitly, run ```interactParsChange.py```         
Because there's a lot of configuration involved, more details you can see the codes directly.

# Details
* We use the cycle-gan model to generate the sketch from photo which trained on CUHK database. You can use the Cycle_GAN model from [HERE](https://github.com/junyanz/CycleGAN), and train your own model to generate sketch.
* Face parsing need run with Caffe, and there are some requirements for the version of Caffe. More details can see in [Face Parsing](http://aaronsplace.co.uk/papers/jackson2016guided/index.html)
# Dataset
You can Download CUHK dataset from [HERE.](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html)
# Related Work
* [A CNN Cascade for Landmark Guided Semantic Part Segmentation](http://aaronsplace.co.uk/papers/jackson2016guided/index.html)
* [Cycle_GAN](https://github.com/junyanz/CycleGAN)   
Thanks for their great work.
# Results




