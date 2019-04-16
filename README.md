# Pedestrian Detection in Thermal Images using Saliency Maps

## Introduction
This repository contains the code for our (paper)[https://arxiv.org/abs/1904.06859] published in [CVPR PBVS 2019 workshop](http://vcipl-okstate.org/pbvs/19/).


Thermal images are mainly used to detect the presence of people at night or in bad lighting conditions, but perform poorly at daytime. To solve this problem, most state-of-the-art techniques use a fusion network that uses features from paired thermal and color images. We propose to augment thermal images with their saliency maps as an attention mechanism to provide better cues to the pedestrian detector, especially during daytime. We investigate how such an approach results in improved performance for pedestrian detection using only thermal images, eliminating the need for color image pairs. We train a state-of-the art Faster R-CNN for pedestrian detection and explore the added effect of PiCA-Net and R3-Net as saliency detectors. Our proposed approach results in an absolute improvement of 13.4 points and 19.4 points in log average miss rate over the baseline in day and night images respectively. We also annotate and release pixel level masks of pedestrians on a subset of the KAIST Multispectral Pedestrian Detection dataset, which is a first publicly available dataset for salient pedestrian detection.

For more details visit our [website](https://information-fusion-lab-umass.github.io/Salient-Pedestrian-Detection/)



## Preparation


First of all, clone the code
```
https://github.com/DebasmitaGhose/Multimodal_Influenza_Detection.git
```

Then, create a folder:
```
cd faster-rcnn.pytorch && mkdir data
```

### prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0 (**now it does not support 0.4.1 or higher**)
* CUDA 8.0 or higher

### Data Preparation



### Pretrained Model

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the data/pretrained_model/.

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

```
Above, BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. **On Titan Xp with 12G memory, it can be up to 4**.

If you have multiple (say 8) Titan Xp GPUs, then just use them all! Try:
```
python trainval_net.py --dataset kaist --net vgg16 --cuda --mGPUs --epochs=$EPOCHS
--s=SESSION --lr_decay_step=$LR_DECAY --use_tfb --bs=$BATCH_SIZE  --data_split $DATASET
```

Change dataset to "coco" or 'vg' if you want to train on COCO or Visual Genome.

## Test

If you want to evlauate the detection performance of a pre-trained vgg16 model on pascal_voc test set, simply run
```
python test_net.py --dataset kaist --net vgg16 --cuda  --vis 
--checksession $SESSION --checkepoch $EPOCH  --checkpoint $CHECKPOINT  --data_split $DATA_SPLIT

```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.

## Demo

If you want to run detection on your own images with a pre-trained model, download the pretrained model listed in above tables or train your own models at first, then add images to folder $ROOT/images, and then run
```
python demo.py --net vgg16 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir path/to/model/directoy
```

Then you will find the detection results in folder $ROOT/images.

**Note the default demo.py merely support pascal_voc categories. You need to change the [line](https://github.com/jwyang/faster-rcnn.pytorch/blob/530f3fdccaa60d05fa068bc2148695211586bd88/demo.py#L156) to adapt your own model.**

Below are some detection results:

<div style="color:#0000FF" align="center">
<img src="images/img3_det_res101.jpg" width="430"/> <img src="images/img4_det_res101.jpg" width="430"/>
</div>

## Webcam Demo

You can use a webcam in a real-time demo by running
```
python demo.py --net vgg16 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir path/to/model/directoy \
               --webcam $WEBCAM_ID
```
The demo is stopped by clicking the image window and then pressing the 'q' key.

## Authorship

This project is equally contributed by [Debasmita Ghose],[Shasvat Desai],[Sneha Bhattacharya] and, [Deep Chakraborty]

## Citation
@inproceedings{Kaist_Salient_Pedestrian_Dataset,
  author    = {Debasmita Ghose and
               Shasvat Desai and
               Sneha Bhattacharya and
               Deep Chakraborty and
               Madalina Fiterau and
	       Tauhidur Rahman},
  title     = {Pedestrian Detection in Thermal Images using Saliency Maps},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages     = {},
  year      = {2019}
}
