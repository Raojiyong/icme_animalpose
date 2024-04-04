# ICME & Multi-Modal Video Reasoning and Analyzing Competition

## Introduction
Animal Pose Estimation on Animal Kingdom

## Environment 
The code is developed using python 3.6 on Ubuntu 20.04 NVIDIA GPUs are needed.
The code is developed and tested using 2 NVIDIA RTX 3090 cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. We use the v1.8.1 pytorch and v1.9.1 torchvision.
2. Clone this repo, and we'll call the directory that you cloned as ${ROOT}.
3. Install dependencies:
```shell
pip install -r requirements.txt
```
4. Make libs:
```shell
cd ${ROOT}/lib
make
```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi)
```shell
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python3 setup.py install --user
```
6. Init output and log directory:
```shell
mkdir output
mkdir log
```
Your directory tree should look like this:
```shell
${ROOT}
├── data
├── experiments
├── lib
├── log
├── pretrained
├── output
├── tools 
├── README.md
└── requirements.txt
```
7. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/drive/folders/1mISQZZXyvsDUGhrKMPlPpwRLx2H2SLmj?usp=drive_link)).
And our models' is pretrained on COCO,the COCO pretrained can be downloaded from([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing))
```shell
${ROOT}
 `-- pretrained
     `-- pytorch
         |-- P1S1
         |   |-- model_best.pth
         |-- P1S2
         |   |-- model_best.pth
         |-- P1S3
         |   |-- model_best.pth
         |-- P1S4
         |   |-- model_best.pth
         |-- P1S5
         |   |-- model_best.pth
         |-- pose_coco
         |   |-- pose_hrnet_w48_256x192.pth
```
### Data preparation
Animal Kingdom, please download from [Animal Kingdom Dataset](https://github.com/SUTDCV/Animal-Kingdom).
split5_fold.py is used to split 5
```shell
${POSE_ROOT}
|-- data
`-- |-- ak
    `-- |-- pose_estimation
        |   | annotation
            |   | ak_P1
                |-- train.json
                |-- test.json
                |-- split5_fold.py
        |   | dataset
            |-- AAACXZTV
            |-- AAAUILHH
            ...
```
### Training and Testing
Training on Animal Kingdom train set.
```shell
python tools/train.py --cfg experiments/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_all.yaml
```
Testing on Animal Kingdom test set.
```shell
python tools/test.py --cfg experiments/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_all.yaml \
TEST.MODEL_FILE pretrained/2Encoder_w48_256x256_adaW_cutmix_p4_all/model_best.pth
```
### Ensemble Testing
Running the following command to split the train set to 5 folds:
```shell
cd ${ROOT}data/ak/pose_estimation/annotation/ak_P1
python split5_fold.py
```
Training on the 5 train subsets. # represents the number of the subset.
```shell
python tools/train.py --cfg experiments/ak/kitpose_part/P1S#.yaml
```
Ensemble testing:
```shell
python tools/ensemble_test.py --cfg experiments/ak/kitpose_part/mean_ensemble_test.yaml
```
