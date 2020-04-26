# Not so strong baseline for Video-Person-ReID


### Introduction
This repository contains a not so strong baseline for video-based person reID. It is mainly forked from [video-person-reid](https://github.com/jiyanggao/Video-Person-ReID) and [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline). What i done is just merging them ;) and introducing the [nvidia-apex](https://github.com/NVIDIA/apex) to convert the model into a syncbn_model as well as doing slight modifies on the model and tricks, the reason of introducing apex is mainly because my poverty, if you owes a 32GB V100 Graphic Card, you can just ignore the apex operation and run the codes on a single card, then i nearly contributes nothing in this work :).

Requirements:
```
pytorch >= 0.4.1
torchvision >= 0.2.1
tqdm
[nvidia-apex](https://github.com/NVIDIA/apex), please follow the detailed install instructions 
```


### Dataset
#### MARS
Experiments on MARS, as it is the largest dataset available to date for video-based person reID. Please follow [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) to prepare the data. The instructions are copied here: 

1. Create a directory named `mars/`.
2. Download dataset to `mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
    ```
    mars/
        bbox_test/
        bbox_train/
        info/
    ```
5. Change the global variable `_C.DATASETS.ROOT_DIR` to `/path2mars/mars` and `_C.DATASETS.NAME` to `mars` in config or configs.

#### Duke-VID
1. Create a directory named `duke/` under `data/`.
2. Download dataset to `data/duke/` from http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip.
3. Extract `DukeMTMC-VideoReID.zip`.
    ```
    duke/
        train/
        gallery/
        query/
    ```
4. Change the global variable `_C.DATASETS.ROOT_DIR` to `/path2duke/duke` and `_C.DATASETS.NAME` to `duke` in config or configs.

### Usage
To train the model, please run

    python main_baseline.py
 
Please modifies the settings directly on the config files.   
(to be complete)

### Performance
Best performance on MARS:
mAP : 81.2%
Rank-1 : 86.6%
Rank-5 : 96.0%

Ablation study and experiments on Duke-VID is undergoing, 

Since I'm graduated, I have no graphic cards to conduct the expriments, my sincere aplogies....Hoping the merge request from U, my rich friends~ 
