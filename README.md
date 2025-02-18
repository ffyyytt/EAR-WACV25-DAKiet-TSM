# EAR-WACV25-DAKiet-TSM

# Model
Temporal Shift Module (TSM) with backbone: resnext50_32x4d

# Data
## Config 1 (Public Leaderboard: 0.84402)
- Toyota Smarthome dataset
- ETRI-Activity3D - RGB videos (RGB_P091-P100)
- ETRI-Activity3D-LivingLab - RGB videos (RGB(P201-P230))

## Config 2 (Public Leaderboard: 0.78856)
- Toyota Smarthome dataset
- ETRI-Activity3D - RGB videos (full)
- ETRI-Activity3D-LivingLab - RGB videos (full)

## Process
#### Toyota Smarthome dataset
Just extract the dataset to data/train/

#### ETRI-Activity3D 
Move to the Toyota Smarthome dataset folder (data/train/) and rename.
```python
def filename_to_label(filename):
    if filename.startswith("A052") or filename.startswith("A053") or filename.startswith("A054") or filename.startswith("A055"):
        return "Locomotion"
    if filename.startswith("A002") or filename.startswith("A005") or filename.startswith("A006") or filename.startswith("A008") or filename.startswith("A007") or filename.startswith("A009") or filename.startswith("A018") or filename.startswith("A019") or filename.startswith("A020") or filename.startswith("A021") or filename.startswith("A022") or filename.startswith("A023") or filename.startswith("A024") or filename.startswith("A025") or filename.startswith("A026") or filename.startswith("A027") or filename.startswith("A028") or filename.startswith("A029") or filename.startswith("A030") or filename.startswith("A034"):
        return "Manipulation"
    if filename.startswith("A035") or filename.startswith("A036") or filename.startswith("A039") or filename.startswith("A044") or filename.startswith("A045") or filename.startswith("A046") or filename.startswith("A047") or filename.startswith("A048") or filename.startswith("A049") or filename.startswith("A050") or filename.startswith("A051"):
        return "Communication"
    if filename.startswith("A003") or filename.startswith("A010") or filename.startswith("A011") or filename.startswith("A012") or filename.startswith("A013")  or filename.startswith("A014") or filename.startswith("A015") or filename.startswith("A016") or filename.startswith("A017") or filename.startswith("A040") or filename.startswith("A043"):
        return "Hygiene"
    if filename.startswith("A000")or filename.startswith("A001") or filename.startswith("A004") or filename.startswith("A038"):
        return "Eating_Drinking"
    if filename.startswith("A031") or filename.startswith("A032") or filename.startswith("A033") or filename.startswith("A037") or filename.startswith("A041") or filename.startswith("A042"):
        return "Leisure"
    else:
        print(filename)

files = sorted(glob.glob("data/RGB*/*/*.mp4"))
for i, file in enumerate(tqdm(files)):
    name = file.split("\\")[-1]
    if filename_to_label(name):
        os.rename(file, f"data/train/{filename_to_label(name)}_Z2_{i}.mp4")
```
#### Final
```python
import os
import cv2
import glob
import random
import threading
from tqdm.notebook import *

def filename_to_label(filename):
    if filename.startswith("Locomotion") or filename.startswith("Enter") or filename.startswith("Getup") or filename.startswith("Laydown") or filename.startswith("Leave") or filename.startswith("Sitdown")  or filename.startswith("Walk"):
        return 0
    if filename.startswith("Manipulation") or filename.startswith("Cook") or filename.startswith("Cutbread") or filename.startswith("Makecoffee") or filename.startswith("Maketea") or filename.startswith("Pour"):
        return 1
    if filename.startswith("Communication") or filename.startswith("Usetelephone"):
        return 2
    if filename.startswith("Takepills") or filename.startswith("Hygiene"):
        return 3
    if filename.startswith("Drink") or filename.startswith("Eat") or filename.startswith("eat"):
        return 4
    if filename.startswith("Leisure") or filename.startswith("Readbook") or filename.startswith("Uselaptop") or filename.startswith("Usetablet") or filename.startswith("WatchTV"):
        return 5
    
    print(filename)
    a += 1

def do(files, k):
    for i, file in enumerate(tqdm(files)):
        txtfile.append([16*i+k, mp4_to_jpg(file, 16*i+k), filename_to_label(file.split("\\")[-1])])
        os.remove(file)
        if k == 0:
            stxt = "\n".join([" ".join(map(str, s)) for s in txtfile])
            with open("data/train_img/train.txt", "w") as f:
                f.write(stxt)
            with open("data/train_img/val.txt", "w") as f:
                f.write(stxt)

def mp4_to_jpg(filename, idx):
    os.makedirs(f"data/train_img/{idx}")
    vidcap = cv2.VideoCapture(filename)
    c = 0
    while True:
        success, image = vidcap.read()
        if not success:
            return c
        c += 1
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(f"data/train_img/{idx}/{c:06d}.jpg", image)

txtfile = []
files = sorted(glob.glob("data/train/*.mp4"))
random.shuffle(files)

ts = []
for i in range(16):
    ts.append(threading.Thread(target=do, args=(files[i::16], i)))
    ts[-1].start()
    
for i in range(16):
    ts[i].join()
```


# Running
Example: https://www.kaggle.com/code/fdfyaytkt/ear-wacv25-dakiet-tsm-rgb
Model: https://huggingface.co/fdfyaytkt/ear-wacv25-tsm-rgb-resnext50_32x4d
## Train
```console
python main.py elderly RGB --arch resnext50_32x4d --num_segments 8 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 20 40 --epochs 100 --batch-size 4 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
```
## Eval
```console
python generate_submission.py elderly --arch=resnext50_32x4d --csv_file=submission.csv  --weights=checkpoint/TSM_elderly_RGB_resnext50_32x4d_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar --test_segments=8 --batch_size=1 --test_crops=1
```
