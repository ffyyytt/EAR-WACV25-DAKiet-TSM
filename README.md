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

# Running
## Train
```console
python main.py elderly RGB --arch resnext50_32x4d --num_segments 8 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 20 40 --epochs 100 --batch-size 4 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb
```
## Eval
```console
python generate_submission.py elderly --arch=resnext50_32x4d --csv_file=submission.csv  --weights=checkpoint/TSM_elderly_RGB_resnext50_32x4d_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar --test_segments=8 --batch_size=1 --test_crops=1
```
