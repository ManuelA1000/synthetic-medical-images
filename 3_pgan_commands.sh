## train
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_normal.json -n normal
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_pre-plus.json -n preplus
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_plus.json -n plus


## inception score
python ./pytorch_GAN_zoo/eval.py inception -m PGAN -d ./out/output_networks --no_vis -n normal
python ./pytorch_GAN_zoo/eval.py inception -m PGAN -d ./out/output_networks --no_vis -n preplus
python ./pytorch_GAN_zoo/eval.py inception -m PGAN -d ./out/output_networks --no_vis -n plus


## synthesize    NOTE: using --no_vis causes error
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./out/train/synth_even/1 --size_dataset 7752 -d ./out/output_networks -n normal
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./out/train/synth_even/2 --size_dataset 1305 -d ./out/output_networks -n preplus
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./out/train/synth_even/3 --size_dataset 270 -d ./out/output_networks -n plus


# frechet inception distance
python -m pytorch_fid --dims 64 train/real/1 test/1
python -m pytorch_fid --dims 64 train/real/1 train/synth/1

python -m pytorch_fid --dims 64 train/real/2 test/2
python -m pytorch_fid --dims 64 train/real/2 train/synth/2

python -m pytorch_fid --dims 64 train/real/3 test/3
python -m pytorch_fid --dims 64 train/real/3 train/synth/3
