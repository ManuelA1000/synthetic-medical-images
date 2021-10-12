## train
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_normal.json -n normal

python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_pre-plus.json -n preplus

python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_plus.json -n plus


## synthesize    NOTE: using --no_vis causes error
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./out/train/synthetic/1 --size_dataset 5000 -d ./out/output_networks -n normal

python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./out/train/synthetic/2 --size_dataset 5000 -d ./out/output_networks -n preplus

python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./out/train/synthetic/3 --size_dataset 5000 -d ./out/output_networks -n plus
