## train
python ./pytorch_GAN_zoo/train.py StyleGAN --no_vis -n normal_stylegan -c ./pytorch_GAN_zoo/configs/config_normal.json
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -n preplus -c ./pytorch_GAN_zoo/configs/config_pre-plus.json
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -n plus -c ./pytorch_GAN_zoo/configs/config_plus.json

## synthesize
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/train/synthetic/1 --size_dataset 2000 -n normal
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/train/synthetic/2 --size_dataset 2000 -n preplus
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/train/synthetic/3 --size_dataset 2000 -n plus

python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/val/synthetic/1 --size_dataset 500 -n normal
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/val/synthetic/2 --size_dataset 500 -n preplus
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/val/synthetic/3 --size_dataset 500 -n plus
