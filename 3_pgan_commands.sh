## train
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -n normal -c ./pytorch_GAN_zoo/configs/config_normal.json
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -n preplus -c ./pytorch_GAN_zoo/configs/config_pre-plus.json
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -n plus -c ./pytorch_GAN_zoo/configs/config_plus.json

## synthesize
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/train/synthetic/1 --size_dataset 14540 -n normal
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/train/synthetic/2 --size_dataset 2320 -n preplus
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --np_vis -d ./out/output_networks --save_dataset ./data/train/synthetic/3 --size_dataset 525 -n plus

# frechet inception distance
python pytorch-fid/src/pytorch_fid/fid_score.py --dims 2048 ./data/fid/train/ ./data/fid/val_test
python pytorch-fid/src/pytorch_fid/fid_score.py --dims 2048 ./data/fid/train/ ./data/fid/synthetic
