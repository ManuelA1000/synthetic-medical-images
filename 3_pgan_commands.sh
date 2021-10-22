## train
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_normal.json -n normal
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_pre-plus.json -n preplus
python ./pytorch_GAN_zoo/train.py PGAN --no_vis -c ./pytorch_GAN_zoo/configs/config_plus.json -n plus

## synthesize    NOTE: using --no_vis causes error
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./data/train/synth_even/1 --size_dataset 14540 -d ./out/output_networks -n normal
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./data/train/synth_even/2 --size_dataset 2320 -d ./out/output_networks -n preplus
python ./pytorch_GAN_zoo/eval.py visualization -m PGAN --save_dataset ./data/train/synth_even/3 --size_dataset 525 -d ./out/output_networks -n plus

# frechet inception distance
python pytorch-fid/src/pytorch_fid/fid_score.py --dims 64 ./out/train/real/1 ./out/val_test_combined/1
python pytorch-fid/src/pytorch_fid/fid_score.py --dims 64 ./out/train/real/1 ./out/train/synth/1

python pytorch-fid/src/pytorch_fid/fid_score.py --dims 64 ./out/train/real/2 ./out/val_test_combined/2
python pytorch-fid/src/pytorch_fid/fid_score.py --dims 64 ./out/train/real/2 ./out/train/synth/2

python pytorch-fid/src/pytorch_fid/fid_score.py --dims 64 ./out/train/real/3 ./out/val_test_combined/3
python pytorch-fid/src/pytorch_fid/fid_score.py --dims 64 ./out/train/real/3 ./out/train/synth/3
