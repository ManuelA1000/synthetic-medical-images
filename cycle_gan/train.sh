## ALL VIEWS
python train.py --model cycle_gan --name normal2plus --dataroot ../out/gan/no2plus/images --n_epochs 50 --n_epochs_decay 51

python train.py --model cycle_gan --name normal2preplus --dataroot ../out/gan/no2preplus/images --n_epochs 50 --n_epochs_decay 51

python train.py --model cycle_gan --name preplus2plus --dataroot ../out/gan/preplus2plus/images --n_epochs 300 --n_epochs_decay 301



## POSTERIOR VIEWS
python train.py --model cycle_gan --name normal2plus --dataroot ../out/gan/no2plus/images --n_epochs 100 --n_epochs_decay 101

python train.py --model cycle_gan --name normal2preplus --dataroot ../out/gan/no2preplus/images --n_epochs 100 --n_epochs_decay 101

python train.py --model cycle_gan --name preplus2plus --dataroot ../out/gan/preplus2plus/images --n_epochs 600 --n_epochs_decay 601
