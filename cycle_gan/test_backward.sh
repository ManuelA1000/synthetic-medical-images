python test.py --model cycle_gan --name normal2plus --dataroot ../out/cnn/train/real/3

python test.py --model cycle_gan --name normal2preplus --dataroot ../out/cnn/train/real/2

python test.py --model cycle_gan --name preplus2plus --dataroot ../out/cnn/train/real/3


cd results/normal2plus/test_latest/images/
rm -v !(*fake_A*)
cd ../../../..


cd results/normal2preplus/test_latest/images/
rm -v !(*fake_A*)
cd ../../../..


cd results/preplus2plus/test_latest/images/
rm -v !(*fake_A*)
cd ../../../..
