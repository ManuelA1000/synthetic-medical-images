python test.py --model cycle_gan --name normal2plus --dataroot ../out/cnn/convert/1

python test.py --model cycle_gan --name normal2preplus --dataroot ../out/cnn/convert/1

python test.py --model cycle_gan --name preplus2plus --dataroot ../out/cnn/convert/2


cd results/normal2plus/test_latest/images/
rm -v !(*fake_B*)
cd ../../../..

cd results/normal2preplus/test_latest/images/
rm -v !(*fake_B*)
cd ../../../..

cd results/preplus2plus/test_latest/images/
rm -v !(*fake_B*)
cd ../../../..
