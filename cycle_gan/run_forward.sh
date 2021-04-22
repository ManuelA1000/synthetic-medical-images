python test.py --model cycle_gan --name no2plus --dataroot ../out/gan/test/1

cd results/no2plus/test_latest/images/
rm -v !(*fake_B*)
cd ../../../..

python test.py --model cycle_gan --name no2preplus --dataroot ../out/gan/test/1

cd results/no2preplus/test_latest/images/
rm -v !(*fake_B*)
cd ../../../..

python test.py --model cycle_gan --name preplus2plus --dataroot ../out/gan/test/2

cd results/preplus2plus/test_latest/images/
rm -v !(*fake_B*)
cd ../../../..
