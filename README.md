# synthetic-plus-classifier
  
Use a GAN to generate synthetic images of infant retinas with normal, pre-plus, and plus disease vasculature. Then train a CNN to learn the features of this disease. Finally, evaluate the learned model on real images of the disease.


## 5 Splits
- Split 1 + 2: Train GAN
- Split 3: Generate data from GANs and train CNN
- Split 4: Evaluate CNN during training
- Split 5: Test CNN
