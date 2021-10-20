# synthetic-medical-images

## Abstract

### Purpose
Developing robust artificial intelligence (AI) models for medical imaging is often limited by access to large quantities of diverse patient data, from either barriers to sharing patient data, disease rarity, or quality/consistency of diagnostic labels. Retinopathy of prematurity (ROP), a potentially-blinding disorder of premature infants, suffers from all these challenges. Generative adversarial networks (GANs) may help, as they can synthesize highly-realistic images that may increase both the size and diversity of medical datasets.

### Design
Diagnostic validation study of convolutional neural networks (CNNs) for plus disease detection, a component of severe ROP, using a synthetic image-based dataset.

### Participants
Retinal fundus images were obtained from preterm infants during routine ROP screenings.

### Methods
Synthetic retinal vessel maps (RVMs) were generated from retinal fundus images. Progressively-growing GANs (PGANs) were trained to generate RVMs representing normal, pre-plus, or plus disease vasculature. CNNs were then trained to detect plus disease using real or synthetic RVMs, and were evaluated by testing on two different datasets of real RVMs.

### Main Outcome Measures
Features between real and synthetic RVMs were evaluated using Uniform Manifold Approximation and Projection (UMAP). Similarities between real and synthetic image sets were evaluated using the Frechet Inception Distance (FID). CNN performance was evaluated using area under the precision-recall curve (AUC-PR), area under the receiver operating characteristics curve (AUC-ROC), and confusion matrices.

### Results
Real and synthetic RVMs overlapped, by diagnosis, in the UMAP feature space. They were also more dissimilar to one another than real images were to one another, as indicated by the FID. A CNN trained on synthetic RVMs detected plus disease (AUC-PR: 0.833, AUC-ROC: 0.988) as well as a CNN trained on real RVMs (AUC-PR: 0.794, AUROC: 0.987). However, a CNN trained on synthetic data better characterized images in a test dataset graded by international experts.

### Conclusions
These findings suggest that synthetic medical datasets may be useful for training robust medical AI models and that GANs, potentially, may be able to synthesize realistic data without revealing protected health information, which could allow for dissemination of medical datasets to the broader research community without risking patients’ privacy.

## Data Sharing
Some output files may be made available upon request.
