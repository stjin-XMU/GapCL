# GapCL
This is the repository for *Adaptive Symmetric Adversarial Perturbation Augmentation for Molecular Graph Representations*
We design  a integrating **G**radient-based **a**dversarial **p**erturbations with dual-fusion enhanced **C**ontrastive **L**earning methods (GapCL). It can be directly integrated into graph-based models, and significantly improving the capabilities of these models (e.g., GCN, GAT, MPNN, CoMPT, Uni-mol) for molecular representation learning.
# Abstract
High-quality molecular representation is essential for AI-driven drug discovery. Despite recent progress in Graph Neural Networks (GNNs) for this purpose, challenges such as data imbalance and overfitting persist due to the limited availability of labeled molecules. Augmentation techniques have become a popular solution, yet strategies that modify the topological structure of molecular graphs could lead to the loss of critical chemical information. Moreover, adversarial augmentation approaches, given the sparsity and complexity of molecular data, tend to amplify the potential risks of introducing noise. This paper introduces a novel plug-and-play architecture, GapCL, which employs a symmetric perturbation mechanism during gradient-based adversarial enhancement to ensure that the perturbed graphs retain potentially essential chemical space information. Additionally, in contrast to existing perturbation update strategies, GapCL utilizes dual-fusion enhanced contrastive learning for constraints to implement an adaptive perturbation strategy tailored for different benchmark models. GapCL aims to improve the robustness and generalization capability of molecular graph representation models. Extensive experiments demonstrate that GNNs equipped with GapCL achieve state-of-the-art performance across all 12 tasks on the MoleculeNet benchmark, significantly outperforming those equipped with simple adversarial enhancement schemes in terms of robustness. Further visualization studies also indicate that equipping models with GapCL achieves better representational capacity. Code is available at https://anonymous.4open.science/r/GapCL-B5F6.
![GapCL model](https://github.com/stjin-XMU/GapCL/blob/main/GapCL.png)

# Environment
## GPU environment
CUDA 11.8

## create a new conda environment
- conda create -n GapCL python=3.8
- conda activate GapCL

## Requirements
- dgl==2.1.0+cu118
- dglgo==0.0.2
- networkx==3.1
- numpy==1.24.4
- ogb==1.3.6
- pandas==2.0.3
- pillow==10.4.0
- PyYAML==6.0.1
- rdkit==2022.9.1
- rdkit-pypi==2022.9.5
- requests==2.32.3
- scikit-learn==1.3.2
- scipy==1.10.1
- torch==2.2.1+cu118
- torchaudio==2.2.1+cu118
- torchdata
- torchvision==0.17.1+cu118
- tqdm==4.66.4
- deepchem==2.8.0

## install environment
This repositories is built based on python == 3.8.19. You could simply run
''' pip install -r requirements.txt 
to install other packages.

# Datasets
We selected 12 subtasks from the MoleculeNet dataset ([Wu et al. 2018](10.1039/C7SC02664A)) for experimental evaluation, comprising 9 classification tasks and 3 regression tasks. 

| Dataset | #Molecule | #Task | #Task Type |
| :---: | :---: | :---: |:---: |
| BBBP  | 2,035 | 1 | Classification|
| BACE | 1,513 | 1 | Classification |
| HIV | 41,127 | 1 | Classification |
| Tox21 | 7,821 | 12 | Classification | 
| SIDER | 1,379 | 27 | Classification |
| ClinTox | 1,468 | 2 | Classification |
| ToxCast | 8615 | 617 | Classification |
| MUV | 93127 | 17 | Classification |
| PCBA | 437,928 | 128 | Classification |
| ESOL | 1,128 | 1 | Regression |
| FreeSolv | 642 | 1 | Regression |
| Lipophilicity  | 4,198 | 1 | Regression | 

# Baselines
We refer to some excellent implementations of baselines used in our paper.
## Graph representation models
- GCN ([Kipf and Welling 2016](https://doi.org/10.48550/arXiv.1609.02907))
  
  https://github.com/tkipf/gcn
  
  https://github.com/tkipf/pygcn
  
- GAT ([Velickovic et al. 2017](https://doi.org/10.48550/arXiv.1710.10903))
  
  https://github.com/Diego999/pyGAT
  
- MPNN ([Gilmer et al. 2017](https://arxiv.org/pdf/1704.01212))
  
  https://github.com/brain-research/mpnn
  
- CoMPT ([Chen et al.2021](https://doi.org/10.24963/ijcai.2021/309))
  
  https://github.com/jcchan23/CoMPT
  
- Uni-mol ([Zhou et al. 2023](https://openreview.net/forum?id=6K2RM6wVqKu))
  
  https://github.com/deepmodeling/Uni-Mol
  
## Adversarial learning methods
- PGD ([Madry et al. 2017](https://doi.org/10.48550/arXiv.1706.06083))
  
  https://github.com/Harry24k/PGD-pytorch
  
- FLAG ([kong2022robust](https://arxiv.org/abs/2010.09891))
  
  https://github.com/devnkong/FLAG
  
## Other comparison models
| Model | #Model Type | #Model | #Model Type |
| :---: | :---: | :---: |:---: |
| D-MPNN  | Supervised | Attentive FP | Supervised |
| N-gram  | Pretraining  | PretrainGNN | Pretraining |
| GROVER | Pretraining  |  GraphMVP | Pretraining  |
| MolCLR | Pretraining  | GEM | Pretraining  |

# 🌟Quick Run
First, you need to preprocess the molecular datasets to the format of Uni-mol. Then call train.py to reproduce the results. For instance, if you want to implement the domain adaptation task with GapCL in SIDER, you can use the following comment:

`python train.py --config_path configs/sider.json --model_type GCN --device cuda:0 --target node --aug_size 0.5 --aug_method gap --save_model`

# Reference
If our paper or code is helpful to you, please do not hesitate to point a star for our repository and cite the following content.



