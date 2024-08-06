# GapCL
This is the repository for *Adversarial Graph Augmentation and Feature Fusion Contrastive to Improve Molecular Graph Representations*
We design  a integrating **G**radient-based **a**dversarial **p**erturbations with dual-fusion enhanced **C**ontrastive **L**earning methods (GapCL). It can be directly integrated into graph-based models, and significantly improving the capabilities of these models (e.g., GCN, GAT, MPNN, CoMPT, Uni-mol) for molecular representation learning.
# Abstract
High-quality molecular representation is essential for AI-driven drug discovery. Despite recent progress in Graph Neural Networks (GNNs) for this purpose, challenges such as data imbalance and overfitting persist due to the limited availability of labeled molecules. Augmentation techniques have become a popular solution, yet strategies that modify the topological structure of molecular graphs could lead to the loss of critical chemical information. Moreover, adversarial augmentation approaches, given the sparsity and complexity of molecular data, tend to amplify the potential risks of introducing noise. This paper introduces a novel plug-and-play architecture, GapCL, which employs gradient-based adversarial perturbations for enhancement while incorporating dual-fusion enhanced contrastive learning for constraint to implement adaptive perturbation strategies for various benchmark models. GapCL aims to refine adversarial augmentation strategies by balancing noise suppression and critical information enhancement, thereby improving the capability of GNNs to learn chemical space information from molecular graphs. Extensive experiments demonstrate that GNNs equipped with GapCL achieve state-of-the-art performance across all 12 tasks on the MoleculeNet benchmark, significantly outperforming those equipped with simple adversarial enhancement schemes in terms of robustness. Further visualization studies also indicate that equipping models with GapCL achieves better representational capacity.
![GapCL model](https://github.com/stjin-XMU/GapCL/blob/main/GapCL.png)

# Environment
## GPU environment
CUDA 11.6

## create a new conda environment
- conda create -n GapCL python=3.9
- conda activate GapCL

## install environment
- pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple
- conda install -c dglteam dgl-cuda11.6
- pip install chardet
- pip install rdkit==2022.9.1
- pip install pynauty
- pip isntall Pyarrow
- pip install pandas
- pip install scikit-learn
- pip install communities

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

# Usage Tour
First, you need to preprocess the molecular datasets to the format of Uni-mol. Then call train.py to reproduce the results. For instance, if you want to implement the domain adaptation task with GapCL in BBBP, you can use the following comment:

‘’‘ python train.py ./configs_im/bbbp_gap.json GCN cuda:0 ’‘’

🌟Tips: Although the paper has provided detailed experimental descriptions, in order to accelerate your reproduction, please focus on the following points and parameters:

1. Use --gnn_type and --pretrain_gnn_path to specify different GNN methods and corresponding initialization pre-training weights;

2.Perform grid search for --weight_te and --weight_ke in [0.001, 0.01, 0.1, 1, 5];

3.For specific --weight_te and --weight_ke values, set --runseed from 0 to 9 and calculate the mean and variance.

# Reference
If our paper or code is helpful to you, please do not hesitate to point a star for our repository and cite the following content.



