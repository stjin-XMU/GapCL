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
conda create -n GapCL python=3.9
conda activate GapCL

## install environment
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c dglteam dgl-cuda11.6
pip install chardet
pip install rdkit==2022.9.1
pip install pynauty
pip isntall Pyarrow
pip install pandas
pip install scikit-learn
pip install communities

# Dataset
We selected 12 subtasks from the MoleculeNet dataset ([Wu et al. 2018](10.1039/C7SC02664A)) for experimental evaluation, comprising 9 classification tasks and 3 regression tasks. 

  “`
   | 列1标题 | 列2标题 | 列3标题 |
   | ——- | ——- | ——- |
   | 行1单元格1 | 行1单元格2 | 行1单元格3 |
   | 行2单元格1 | 行2单元格2 | 行2单元格3 |
   “`

| Dataset | #Molecule | #Task | #Task Type|
| BBBP | --- | --- |
| 单元格1 | 单元格2 | 单元格3 |



