# MTLocInt
MTLocInt: A multi-task learning framework based on three-dimensional chaos game representation (3D-CGR) for lncRNA subcellular localization prediction and lncRNA-protein interaction recognition.

## Requirements
python==3.8\
torch==1.10.0\
torchvision==0.11.1\
tensorflow==2.2.0\
tensorboard==2.2.2\
keras==1.1.2\
pandas==2.0.3\
h5py==2.10.0\
biopython==1.83\
matplotlib==2.5.0\
scipy==1.10.1\
protobuf==3.20.0\
opencv-python==4.10.0.86\
scikit-learn==1.3.2\
scikit-multilearn==0.2.0\
scikit-image==0.21.0\
numpy==1.20.3

## 1 Setup instructions
Install the dependencies as specified in the Requirements section.

## 2 Parameter settings

## 3 Prepare input data
There are multiple datasets involved in the process:
### 3.1 For lncRNA subcellular localization prediction
#### 3.1.1 Dataset Loc I
#### Benchmark dataset
Sequences: dataset_preparation\dataset_Loc_I\seq_729.csv;\
Labels: dataset_preparation\dataset_Loc_I\label_729.csv.
#### Independent test set
Sequences: dataset_preparation\dataset_Loc_I\seq_holdout_82.csv;\
Labels: dataset_preparation\dataset_Loc_I\label_holdout_82.csv.
#### 3.1.2 Dataset Loc II
#### Benchmark dataset
Sequences: dataset_preparation\dataset_Loc_II\homo_219.fasta;\
Labels: dataset_preparation\dataset_Loc_II\label_homo_219.csv.
#### Independent test set
Sequences: dataset_preparation\dataset_Loc_II\mus_65.fasta;\
Labels: dataset_preparation\dataset_Loc_II\label_mus_65.csv.
### 3.2 For lncRNA-protein interaction recognition
#### Dataset Int
LncRNA sequences: dataset_preparation\dataset_Int\NPInter_rdrna_seq_3046.fasta;\
Protein sequences: dataset_preparation\dataset_Int\NPInter_rdprotein_seq_136.fasta;\
Labels: Combine positive and negative interaction labels from positive_NPInter.csv and negative_NPInter.csv.
## Generate CGR images
Run the .m files in Matlab (2021b) to generate CGR images in batches:
#### CGR_3D_RNA_729_AG_T.m, CGR_3D_RNA_82_AG_T.m, CGR_3D_RNA_3046_AG_T.m, and CGR_3D_Protein_136_AG_T.m are used to generate CGR images for different datasets.
#### cgr3drna_AG_T.m and cgr3dprotein_AG_T.m are sub-functions that produce CGR coordinates for RNA and protein sequences respectively.
