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
### 3.1 For lncRNA subcellular localization prediction\
#### 3.1.1 Dataset Loc I
#### Benchmark dataset
Sequences from dataset_preparation\dataset_Loc_I including seq_729.csv (729 samples from H. sapiens) and seq_holdout_82.csv (82 samples from H. sapiens).\
Labels from dataset_preparation\dataset_Loc_I including label_729.csv and label_holdout_82.csv.\
#### 3.1.2 Dataset Loc II
#### Benchmark dataset
Sequences from dataset_preparation\dataset_Loc_II including homo_219.fasta (219 samples from H. sapiens) and mus_65.fasta (65 samples from M. musculus).\
Labels from dataset_preparation\dataset_Loc_II including label_homo_219.csv and label_mus_65.csv.\
