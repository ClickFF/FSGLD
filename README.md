# FSGLD
A Full-Spectrum Generative Lead Discovery (FSGLD) Pipeline via DRUG-GAN: A Multiscale Method for Drug-like/Target-specific Compound Library Generation

## Introduction

We present the Full-Spectrum Generative Lead Discovery (FSGLD), a deep learning-driven pipeline for efficient drug lead identification. FSGLD integrates generative modeling with molecular docking, molecular dynamics simulations, ligand-residue interaction profile, MM-PBSA, thermodynamic integration (TI), and experimental validation to bridge theoretical design and practical application. The core multiscale DRUG-GAN models enable de novo design for both drug-like and target-specific compounds across three scenarios: I. generation of random drug-like compounds, II. generation of target-specific compounds, III. generation of target-biased compound series featuring shared chemical structures. FSGLD significantly outperformed traditional computer-aided drug design methods in generating novel chemicals which specifically target the CB2 receptor. 

## System/software requirements
The source code developed in Python 3.10.8 and Tensorflow 2.10.0. The required python dependencies are given below. FSGLD is supported for cpu/gpu and there is no additional non-standard hardware requirements.
```
-tensorflow 2.10.0
(-tensorflow-gpu 2.10.0 #if gpu available
-cudatoolkit 11.2.2
-cudnn 8.4)
-matplotlib 3.10.1
-pandas 2.2.3
-numpy 1.26.4
-scikit-learn 1.6.1

Optional: You may need to install openbabel (https://openbabel.org/docs/Installation/install.html) to convert searched molecules (.sdf) to FP2 format to do the second-round similarity search.
```

## Datasets
The training sets for three scenarios are located in DCGAN_s1/data, CDCGAN_s2/data, and CDCGAN_s3/data. For similarity search, for the size limitation, we only provide a template.bin file for demo. It has been located in similarity_search/s1_s2 and similarity_search/s3

## Model training and molecular generation
To train dcGAN/CDCGAN, generate and evaluate drug-like molucules/CB2 compounds/CB2 compound series, you can run the following command in all three folders:
```python
$ python train.py
$ python generate.py
```
The evaluation metrics for the generated 10,000 samples will be printed after molecular generation, including uniqueness, diversity, novelty, average similarity and maximal similarity. 

## Similarity search
In our work, we calculated tanimoto similarity between our generated fingerprints (FPs) and compounds in ChEMBL/ZINC library. In similarity_search folder, we provide csh and ELF code for you to convert MACCS FPs to FP2 FPs and conducted similarity search. tanimoto_ss is used for simple similarity search, while tanimoto_ss_mcs is used for similarity search fixed with MCS features. We provide MCS information for all mcs i, mcs ii and mcs iii, which are located in similarity_search/mcs. For the searched ChEMBL/ZINC compounds using generated samples in MACCS format, you can first use openbabel to convert searched compounds (sdf format) to FP2 in hex format, then used our provided hex2bin2 to convert hex format to binary format.
Take similarity search for molecules with MCS ii features as an example (similarity_search/s3)
1. Similarity search in similarity_search/s3/first_round
```bash
$ ./gen_bat
```
3. Analyze the searched molecules in output/ and selected top ones. Retrive ChEMBL/ZINC sdf files, Use openbabel to convert sdf to FP2 FPs.  
4. Convert MACCS to FP2:
```bash
$ ./hex2bin2 -i maccs.bin -o fp2.bin
```
5. Similarity search (`./gen_bat`) for the second round.
