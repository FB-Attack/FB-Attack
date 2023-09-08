# Leakage-abuse attacks against Forward and Backward Private Searchable Symmetric Encryption 
This repository contains the code to evaluate query recovery attacks in Dynamic Searchable Symmetric Encryption (DSSE) schemes in the paper:
*  Lei Xu, Leqian Zheng, Chengzhi Xu, Xingliang Yuan, and Cong Wang. "*Leakage-Abuse Attacks Against Forward and Backward Private Searchable Symmetric Encryption.*" (CCS 2023).

## Folders

The ```dataset``` folder contains preprocessed documents for searchable encryption, while the ```preprocess``` folder contains scripts for obtaining these results from raw datasets. The remaining folders contain scripts for launching our attacks, with names corresponding to those in our paper.
DISCLAIMER: the code should work, but it's not very polished/efficient. 

## Datasets
### Enron
The original Enron dataset is enron_mail_20150507.tar.gz, downloaded from https://www.cs.cmu.edu/~enron/.

### Lucene
User posts downloaded from http://mail-archives.apache.org/mod_mbox/lucene-java-user.
