# HiCLRE
Source code and data for "HiCLRE: A Hierarchical Contrastive Learning Framework for Distantly Supervised Relation Extraction"(Dongyang Li, Taolin Zhang, Nan Hu, Chengyu Wang, Xiaofeng He)
#### Reqirements
1. python 3.7
2. pytorch-1.8.1
3. transformers-4.10.3
4. tqdm
5. sklearn

#### Datasets:
All of the dataset files should be put in `./benchmark/no_preprocessing`
- [NYT10, NYT10-M](https://github.com/thunlp/OpenNRE/tree/master/benchmark)
- [KBP](https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5152)
- [GDS](https://arxiv.org/pdf/1804.06987.pdf)

#### Training
Step 1: Data Preprocessing

```
python benchmark/data_preprocessing.py
```
Step 2: Training

```
python example/train_distant.py \
    --dataset nyt10 \
    --batch_size 24
```
#### Acknowledgements
Thank [OpenNRE](https://github.com/thunlp/OpenNRE) for the help of code and datasets.

