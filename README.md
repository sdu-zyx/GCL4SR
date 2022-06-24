# GCL4SR

This is the PyTorch implementation of GCL4SR described in the paper:

> Yixin Zhang, Yong Liu, Yonghui Xu, Hao Xiong, Chenyi Lei, Wei He, Lizhen Cui, and Chunyan Miao. [Enhancing Sequential Recommendation with Graph Contrastive Learning.](https://arxiv.org/abs/2205.14837) In IJCAI 2022.

# Overview
The sequential recommendation systems capture users' dynamic behavior patterns to predict their next interaction behaviors. Most existing sequential recommendation methods only exploit the local context information of an individual interaction sequence and learn model parameters solely based on the item prediction loss. Thus, they usually fail to learn appropriate sequence representations. This paper proposes a novel recommendation framework, namely Graph Contrastive Learning for Sequential Recommendation (GCL4SR). Specifically, GCL4SR employs a Weighted Item Transition Graph (WITG), built based on interaction sequences of all users, to provide global context information for each interaction and weaken the noise information in the sequence data. Moreover, GCL4SR uses subgraphs of WITG to augment the representation of each interaction sequence. Two auxiliary learning objectives have also been proposed to maximize the consistency between augmented representations induced by the same interaction sequence on WITG, and minimize the difference between the representations augmented by the global context on WITG and the local representation of the original sequence. Extensive experiments on real-world datasets demonstrate that GCL4SR consistently outperforms state-of-the-art sequential recommendation methods.

Figure 2 shows the overall framework of GCL4SR. Observe that GCL4SR has the following main components: 1) graph augmented sequence representation learning, 2) user-specific gating, 3) basic sequence encoder, and 4) prediction layer.
![avatar](https://github.com/sdu-zyx/GCL4SR/blob/main/figures/GCL4SR.png)

# Requirement:
This implementation is based on pytorch geometric. To run the code, you will need the following dependencies:

- python 3
- torch 
- torch-geometric 
- tqdm 
- pickle 
- scipy 

# Datasets:

## data format
Taking home dataset as an example
```shell script
home.txt 
one user per line
user_1 item_1,item_2,...
user_2 item_1,item_2,...
0 1,2,3,4,5,6,7,8
1 5,9,10,11,12
...

all_train_seq.txt
have the same format as home.txt, but remove the last and the second last interaction item
0 1,2,3,4,5,6
1 5,9,10
...

train.pkl
have four list, containing user_id, item_seq, target_item, seqence_len
(
[0, 0, 0, 0, 0, 1, ...], 
[[1, 2, 3, 4, 5],
 [1, 2, 3, 4],
 [1, 2, 3],
 [1, 2],
 [1],
 [5, 9]
 ...],
[6, 5, 4, 3, 2, 10, ...],
[5, 4, 3, 2, 1, 2, ...]
)

test.pkl and valud.pkl
have the same format as train.txt
```

## build weighted item transition graph
Using all observed data(all_train_seq.txt) to build weighted item transition graph, execute:
```shell script
    python build_witg.py 
```

Figure 1 shows an example about the transition graph without edge weight normalization.
![avatar](https://github.com/sdu-zyx/GCL4SR/blob/main/figures/WITG.png)


# Usage:
For example, to run GCL4SR under Home dataset, execute:
```shell script
    python runner.py --data_name='home'
```

You can also change parameters according to the usage, which is also including detailed explanation of each hyper-parameter:
```shell script
    python runner.py -h
```


# License
If you find our work useful in your research, please consider citing the paper:

```
@inproceedings{IJCAI-GCL4SR,
  author    = {Yixin Zhang and
               Yong Liu and
               Yonghui Xu and
               Hao Xiong and
               Chenyi Lei and
               Wei He and
               Lizhen Cui and
               Chunyan Miao},
  title     = {Enhancing Sequential Recommendation with Graph Contrastive Learning},
  booktitle ={IJCAI 2022},
  pages     ={arXiv:2205.14837},
  year      ={2022}
}
```

# Contact
This implementation is partly based on [S3-Rec](https://github.com/aHuiWang/CIKM2020-S3Rec) modules.
If you have any questions or concerns, please send an email to yixinzhang@mail.sdu.edu.cn.
