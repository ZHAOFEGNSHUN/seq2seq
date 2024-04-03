#!/bin/bash
#SBATCH --job-name=seq2seq

#SBATCH --nodes=1
### 指定该作业需要1个节点数

#SBATCH --gres=gpu:1
###（声明需要的GPU数量）【单节点最大申请1个GPU】

# source ~/.bashrc # 激活预先设置好的一些环境变量参数，可以不写
conda activate pytorch # 激活执行这段代码的环境
cd /home/u202220081002025/seq2seq # 进到代码执行的目录
srun python train.py