
## Environment Setup

- python-3.7
- PyTorch 1.0
- python library: python-sat, numpy, tqdm, pyparsing
- gcc/g++ 5.4.0 (or higher)
- make, cmake

You may follow the following instructions to set up the environment:
```bash
# create a python3 virtual environment
conda create -n  metal_env python=3 numpy tqdm pyparsing
conda activate metal_env
# install pytorch
conda install pytorch-cpu torchvision-cpu -c pytorch
# install sat solver
pip install python-sat
# install the dev version of metal
pip install -e .
# switch to the main entry
cd metal/main
```

## Experiments
### Out-of-the-box setting
No (pre-)training is required, and each instance will be solved from scratch (i.e. starting with randomly initialized weights).

```
./run_single.sh $BENCH_NAME $LOG_DIR_NAME
```

### Meta-learning setting
First, train the model on 60% tasks, and then test on remaining 40% tasks. Note that each task has its own grammar and specification. 

```
# train a meta-learner (trained model will be saved under the directory 'benchmarks')
./run_meta.sh
# test the trained meta-learner
./run_test.sh
```


## Reference

    @inproceedings{si2019metal,
        author    = {Si, Xujie and Yang, Yuan and Dai, Hanjun and Naik, Mayur and Song, Le},
        title     = {Learning a Meta-Solver for Syntax-Guided Program Synthesis},
        year      = {2019},
        booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
    }
  
