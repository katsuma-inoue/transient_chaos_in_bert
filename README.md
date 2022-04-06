# Transient Chaos in BERT

This repository includes scripts to demonstrate the complicated trajectory of ALBERT showing transient chaos described in [Transient Chaos in BERT](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.013204).

```cite
@article{inoue2022transient,
  title = {Transient Chaos in BERT},
  author = {Katsuma, Inoue and Soh, Ohara and Yasuo, Kuniyoshi and Kohei, Nakajima},
  journal = {Physical Review Research},
  volume = {4},
  issue = {1},
  pages = {013204},
  year = {2022},
  month = {March},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.4.013204},
  url = {https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.013204}
}
```

## Setup

We recommend using GPUs to emulate the experiments.

```bash
# install anaconda version of python
pyenv install anaconda3-2020.02
# make virtualenv
python -m venv venv
# activate env. For windows user, type .venv\Scripts\activate.bat
. venv/bin/activate
```

Installation

```bash
## For GPU users.
python -m pip install -r requirements-gpu.txt

## For CPU only users.
python -m pip install -r requirements.txt
```

Or, you can use pipenv

```bash
python -m pip install pipenv
pipenv install
pipenv shell  # activate virtual environment
```

To download pretrained weights and GLUE dataset, run the following command.

```bash
mkdir ../data
./setup/download.sh
```

The following scripts downloads the Wikipedia dataset used in masked language modeling and calculation of local Lyapunov exponent and synchronization offsets.  
This will take ~ 12 hours.

```bash
./setup/download-wiki.sh
```

## How to use?

The following snippet provides the minimum setups to get the ALBERT's trajectory.

```python
import os
import sys
import numpy as np
sys.path.append(".")
os.environ["USE_BERT_ATTENTION_PROJECT"] = "0"  # Only use the output of Albert encoder layer.
# Please specify this value to "1" for catching the attention values of the Transformer's encoders.

from src.library.model.albert_system import AlbertSystem

max_seq_len = 512
max_step = 100

net = AlbertSystem("albert_large", version="2", seq_len=max_seq_len, pretrained_weights=True)
input_text = ["ALBERT stands for A Lite BERT (Bi-directional encoder representation from transformer)."]
inputs, _ = net.tokenizer.tokenize_text(input_text, max_seq_len=max_seq_len, return_pieces=False)
for k, v in inputs.items():
  inputs[k] = np.array(v, dtype=np.int64).reshape(1, -1)

net.set_input(**inputs)
callback_gpu = lambda x: x
net_record = net.step(step_num=max_step, save_state=True, callback_gpu=callback_gpu)  # (max_step, 1, max_seq_len, 1024)
```

### Sample Notebooks

Sample notebooks are available in the following environments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/katsuma-inoue/transient_chaos_in_bert/blob/main/notebook/sample_trajectory.ipynb)

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/katsuma-inoue/transient_chaos_in_bert/blob/main/notebook/sample_trajectory.ipynb)

### solve STS-B task with several variations of ALBERT encoder layers

You can verify the performances of GLUE tasks including STS-B from 1 to 100 layers by running the following script.  
The results will be stored at `../result/glue/{task_type}`

```bash
python src/script/run_glue-tasks.py
```

The option `--fix_encoder` fixes the weights of the Transformer's encoders.  
With the option `--init_model`, you can initialize model weights of the Transformer's encoders before training, otherwise, you will use pre-trained parameters of ALBERT.

Also, when you encountered error related to GPU memory, you may avoid the error by specifying the device id like `CUDA_VISIBLE_DEVICES=0 python src/meta/run_glue-task_names.py`.

You can change the other tasks by specifying `--task_type` among `"CoLA", "MNLI", "MRPC", "SST-2", "MRPC", "STS-B", "QQP", "QNLI", "RTE", "WNLI"`.

### Masked Language Modeling (MLM) task

Before runnning the training, please prepare the MLM dataset from the Wikipedia dataset.

```bash
python setup/script/make_mlm_dataset.py --max_seq_len 128 --N 10000 --test_ratio 0.05
```

The argument of the option `--N` specifies the size of the dataset.

The script above will output `../data/corpus_for_mlm_train.txt` and `../data/corpus_for_mlm_test.txt`.

Now, it's time to run the MLM task.

```bash
python src/sh/run_mlm.py --start_layer 1 --end_layer 24 --gpus 0,1,2 --init_readout
```

You can calculate the accuracy of Masked Language Modeling with 1 encoder layer to 24 encoder layers.
If you add `--init_readout`, you will initialize readout layer used when pretraining.
