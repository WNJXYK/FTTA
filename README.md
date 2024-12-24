# Fully Test-time Adaptation for Tabular Data
<p align="center">
🏠 <a href="" target="_blank">Homepage</a> • 📃 <a href="https://arxiv.org/abs/" target="_blank">Paper</a><br>
</p>

This is the official code for paper titled: Fully Test-time Adaptation for Tabular Data

## Quick Start

### 1. Prepare Python Environment

Clone FTTA repository, create conda environment, and install required packages. This repository is constructed based on Tableshift repository. For any detail of TableShift, please visit TableShift at [tableshift.org](https://tableshift.org/index.html)

NOTE: The requirements.txt file in FTTA is silently different from that in TableShift repo due to the package compatibility issues in Python.

```shell
git clone https://github.com/WNJXYK/FTTA.git
cd FTTA/src
conda env create -f environment.yml
conda activate ftta
python examples/run_expt.py
```

The final line above will print some detailed logging output as the script executes. When you see `training completed! test accuracy: 0.6221` your environment is ready to go! (Accuracy may vary slightly due to randomness.)

### 2. Run FTTA Approach

Run the FTTA approach using the command below:

```shell
conda activate ftta
python model_train.py --experiment {exp_name} --model {model_name}
```

For example, run the command below, soon it will show the result of FTTA methods and Unadapt methods.

```shell
conda activate ftta
python model_train.py --experiment diabetes_readmission --model mlp
```

For the first time running, dataset will be automatically download to `tmp` folder. For better reproduction our results, we provide pre-trained models for supported experiment and models, each experiment setting provides one model in  `scr/models` folder. Supported experiment and model are showed follows, we will expand the support list in the future.

Supported models and experiments in FTTA is

| Dataset              | String Identifier      |
| -------------------- | ---------------------- |
| Voting               | `anes`                 |
| ASSISTments          | `assistments`          |
| HELOC                | `heloc`                |
| Hospital Readmission | `diabetes_readmission` |

| Models         | String Identifier |
| -------------- | ----------------- |
| MLP            | `mlp`             |
| TabTransformer | `tabtransformer`  |
| FT-Transformer | `ft_transformer`  |

### 3. Explore more on FTTA methods

If you want to explore FTTA on more datasets, some key files may be helpful to you. 

##### Some key code files

| Path                                    | Function                                                     |
| --------------------------------------- | ------------------------------------------------------------ |
| `/tableshift/FTTA_src/FTTA.py`          | Definition of FTTA methods. When creating a FTTA class, a well trained tabular model, prior of training set and a optimizer type is need specifying. |
| `/tableshift/models/torchutils.py`      | The evaluation process is defined here. Tf you want to test some other methods, modify evaluate function is needed. |
| `/tableshift/models/default_hparams.py` | Algorithms' default hyper-parameters is defined here. We set 'n_epoch = 0' for test-time adaptation using trained model or provided model. If you want to train your own model, set 'n_epoch' to a positive integer. |

### 4. Dataset Availability

The availability of dataset is the same as TableShift benchmark. If you want to add more dataset, see the guidelines of TableShift benchmark at [tableshift.org](https://tableshift.org/index.html).

### 5. Q&A

If you have any questions, feel free to contact us at [yuky@lamda.nju.edu.cn](mailto:yuky@lamda.nju.edu.cn) or submit an issue here.

## Acknowledgements

We thank the author of TableShift benchmark providing a convenient framework to develop tabular algorithms.

## Citation

Please cite the paper if you refer to our code or paper from FTTA.

```
@inproceedings{zhou24ftta,
    author       = {Zhi Zhou and Yu-Kun Yang and Lan-Zhe Guo and Yu-Feng Li},
    title        = {Fully Test-time Adaptation for Tabular Data},
    booktitle    = {Proceedings of the 39th AAAI conference on Artificial Intelligence},
    year         = {2025}
}
```

