# final project

This code repository is aimed to carry out experiments on CAML model and other baseline models on multi-label text classification task.
Our experiment results saved in the folder "saved_models".

We reference the source code of data processing etc. from https://github.com/jamesmullenbach/caml-mimic


## Dependencies
* Python 3.6
* pytorch 1.3.0
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.20.3
* jupyter-notebook 5.0.0
* gensim 3.2.0
* nltk 3.2.4


## Data processing

Firstly, We cannot provide data from MIMIC3 due to licensing issues.
You need to download data files in the following directory manually from https://mimic.physionet.org/

To get started, first edit `constants.py` to point to the directories holding your copies of the MIMIC-III datasets. Then, organize your data with the following structure:

```
mimicdata
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions (already in repo)
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (already in repo)
```

Now, make sure your python path includes the base directory of this repository. Then, in Jupyter Notebook, run all cells (in the menu, click Cell -> Run All) in  `notebooks/dataproc_mimic_III.ipynb`. These will take some time, so go for a walk or bake some cookies while you wait. 

## Saved models

To directly reproduce the results in our report, first run the data processing steps above. We provide our pre-trained models for ablation experiment A and B in the report Section 5.3 , and our improved model in the report Section 5.4. They are saved as `model.pth` in their respective directories(like `predictions\improved_model_in_Section_5.4`). We also provide an `evaluate_model.sh` script to reproduce our results from the models.

## Training a new model

To train a new model from scratch, please use the script `learn/training.py`. Execute `python training.py -h` for a full list of input arguments and flags. The `train_new_model.sh` scripts in the `predictions/` subdirectories can serve as examples (or you can run those directly to use the same hyperparameters).

## Model predictions

The predictions that provide the results in the report are provided in `predictions/`. Each directory contains: 

* `preds_test.psv`, a pipe-separated value file containing the HADM_ID's and model predictions of all testing examples
* `train_new_model.sh`, which trains a new model with the hyperparameters provided in the paper.


