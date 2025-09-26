# Practical Aberration Correction using Deep Transfer Learning with Limited Experimental Data


The paper is now available at [Optics Express](https://doi.org/10.1364/OE.557993)

# Table of contents
1. [Installation](#Installation)
2. [Sample Usage](##Sample-Usage)

## Installation
```bash
pip install -r requirements.txt
```

## Dataset
- [ed10](https://zenodo.org/record/14023331/files/ed10.zip?download=1)
- [ed24](https://zenodo.org/record/14023331/files/ed24.zip?download=1)

## Models
- [models](https://zenodo.org/record/14023331/files/models.zip?download=1)


## Sample Usage 
### Simulated data

Train model on SD25 simulated data

`python train_eval.py --config /config/sd25/conf.yaml`

Infer on SD25 simulated data: 
- Modify the arguments in the config file 'infer' to `true` and 'resume_path' to `/models/sd25/model_best.pth.tar`

Pre-train model on SD10 simulated data

`python train_eval.py --config /config/sd10/conf.yaml`

### Experimental data

Infer model on ED24 experimental data

`python train_eval.py --config /config/ed24/conf.yaml`

Infer model on ED10 experimental data using 1 channel only

`python train_eval.py --config /config/ed10/inputChannels/ch1/conf.yaml`

Fine-tune model with pre-training on 1% training data of ED10 experimental data 

`python train_eval.py --config /config/ed10/withAndWithoutPretrained/withoutPretrained1/conf.yaml`

Fine-tune model without pre-training on 50% training data of ED10 experimental data 

`python train_eval.py --config /config/ed10/withAndWithoutPretrained/withPretrained50/conf.yaml`
