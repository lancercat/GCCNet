# GCCNet
Code for the corresponding neurocomputing paper.

## Description
This repo contains the main application of the GCCblock on text detection and also the application on CoQA(a nlp task) 

We will see if we can release the ensembler, but at least here is the code. 

## Paths

This repo has 3 improtant paths, which are `$DATASET`,`$CODEROOT`, and `$MODELROOT`.

`$DATASET` holds training and evaluation datasets.


`$CODEROOT` holds the code.

`MODELROOT` holds the trained model and evaluation results.

By default, these paths are set to:

```
$DATASET: /home/username/pubdata/datasets/testing_sets
$CODEROOT: /home/username/cat/project_tf_family
$MODELROOT: /home/username/cat/project_uniabc_data
```
You may want to change them in `utils/libpath.py`, note the name can be different.

## Trained models
https://drive.google.com/drive/folders/1ZFVrKlKAYQO77cIIzcTYW0Iyeni2qkwe?usp=sharing

You need to put these two folders into `MODELROOT`


## Evaluation

Run `project_easts/tester.py`

## About
Code for the corresponding neurocomputing paper GCCNet: Grouped Channel Composition Network for Scene Text Detection.

Should you encounter any problems using the code, feel free to open an issue or email me (lasercat@gmx.us)

