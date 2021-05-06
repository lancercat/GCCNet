# GCCNet
Code for the corresponding neurocomputing paper.

## Description
This repo contains the main application of the GCCblock on text detection and also the application on CoQA(a nlp task) 

We will see if we can release the ensembler, but at least here is the code. 

## Trained models

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

## Evaluation

Run `project_easts/tester.py`

## About
Code for the corresponding neurocomputing paper.

Should you encounter any problems using the code, feel free to open an issue or email me (lasercat@gmx.us)

