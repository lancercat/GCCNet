#!/usr/bin/env sh
git pull;
export PYTHONPATH=$PWD/../
python  trainer.py $1;