#!/usr/bin/env bash


SPATH=$( cd "$( dirname $0 )" && pwd )

cd $1 && rm -rf ./submit.zip && zip -q submit.zip ./*txt
cd $SPATH/icdar15_val_tool && python2 script.py -g=gt.zip -s=$1/submit.zip
