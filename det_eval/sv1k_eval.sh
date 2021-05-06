#! /bin/bash
SPATH=$( cd "$( dirname $0 )" && pwd )
cd $1 && rm -rf ./submit.zip && zip -q submit.zip ./*txt
cd $SPATH/sv1k_eval_tool && python2 script.py -g=gt.zip -s=$1/submit.zip

