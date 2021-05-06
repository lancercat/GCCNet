#! /bin/bash
SPATH=$( cd "$( dirname $0 )" && pwd )
#! /bin/bash
sed -i 's/,\".*$//g' *.txt &> baka.kaba
cd $1 && rm -rf ./submit.zip && zip -q submit.zip ./*txt
cd $SPATH/welf_v2_lsvt_eval_tool && python2 script.py -g=gt.zip -s=$1/submit.zip



