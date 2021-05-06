rm tmp*
BASE=$(pwd)
CH_DUMMY=dummy
MSK_DUMMY=dummy
j=0
for i in $(find $BASE | grep jpg| sort)
do
    echo $i >> tmp_iml;
done
for i in $(find $BASE | grep txt| sort)
do
    echo $i >> tmp_lnl;
    echo $CH_DUMMY >> tmp_chl
    echo $MSK_DUMMY >> tmp_mskl
done

cat  tmp_iml > a.lst
cat  tmp_mskl >> a.lst
cat  tmp_chl >> a.lst
cat  tmp_lnl >> a.lst
mv a.lst a.index
rm tmp*
    
