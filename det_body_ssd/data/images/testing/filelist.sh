
imgFolder=JPEGImages
anoFolder=Annotations

imglist=`ls $imgFolder`
anolist=`ls $anoFolder`

for img in $imglist
do 
 #for ano in $anolist
 #do
 	#if [ ${img%.*} == ${ano%.*} ];then

 		echo ${imgFolder}/$img ${anoFolder}/$ano >>list.txt
 	#fi
 #done
done