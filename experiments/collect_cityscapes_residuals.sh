#!/bin/bash

filenames=`ls /network/tmp1/bhattdha/detectron2_cityscapes/$1/*.pth`
for eachfile in $filenames
do
	eachfile=$(basename $eachfile)
	echo $eachfile
	python ~/detectron2/experiments/collect_cityscapes_residuals.py -name $1 -model_name $eachfile
done
