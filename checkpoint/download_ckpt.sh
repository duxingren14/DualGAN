DATASET_NAME=$1

if [[ $DATASET_NAME != "day-night" &&  $DATASET_NAME != "oil-chinese" && $DATASET_NAME != "facades" && $DATASET_NAME != "sketch-photo" ]]; then
echo "Usage: bash ./checkpoint/download_ckpt.sh model_name"
echo "dataset names: facades, day-night, oil-chinese, sketch-photo"
exit 1
fi

LINK=http://www.cs.mun.ca/~yz7241/dualgan/ckpts/$DATASET_NAME.zip
FILE=./checkpoint/$DATASET_NAME.zip
DIR=./checkpoint/$DATASET_NAME/
if [ -d "checkpoint" ]; then
	wget -N $LINK -O $FILE
	unzip $FILE -d ./checkpoint/
	rm $FILE
else
	echo "Remember to run the scripts in the root directory!"
	echo "Usage: bash ./checkpoint/download_ckpt.sh dataset_name"
fi
