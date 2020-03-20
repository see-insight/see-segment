#!/bin/bash --login

echo Command $0 $1 $2 $3 $4

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="${DIR}:${DIR}/../:${PYTHONPATH}"

if [ -z $1 ]
then
	export out_folder="CocoOutput"
else
	export out_folder="$1"
fi

echo "Setting Output folder to $out_folder"

cp CocoReport.ipynb ${out_folder}

cd ${out_folder}
jupyter nbconvert --no-input --allow-errors --execute --to html CocoReport.ipynb
cd ..
