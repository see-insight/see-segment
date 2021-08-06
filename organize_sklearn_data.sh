#!/bin/bash

file_head="slurm-31129571"
# file_head="slurm-31097725"
ls ${file_head}_*.out

for i in {0..74}
do
    ds_index=$((i%3))
    case $ds_index in
        0) 
            echo "moons"
            cp ${file_head}_${i}.out 0802_sklearn_data/moons;;
        1) 
            echo "circles"
            cp ${file_head}_${i}.out 0802_sklearn_data/circles;;
        2) 
            echo "linearly separable"
            cp ${file_head}_${i}.out 0802_sklearn_data/linearly_separable;;
        *) echo "unexpected case";;
    esac
done
