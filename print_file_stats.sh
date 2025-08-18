#!/bin/bash

# Sample Run:
# ./print_file_stats.sh
#
# Shows the number of folders and files in data/carla/carla_abhinav directory
# which is useful to see if all data and folders are present.


num_files () {
    if test -d $1
    then
        cd $1
        DIR=$(pwd)
        NUM_FOLDERS=$(ls -1 | wc -l)
        NUM_FILES=$(find . -type f | wc -l)
        printf "%16s %6d %10d\n" "$1" $NUM_FOLDERS $NUM_FILES
        cd ../..
    else
        printf "%15s ABSENT\n" "$1"
    fi
}

cd data/carla/carla_abhinav
echo "____________________________________"
echo "   REL. PATH     FOLDER     FILES"
echo "____________________________________"
num_files "pitch0/town03"
num_files "pitch0/town05"
num_files "height6/town03"
num_files "height6/town05"
num_files "height12/town03"
num_files "height12/town05"
num_files "height18/town03"
num_files "height18/town05"
num_files "height24/town03"
num_files "height24/town05"
num_files "height27/town03"
num_files "height27/town05"
num_files "height30/town03"
num_files "height30/town05"

num_files "height-6/town03"
num_files "height-6/town05"
num_files "height-12/town03"
num_files "height-12/town05"
num_files "height-18/town03"
num_files "height-18/town05"
num_files "height-24/town03"
num_files "height-24/town05"
num_files "height-27/town03"
num_files "height-27/town05"
