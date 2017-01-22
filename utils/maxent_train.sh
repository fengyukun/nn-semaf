#!/bin/bash

# ----------------------------------------------------------------------
# Author:	Yukun Feng
# Date:	11-18-16
# Email:	fengyukun@baidu.com
# Description:	Train maxent model for semantic frame labeling
# ----------------------------------------------------------------------

# Global variable
MAXENT_PATH=$HOME/local/bin/maxent

#@Brief: Train a model and test it for each file in given directories
#@Input: train_dir, each train file in train_dir must have the same name with test files in
# test_dir 
#@Input: test_dir
#@Return: 0 for success and 1 for error
function train_and_test() {
    if (( $# != 2 )); then
        echo "Parameter not enough. Current $#"
        return 1
    fi
    local train_dir=$1
    local test_dir=$2

    accuracy_sum=0
    counter=0
    for train_file in $(ls $train_dir); do
        train_file_path=$train_dir/$train_file
        test_file_path=$test_dir/$train_file

        echo "To train and test $train_file"
        echo ""

        # Get the accuracy
        result_info=$($MAXENT_PATH $train_file_path $test_file_path -i 10 2>/dev/null)
        # $MAXENT_PATH $train_file_path $test_file_path -i 80 -v --heldout $test_file_path
        if (( $? != 0 )); then
            echo "Error happens for training $train_file. Skip it"
            continue
        fi
        accuracy=$(echo $result_info | tail -n 1 |\
                 perl -ne 'if ($_ =~ /Accuracy: (.*)% \(/) {print $1}')
        accuracy_sum=$(bc <<< "scale=4; $accuracy_sum+$accuracy")
        counter=$(($counter+1))

        echo "Accuracy $accuracy for $train_file"
        echo ""
    done
    echo "Total: $counter. Average accuracy:"
    bc <<< "scale=4; $accuracy_sum/$counter"

    return 0
}

TRAIN_DIR="../data/corpus/parsed_wsj_pdev/pdev_new_split/train/"
TEST_DIR="../data/corpus/parsed_wsj_pdev/pdev_new_split/test/"

train_and_test $TRAIN_DIR $TEST_DIR
