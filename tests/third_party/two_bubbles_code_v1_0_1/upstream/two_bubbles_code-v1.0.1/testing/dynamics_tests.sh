#!/bin/bash

echo 'testing the bubble evolution code (current hyper_bubbles.py)'

echo 'Give path to simulation file'
read bubble_file

if [ -z $bubble_file ]
then
  bubble_file=./../
fi
echo $bubble_file

echo 'Give file in which to save results'
read save_file

if [ -z $save_file ]
then
  save_file=./
fi
echo $save_file

echo 'Give lambda_bar'
read lambda_bar

echo 'Give gamma'
read gamma

echo 'give maximum ds (optional, automatically 0.01, press enter to continue)'
read ds


cd $bubble_file

echo 'comparing energy derivatives in ktw eq 44 with a specific ds'
echo 'Proceed (y/n)?'
read proceed1
if [ "$proceed1" != "n" ]; then
  which_test='energy_derivative_comparison'
  python3 hyper_bubbles.py $lambda_bar $gamma $save_file $which_test $ds
fi

echo 'comparing energy derivatives in ktw eq 44 and looking at its convergence'
echo 'Proceed (y/n)?'
read proceed2
if [ "$proceed2" != "n" ]; then
  which_test='energy_derivative_convergence'
  python3 hyper_bubbles.py $lambda_bar $gamma $save_file $which_test
fi

echo 'comparing gamma calculated from the evolved field and the theoretical' \
'function for expanding bubble'
echo 'Proceed (y/n)?'
read proceed3
if [ "$proceed3" != "n" ]; then
python3 hyper_bubbles.py $lambda_bar $gamma $save_file 'gamma_test' $ds
fi
