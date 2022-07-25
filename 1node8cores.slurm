#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
module load foss/2019b
module load python/3.7.4
module load numpy/1.18.0-python-3.7.4
module load mpi4py/3.0.2-timed-pingpong
srun -N 1 -n 8 python3 search.py --twitter_path='bigTwitter.json' --grid_path='sydGrid.json'

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
~                     
