#!/bin/bash
#Submit this script with: sbatch thefilename
#SBATCH --time=0:10:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -p batch   # partition(s)
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "CDAE"   # job name
#SBATCH --mail-user=up201507414@fe.up.pt   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=test
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
sleep 30
pip3 install sklearn
pip3 install numpy
pip3 install pandas
pip3 install os
pip3 install time
python3 knn_metalearner.py