#!/bin/bash

#PBS -P ek63
#PBS -q normal
#PBS -N "{{:name}}"
#PBS -l ncpus={{:ncpus}}
#PBS -l mem={{:mem}}GB
#PBS -l walltime={{:wallhr}}:{{:wallmin}}:00
#PBS -j oe

source $HOME/jobs/multiphase/config_and_modules.sh
WRITE_DIR=$SCRATCH/multiphase_output/
cd $HOME/jobs/multiphase/

mpiexec -n {{:ncpus}} julia --project \
    $PROJECT_DIR/scripts/benchmarks.jl \
    {{:Px}} \
    {{:Py}} \
    {{:Pz}} \
    {{:n}} \
    $WRITE_DIR \
    {{:run_type}}