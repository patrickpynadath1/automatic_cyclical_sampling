#!/bin/bash

for data in static dynamic omniglot caltech
do
  for sampler in cs dmala
  do
    bash bash_scripts/ebm_scripts/ebm_${sampler}_${data}.sh $1;
  done
done
