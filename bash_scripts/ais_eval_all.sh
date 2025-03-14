#!/bin/bash

for data in static dynamic omniglot caltech
do
  for sampler in cs dmala
  do
    bash bash_scripts/ais_scripts/ais_${sampler}_${data}.sh $1 > ais_${sampler}_${data}.txt;
  done
done