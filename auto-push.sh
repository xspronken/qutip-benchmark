#!/bin/bash

for i in {79..150}
do
   git pull origin 
   echo "$i" >> ./update-benchmarks.txt 
   git add update-benchmarks.txt
   git commit -m "512-$i"
   git push origin
   sleep 25m
done