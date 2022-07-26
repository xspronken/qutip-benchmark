#!/bin/bash
for i in {1..150}
do
   echo "Welcome $i times" >> update.txt
   git add update.txt
   git commit -m "autorun git actions"
   git push upstream pytest-ci
   sleep 50m
done