#!/bin/bash

rm STOP.STOP

export HOURS=2

for hour in  $(seq 1 $HOURS)
do
    echo "Starting $hour of $HOURS hours"
    sleep $(( 60 * 60 )) #Hours
    echo "Completed $hour of $HOURS hours"
    date
    if [ -f STOP.STOP ]
    then
        echo "STOP.STOP file Exists.  Stoping timer" 
        echo 1
    fi
done

echo "STOPPING"
touch STOP.STOP
