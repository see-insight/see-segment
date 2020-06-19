#!/bin/bash

rm STOP.STOP

echo "Starting Timer"
export STOP_TIME=$(( 6 * 60 * 60 ))

sleep $STOP_TIME 
echo "STOPPING"
touch STOP.STOP
