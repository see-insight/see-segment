#!/bin/bash

rm STOP.STOP

echo "Starting Timer"
export STOP_TIME=$(( 8 * 60 * 60 ))

sleep $STOP_TIME 

touch STOP.STOP
