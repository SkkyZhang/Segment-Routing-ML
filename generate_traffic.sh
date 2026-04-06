#!/bin/bash
while true
do
  echo "Starting a new Elephant Flow..."
  DURATION=$(( ( RANDOM % 21 )  + 10 ))
  sudo docker exec clab-srte4-h1 iperf3 -c 192.168.4.2 -t $DURATION
  
  echo "Waiting for idle period..."
  SLEEP=$(( ( RANDOM % 11 )  + 5 ))
  sleep $SLEEP
done