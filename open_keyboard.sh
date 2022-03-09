#!/bin/bash

# check if the keyboard is already running, then start it if not 
if pgrep  onboard > /dev/null
then
# Get the pid and stop it 
pid=$(grep -E '[0-9]' ./keyboard.pid)
kill -15 $pid 
# start the keyboard and store the pid in this file:
onboard & echo $! > ./keyboard.pid
else
# start the keyboard and store the pid in this file:
onboard & echo $! > ./keyboard.pid
fi