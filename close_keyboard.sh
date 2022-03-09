#!/bin/bash

# check if the keyboard is already running, then start it if not 
if pgrep  matchbox-key > /dev/null
then
# Get the pid and stop it 
pid=$(grep -E '[0-9]' ./keyboard.pid)
kill -15 $pid 
fi