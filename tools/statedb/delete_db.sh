#!/bin/bash
if [ $# -ne 2 ] 
    then
        echo "usage: delete_db <database> <user>"
        exit
fi
echo "StateDB database deletion"
echo "Deleting database $1..."
sudo -u postgres dropdb $1
echo "Deleting user $2..."
sudo -u postgres dropuser $2
echo "Done."
echo 