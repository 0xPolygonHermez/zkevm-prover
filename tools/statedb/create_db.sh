#!/bin/bash
if [ $# -ne 3 ] && [ $# -ne 4 ]
    then
        echo "usage: create_db <database> <user> <password> [<file.sql>]"
        exit
fi
echo "StateDB database creation"
echo "Installing PostgreSQL..."
sudo apt-get -y install postgresql
echo "Creating database $1..."
sudo -u postgres createdb $1
echo "Creating user $2..."
sudo -u postgres psql -c 'create role '$2' with login password '"'"$3"'"';'
sudo -u postgres psql -c 'grant all privileges on database '$1' to '$2';'
echo "Creating table state.merkletree..."
PGPASSWORD=$3 psql -U $2 -h 127.0.0.1 -p 5432 -d $1 -c 'create schema state;'
PGPASSWORD=$3 psql -U $2 -h 127.0.0.1 -p 5432 -d $1 -c 'create table state.nodes (hash bytea primary key, data bytea not null);'
PGPASSWORD=$3 psql -U $2 -h 127.0.0.1 -p 5432 -d $1 -c 'create table state.program (hash bytea primary key, data bytea not null);'
PGPASSWORD=$3 psql -U $2 -h 127.0.0.1 -p 5432 -d $1 -c 'create table state.keyvalue (key bytea primary key, data bytea);'
PGPASSWORD=$3 psql -U $2 -h 127.0.0.1 -p 5432 -d $1 -c 'create table state.version (hash bytea primary key, version bigint);'
PGPASSWORD=$3 psql -U $2 -h 127.0.0.1 -p 5432 -d $1 -c 'create table state.latestversion (version bigint unique);'
PGPASSWORD=$3 psql -U $2 -h 127.0.0.1 -p 5432 -d $1 -c 'insert into state.latestversion (version) values (0);'



if [ $# == 4 ]
    then
        PGPASSWORD=$3 psql -U $2 -h 127.0.0.1 -p 5432 -d $1 -f $4
fi

echo "Done."
echo "Example of connection string to use in the config.json file:"
echo "  \"databaseURL\": \"postgresql://$2:$3@127.0.0.1:5432/$1\""
echo
