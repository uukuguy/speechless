#!/usr/bin/env bash

mkdir -p data/postgres data/export
docker create --name postgres-sql-eval \
    -e POSTGRES_PASSWORD=postgres \
    -p 5432:5432 \
    -v ${PWD}/data/postgres:/var/lib/postgresql/data \
    -v ${PWD}/data/export:/export postgres:15-alpine
