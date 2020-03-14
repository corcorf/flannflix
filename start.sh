#!/bin/bash
export FLASK_APP=web_server.py
# export FLASK_DEBUG=True
export USE_DASK=false
flask run
