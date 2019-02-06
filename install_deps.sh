#!/usr/bin/env bash
#get virtual env ish
sudo apt-get install python-pip
sudo apt-get install git
pip install env
virtualenv env
./env/bin/pip install -r requirements_cpu.txt
mkdir weights
# curl the weights

