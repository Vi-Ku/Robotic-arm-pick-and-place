#! /bin/bash
# This script install all the required module for execution of the python script.
sudo apt-get update -y
sudo apt-get upgrade -y
pip install sympy==0.7.1
pip install mpmath
