#!/bin/bash
export DYLD_LIBRARY_PATH='/usr/local/cuda/lib:glfw/lib';
./bin/565raytracer "$1" "$2" 
