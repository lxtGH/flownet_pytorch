#!/bin/bash
cd ./packge/correlation_package
./make.sh
cd ../resample2d_package
./make.sh
cd ../channelnorm_package
./make.sh
cd ..
