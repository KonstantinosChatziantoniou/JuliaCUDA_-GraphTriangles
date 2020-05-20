#!/bin/bash

wget http://statml.com/download/data_7z/dimacs10/auto.7z
7z x auto.7z
rm -rf readme.html
#rm -rf auto.7z
wget http://statml.com/download/data_7z/dimacs10/delaunay_n24.7z
7z x delaunay_n24.7z
rm -rf readme.html
#rm -rf delaunay_n24.7z
wget http://statml.com/download/data_7z/dimacs10/inf-great-britain_osm.7z
rm -rf readme.html
