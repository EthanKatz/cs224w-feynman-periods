#!/bin/bash
# Download the Balduf & Shaban phi^4 period files for L = 3..12 from Borealis Dataverse.
# File ids come from https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:10.5683/SP3/NLEDGH
set -e
cd "$(dirname "$0")/../data/raw"
B=https://borealisdata.ca/api/access/datafile
dl(){ [ -f "$2" ] || curl -sL "$B/$1" -o "$2"; }
dl 719080 README.txt
dl 719015 periods_03.csv; dl 719013 periods_04.csv; dl 719019 periods_05.csv
dl 719014 periods_06.csv; dl 719018 periods_07.csv; dl 719016 periods_08.csv
dl 719017 periods_09.csv; dl 719007 periods_10.csv
dl 719006 periods_11.csv.bz2; dl 719008 periods_12.csv.bz2
# dl 719002 periods_13.csv.bz2   # 94 MB, all ~0.9M 13-loop graphs (next step)
