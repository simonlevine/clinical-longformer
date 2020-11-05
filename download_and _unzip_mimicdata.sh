#!/usr/bin/env bash


# Simon's physionet password:
# Mivdo9-merjuh-gokwuc

# MIMIC-III:
cd data

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz 

# MIMIC-CXR

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz

unzip physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip -d physionet.org/files/mimic-cxr/2.0.0/

# MIMIC-iii ANNOTATED

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/phenotype-annotations-mimic/1.20.03/ACTdb102003.csv

cd ..