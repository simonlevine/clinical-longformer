#!/usr/bin/env bash


# Simon's physionet password (do NOT share I WILL sue you.)
# Mivdo9-merjuh-gokwuc

# MIMIC-III:
cd data

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz
wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv.gz
wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimiciii/1.4/D_ICD_DIAGNOSES.csv.gz
wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimiciii/1.4/D_ICD_PROCEDURES.csv.gz
# MIMIC-CXR

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip
wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz

cd data && unzip physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip -d physionet.org/files/mimic-cxr/2.0.0/
rm -rf physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip

# MIMIC-iii ANNOTATED

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/phenotype-annotations-mimic/1.20.03/ACTdb102003.csv

#MEDNLI

wget -r -N -c -np --user slevineg --password Mivdo9-merjuh-gokwuc https://physionet.org/files/mednli/1.0.0/
unzip physionet.org/files/mednli/1.0.0/*.zip -d physionet.org/files/mednli/1.0.0/
rm -rf physionet.org/files/mednli/1.0.0/*.zip

cd ..