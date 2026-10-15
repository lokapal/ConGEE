#!/bin/sh
python3 Calibrate.Dispersion.Table.py \
--genes hg38.HEK293.bed --domains TADMap_hg38.bed \
--replicates 10000 \
--output TADmap_f.txt
