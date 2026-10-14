@echo off
python3 generate_random.py --genes hg38.HEK293.bed ^
--mean-domain-size 1000000 --mean-gap 40000 --gap-mode normal --expression uniform ^
--n-files 10000 --cores 120 --outdir random.TAD.1Mbp
