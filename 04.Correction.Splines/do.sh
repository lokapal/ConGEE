#!/bin/sh
python3 conc_spline.py precompute --tsv p05.tsv --correction 5%:corr_p05.tsv -o splines_final_lin.h5
