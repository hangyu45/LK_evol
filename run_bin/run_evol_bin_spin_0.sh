#!/bin/bash

for k in {0..9};
do
    python evol_bin_spin.py \
	--idx $k \
	--out-base run_0_ \
	--out-dir q_0.8_x1_0.7_x2_0.7_data/ \
	--Mt-Ms 50 \
	--qq 0.8 \
	--chi1 0.7 \
	--chi2 0.7 
done
