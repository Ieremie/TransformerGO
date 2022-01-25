#!/bin/bash

input_path="../../../datasets/transformerGO-dataset/go-terms/graph/go-terms.edgelist"
output_path="../../../datasets/transformerGO-dataset/go-terms/emb/go-terms.emd"

python node2vec-master/src/main.py --input $input_path --output $ouput_path --dimensions 64 --iter 10
