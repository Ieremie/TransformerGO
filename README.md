# TransformerGO

This repository contains the official implementation of the paper: 
"TransformerGO: Predicting protein-protein interactions by modelling the attention between sets of gene ontology terms"

**For more details, see:** [TransformerGO: Predicting protein-protein interactions by modelling the attention between sets of gene ontology terms](hyperlink). 

<img src="https://github.com/Ieremie/old-transformerGO/blob/main/TransformerGO.png" width="75%" height="75%">


## About
TransformerGO is a model based on the orginal Transformer architecture. It is used to capture deep semantic similarities between gene ontology terms and predict protein to protein interactions. Introduced in our paper [TransformerGO: Predicting protein-protein interactions by modelling the attention between sets of gene ontology terms](hyperlink). 


## Contents
* [Datasets](#datasets)
* [Generating GO term embeddings](#generating-go-term-embeddings)
* [Training and testing the model](#training-and-testing-the-model)
    * [LSTM](#lstm)
    * [TransformerGO](#transformergo)
* [Attention analysis](#attention-analysis)
    * [Generating heatmaps](#generating-heatmaps)
    * [Analysing the attention for one interaction](#analysing-the-attention-for-one-interaction)
    
## Datasets
The model is trained and evaluated using datasets for two organisms *S. Cerevisiae* and *H. Sapiens*
- TCSS datasets: [An improved method for scoring protein-protein interactions using semantic similarity within the gene ontology](http://baderlab.org/Software/TCSS)  
- StringDB benchmark: [Semantic similarity and machine learning with ontologies](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies)
- TransformerGO datasets are created using the String database. The following notebook can be used to generete **new** datasets for the organisms available on StringDB:[`dataset.ipynb`](./datasets/transformerGO-dataset/dataset.ipynb).


## Generating GO term embeddings
To generate embeddings for Gene Ontology terms, we use the original implementation of [node2vec: Scalable Feature Learning for Networks](https://github.com/aditya-grover/node2vec). 
Parsing the .obo file is done using [`obo-file-parsing.ipynb`](./term-encoding-module/obo-file-parsing.ipynb) and the generation of the edge list (input for node2vec) using 
[`node2vec-embeddings.ipynb`](./node2vec-embeddings.ipynb). 
An example of running node2vec:
```bash
#!/bin/bash
input_path="../../../datasets/transformerGO-dataset/go-terms/graph/go-terms.edgelist"
output_path="../../../datasets/transformerGO-dataset/go-terms/emb/go-terms.emd"
python node2vec-master/src/main.py --input $input_path --output $ouput_path --dimensions 64 --iter 10

```

## Training and testing the model
### LSTM
Training and testing the implementation of protein2vec can be done via [`training-protein2vec.ipynb`](./training-testing/training-protein2vec.ipynb)
Changing the organism or the subset can be done by changing the following paths.
```python
neg_path = "datasets/jains-TCSS-datasets/yeast_data/iea+/negatives.sgd.iea.f"
poz_path = "datasets/jains-TCSS-datasets/yeast_data/iea+/positives.sgd.iea.f"
```
### TransformerGO
Training and testing the implementation of TransformerGO can be done via [`training-transformerGO.ipynb`](./training-testing/training-transformerGO.ipynb)

Multiple datasets are available, and these can be chosen by running the code block corresponding to the desired dataset (TCSS, StringDB benchmark or our datasets).
Running experiements using specific gene ontology terms or different annotation sizes can be run by changing the following variables:
```python
intr_set_size_filter = [0,5000]
go_filter = 'CC'
```

## Attention analysis
### Generating heatmaps 
Generating heatmaps requires a model trained on the same dataset we are analysing. The heatmaps contain the aggregation of the attention weights after passing through each positive interaction from the dataset. Heatmaps can be generated via [`attention-plots.ipynb`](./attention-analysis/attention-plots.ipynb). Examples generated using the StringDB benchmark for *S. Cerevisiae* and *H. Sapiens* can be found in [`attention-heatmaps`](./attention-analysis/attention-heatmaps.ipynb).

### Analysing the attention for one interaction
In [`attention-per-interaction.ipynb`](./attention-analysis/attention-per-interaction.ipynb) we provide a notebook which can be used to analyse the attention values between GO terms given a single interaction. Here the heatmaps for each head and layer are generated.

## Authors
Ioan Ieremie, Rob M. Ewing, Mahesan Niranjan

## Citation
[To be added]

## Contact
ii1g17 [at] soton [dot] ac [dot] uk
