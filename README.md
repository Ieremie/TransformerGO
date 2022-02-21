# TransformerGO

This repository contains the official implementation of the paper: 
"TransformerGO: Predicting protein-protein interactions by modelling the attention between sets of gene ontology terms"

**For more details, see:** [TransformerGO: Predicting protein-protein interactions by modelling the attention between sets of gene ontology terms](https://doi.org/10.1093/bioinformatics/btac104). 

<img src="https://github.com/Ieremie/TransformerGO/blob/main/TransformerGO.png" width="75%" height="75%">


## About
TransformerGO is a model based on the orginal Transformer architecture. It is used to capture deep semantic similarities between gene ontology terms and predict protein to protein interactions. Introduced in our paper [TransformerGO: Predicting protein-protein interactions by modelling the attention between sets of gene ontology terms](https://doi.org/10.1093/bioinformatics/btac104). 


## Contents
* [Datasets](#datasets)
* [Generating GO term embeddings](#generating-go-term-embeddings)
* [Training and testing the model](#training-and-testing-the-model)
    * [LSTM](#lstm)
    * [TransformerGO](#transformergo-model)
* [Attention analysis](#attention-analysis)
    * [Generating heatmaps](#generating-heatmaps)
    * [Analysing the attention for one interaction](#analysing-the-attention-for-one-interaction)
    
## Datasets
The model is trained and evaluated using datasets for two organisms *S. Cerevisiae* and *H. Sapiens*
- TCSS datasets: [An improved method for scoring protein-protein interactions using semantic similarity within the gene ontology](http://baderlab.org/Software/TCSS)  
- StringDB benchmark: [Semantic similarity and machine learning with ontologies](https://github.com/bio-ontology-research-group/machine-learning-with-ontologies)
- TransformerGO's interaction datasets are created using the String database. The following notebook can be used to generete **new** datasets for organisms available on StringDB:[`dataset.ipynb`](./datasets/transformerGO-dataset/dataset.ipynb). To generate a new dataset compatible with our model multiple files must be downloaded. An example for *H. Sapiens*:
```python
#Annotation data
url = 'http://geneontology.org/gene-associations/goa_human.gaf.gz'
#Protein to protein interactions
url = 'https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz'
#Protein aliases 
url = 'https://stringdb-static.org/download/protein.aliases.v11.5/9606.protein.aliases.v11.5.txt.gz'
```


## Generating GO term embeddings
To generate embeddings for Gene Ontology terms, we use the original implementation of [node2vec: Scalable Feature Learning for Networks](https://github.com/aditya-grover/node2vec). 
Parsing the .obo file is done using [`obo-file-parsing.ipynb`](./term-encoding-module/obo-file-parsing.ipynb) and the generation of the edge list (input for node2vec) using 
[`node2vec-embeddings.ipynb`](./term-encoding-module/node2vec-embeddings.ipynb). 
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
### TransformerGO model
Training and testing the implementation of TransformerGO can be done via [`training-transformerGO.ipynb`](./training-testing/training-transformerGO.ipynb)

Multiple datasets are available, and these can be chosen by running the code block corresponding to the desired dataset (TCSS, StringDB benchmark or our datasets).
Running experiements using specific gene ontology terms or different annotation sizes can be run by changing the following variables:
```python
intr_set_size_filter = [0,5000]
go_filter = 'CC'
```
To train or test the model using a new dataset, simply provide the paths to the interaction data, annotation file and the aliases file as follows:
```python
organism = 9606
EMB_DIM = 64
data_path = 'datasets/transformerGO-dataset/'
go_embed_pth = data_path + f"go-terms/emb/go-terms-{EMB_DIM}.emd"
go_id_dict_pth = data_path + "go-terms/go_id_dict"
protein_go_anno_pth = data_path +"stringDB-files/goa_human.gaf.gz"
alias_path = data_path + f'stringDB-files/{organism}.protein.aliases.v11.5.txt.gz'

neg_path = data_path + f'interaction-datasets/{organism}.protein.negative.v11.5.txt'
poz_path= data_path + f'interaction-datasets/{organism}.protein.links.v11.5.txt'
```
Note that the embedings could also be changed by running node2vec on a completly different Gene Ontology graph.

## Attention analysis
### Generating heatmaps 
The heatmaps contain the aggregation of the attention weights after passing through each positive interaction from the dataset. Heatmaps can be generated via [`attention-plots.ipynb`](./attention-analysis/attention-plots.ipynb). Examples generated using the StringDB benchmark for *S. Cerevisiae* and *H. Sapiens* can be found in [`attention-heatmaps`](./attention-analysis/attention-heatmaps). To generate heatmaps on a new dataset, change the paths as shown in the example above. Note that the examples in the paper are generated using only positive interactions from the training dataset.

### Analysing the attention for one interaction
In [`attention-per-interaction.ipynb`](./attention-analysis/attention-per-interaction.ipynb) we provide a notebook which can be used to analyse the attention values between GO terms given a single interaction. Here the heatmaps for each head and layer are generated.

## Authors
Ioan Ieremie, Rob M. Ewing, Mahesan Niranjan

## Citation
```
@article{10.1093/bioinformatics/btac104,
    author = {Ieremie, Ioan and Ewing, Rob M and Niranjan, Mahesan},
    title = "{TransformerGO: Predicting protein-protein interactions by modelling the attention between sets of gene ontology terms}",
    journal = {Bioinformatics},
    year = {2022},
    month = {02},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac104},
    url = {https://doi.org/10.1093/bioinformatics/btac104},
    note = {btac104},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btac104/42546304/btac104.pdf},
}
```

## Contact
ii1g17 [at] soton [dot] ac [dot] uk
