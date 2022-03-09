## Reproduction on Graphine: A Dataset for Graph-aware Terminology Definition Generation

This repo contains the reproduction on the paper: **Graphine: A Dataset for Graph-aware Terminology Definition Generation** from EMNLP 2021. 

The code is based on the [original repo](https://github.com/zequnl/Graphex) with some modifications. The following subsections show the instructions for running the code by order. 


## Data Preparation

### Dataset Download 

The full dataset can be downloaded from [zenodo](https://zenodo.org/record/5320310#.YVlnnZrP02w). Unzip the folder and leave them in the same folder as this repo. For the purpose of replicating the model, it is recommended that you only choose a few subsets (e.g. one or two subsets) for experiments.


### Dataset Split and Preparation for Transformer

The data within each subset is further split into training, validation and test set according to the ratio mentioned in the paper: 0.7, 0.1, 0.2.


As mentioned by the end of the section 3.2.1 from the paper - a Transformer is trained on the terminology-definition pairs to generate definition for names without any definition. The training set is obtained by merging all the training terminologies and definitions from each subset in the data folder. Similar for the validation data.


The following command will (1) split the data and save them into the folder for each DAG (2) merge the training and validation data from each subset and save them under `data_partial/`


```

python3 dataset_split.py

```


### Transformer Training and Testset Definition Generation

The Transformer is based on TorchNLP, link is [here](https://github.com/kolloldas/torchnlp).


To train the Transformer and generate definitions, use the following command:

```
python definition_prepare.py --cuda --model trs --pretrain_emb --noam --label_smoothing --emb_dim 768 --hidden_dim 768 --latent_dim 600 --batch_size 8 --max_enc_steps 250

```

The generated definitions will be saved as `test_def_gen.txt` under each DAG folder in the data folder.


**Note 1:** The data to transformer does not rely on any of the later embeddings, and hence theoretically any data prepartion on the embeddings is redundant. I removed the related codes on node embeddings from `data_reader.py` and save the new one as `data_reader_transformer_only.py`. 



**Note 2:** The beam search used during the decoding steps may raise an error like: `IndexError: tensors used as indices must be long, byte or bool tensors`. There are two possible ways of fixing this: (1) downgrade the Pytorch version (if this is the source of error); (2) replacing the line `prev_k = best_scores_id / num_words` by `prev_k = best_scores_id // num_words`. Details on this problem can be found at [this issue](https://github.com/jacobswan1/Video2Commonsense/issues/3) and [this issue] from Github(https://github.com/VISLANG-Lab/MLGCN-DP/issues/3).


### BioBERT Embedding Generation - Local Semantic Embedding

The BioBERT embedding is generated for terminologies as local semantic embedding. The original code can be found from [biobert_embedding](https://github.com/Overfitter/biobert_embedding). The library can be installed separately through pip: `pip install biobert-embedding==0.1.2`, but it seems does support the latest Pytorch. The scripts are used directly in my case.

Note that the code requires downloading the pretrained BioBERT before embedding generation. One can find these files (`config.json`, `pytorch_model.bin`, `vocab.txt`) from [HuggingFace](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1). The files should be left under the folder `biobert_v1.1_pubmed_pytorch_model/`. Run the following command and the generated embeddings are saved as `vectors/embeddings.txt`


```
python3 bio_bert_emb_gen.py

```


### Node2Vec Data Preparation - Global Embedding

Node2Vec is used to obtain the global embeddings for terminologies and definitions. The link to the original repo is [here](https://github.com/SotirisKot/Content-Aware-Node2Vec). There are three files need to be run to obtain the final global embeddings. 


**Under `Graphex/` folder**, run the following command. It will generate `phrase_dic_term.p`, `reversed_dic_term.p`, `phrase_vocab_term.p`, `phrase_dic_def.p`, `reversed_dic_def.p`, `phrase_vocab_def.p` and `graph.txt` that will be used later by the Node2Vec training.

```
python3 data_prepare.py
```


**Under `Content-Aware-Node2vec/` Folder**. First prepare the data for Node2Vec, the following command shows an instance using *argo* DAG.

```
python3 create_dataset.py --input="PATH/Graphex/data/agro/graph.txt" --dataset="graphex" --directed
```

Then, train the model and do the inference. Note: the config.py needs to be modified accordingly: (1) the `hidden size` and the `dimensions` are both set to be 384 to match the hidden dimension (768) used by Graphex later.

```
python3 experiments.py
```

The output `node_embeddings_phrases.p`


## Graphex Training & Inference

Setting up correct paths in the `experiments.py`, then:

```
python experiments.py --cuda --model trs --pretrain_emb --noam --label_smoothing --emb_dim 768 --hidden_dim 768 --latent_dim 600 --batch_size 8 --max_enc_steps 250

```

**Note 3:** The embedding to each node is obtained by indexing the corresponding dictionary, where the index is obtained by the node dictionary. However, the key used by the embedding in this case is the actual string instead of numeric indices. An error may occur in this case. To fix it, simply using the terminoloy and definition names as the keys for indexing. However, the original codes for Node2Vec clean the input text (i.e. terminologies and definitions in this case). To match the keys used by the global embedding and the node dictionary, additional cleaning has to be done on the node dictionary. The script has corrected this.




## References

The following shows a list of external codes used by the repo:

Graphine Original by Zequn Liu: https://github.com/zequnl/Graphex

BioBERT Embedding by Jitendra Jangid: https://github.com/Overfitter/biobert_embedding

Content-Aware Node2vec by Sotiris: https://github.com/SotirisKot/Content-Aware-Node2Vec

Transformer based on TorchNLP by Kollol Das: https://github.com/kolloldas/torchnlp

