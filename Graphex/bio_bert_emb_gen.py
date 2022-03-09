from biobert_embedding.embedding import BiobertEmbedding
from tqdm import tqdm 

# pretrained biobert embeddings
f = open("data/vocab.txt", "r")
f2 = open("vectors/embeddings.txt", "w")

lines = f.readlines()
biobert = BiobertEmbedding(model_path="/home/caiyang/Documents/Summer 2022 Materials/codes/Graphex/biobert_v1.1_pubmed_pytorch_model/")

for l in tqdm( lines ):
    word_embeddings = biobert.word_vector(l[:-1])
    f2.write(l[:-1])    
    for e in word_embeddings[0]:
        f2.write(" ")
        f2.write(str(e)[7:-1])
    f2.write("\n")
f2.close()