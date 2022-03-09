lr = 0.0001
batch_size = 128  # don't set batch_size = 1
neg_samples = 2
dimensions = 384
walk_length = 40
num_walks = 10
window_size = 10
p = 1
q = 1
n_layers = 1
bidirectional = True  # the first encoder is not a bidirectional one
dropout = 0.0  # for BIGRU-MAX-RES-N2V
epochs = 1
max_pad = False  # you can remove the zero-padding during max pooling by setting it -> -1e9
hidden_size = 384
gru_encoder = 2  # first encoder: last timesteps, second encoder: residual + max pooling, third encoder: residual + self attention
model = 'rnn'  # models: {'average', 'rnn'}
plot_heatmaps = True
train = True
evaluate = True
evaluate_lr = True
evaluate_cosine = True

# these are used for resuming training..when write_data is True it will save the walks to the hard drive
# so training can continue
resume_training = False
write_data = False


# the default dataset is the part-of.
input_edgelist = 'REPLACE BY YOUR PATH/Graphex/data/apo/graph.txt'
dataset_name = 'graphex'
output_file = 'word_embeddings.emb'


train_pos = './datasets/graphex_easy_splits/graphex_train_pos.p'
train_neg = './datasets/graphex_easy_splits/graphex_train_neg.p'
test_pos = './datasets/graphex_easy_splits/graphex_test_pos.p'
test_neg = './datasets/graphex_easy_splits/graphex_test_neg.p'
train_graph = './datasets/graphex_easy_splits/graphex_train_graph.edgelist'

phrase_dic = 'REPLACE BY YOUR PATH/Graphex/data/apo/reversed_dic_term.p'
# phrase_dic = 'REPLACE BY YOUR PATH/Graphex/data/apo/reversed_dic_def.p'

checkpoint_dir = './checkpoints/'
embeddings_dir = './embeddings/'
checkpoint_name = 'checkpoint_epoch_1.pth.tar'
# specify the checkpoint you want to load
checkpoint_to_load = 'checkpoint_epoch_1.pth.tar'
