import torch
import torch.nn as nn

import math
# from model.common_layer import EncoderLayer, DecoderLayer, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask,  get_input_from_batch2, get_output_from_batch

# from transformer.transformer_layers.layers import *
# from transformer.transformer_layers.main import *
# from transformer.transformer_layers.sublayers import *

from transformer.transformer_utils import _get_attn_subsequent_mask, get_input_from_batch2

from transformer.transformer import *

from utils import config
import random
from numpy import random
import os
import pprint
pp = pprint.PrettyPrinter(indent=1)
import os

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class Graphex(nn.Module):

    def __init__(self, vocab, model_file_path=None, is_eval=False, load_optim=False):
        super(Graphex, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab,config.preptrained)
        self.encoder = Encoder(3 * config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter)
            
        self.decoder = Decoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter)
        self.generator = Generator(config.hidden_dim,self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if (config.label_smoothing):
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        if is_eval:
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()
            self.generator = self.generator.eval()
            self.embedding = self.embedding.eval()

    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if(config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 4000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        if config.use_sgd:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=config.lr)
        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            print("LOSS",state['current_loss'])
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])

        if (config.USE_CUDA):
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.generator = self.generator.cuda()
            self.criterion = self.criterion.cuda()
            self.embedding = self.embedding.cuda()
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, data_loader_all=None, train=True):
        ## pad and other stuff
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _, _, node_emb, node_emb2 = get_input_from_batch2(batch)
        dec_batch, _, _, _, _, _ = get_output_from_batch(batch)
        
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        length = enc_batch.size(1)
        node_embedding = node_emb.unsqueeze(1).repeat(1, length, 1)
        node_embedding_def = node_emb2.unsqueeze(1).repeat(1, length, 1)

        encoder_outputs = self.encoder( torch.cat( (self.embedding(enc_batch), node_embedding, node_embedding_def), 2), mask_src)

        # Decode 
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1)

        tgt_mask = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        
        mask = _get_attn_subsequent_mask(config.max_enc_steps)
        dec_mask = torch.gt(tgt_mask.type(torch.uint8) + mask[:, :tgt_mask.size(-1), :tgt_mask.size(-1)].type(torch.uint8), 0)

        pre_logit, attn_dist = self.decoder( self.embedding(dec_batch_shift), encoder_outputs, mask_src, dec_mask  )

        ## compute output dist
        logit = self.generator(pre_logit,attn_dist,enc_batch_extend_vocab, extra_zeros)
        
        ## loss: NNL if ptr else Cross entropy
        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        if(train):
            loss.backward()
            self.optimizer.step()
        if(config.label_smoothing): 
            loss = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        
        return loss.item(), math.exp(min(loss.item(), 100)), loss
