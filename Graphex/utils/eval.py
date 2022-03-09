import torch

import numpy as np

from utils import config
from utils.metric import moses_multi_bleu
from utils.beam_omt import Translator
from utils.beam_omt_transformer import Translator as TrsTranslator
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)


def evaluate_graphex(model, data, data_loader_all=None, model_name='trs', ty='valid', writer=None, n_iter=0, ty_eval="before", verbose=False, log=False, result_file="results/results_transformer.txt", ref_file="results/ref_transformer.txt", case_file="results/case_transformer.txt"):
    t = Translator(model, model.vocab)
    loss,ppl,bleu_score_b = evaluate(model, data, data_loader_all, model_name, ty, writer, n_iter, ty_eval, verbose, log, result_file, ref_file, case_file, t)
    return loss,ppl,bleu_score_b



def evaluate_transformer(model, data, data_loader_all=None, model_name='trs', ty='valid', writer=None, n_iter=0, ty_eval="before", 
                         verbose=False, log=False, result_file="results/results_transformer.txt", ref_file="results/ref_transformer.txt", case_file="results/case_transformer.txt"):

    t = TrsTranslator(model, model.vocab)
    loss,ppl,bleu_score_b = evaluate(model, data, data_loader_all, model_name, ty, writer, n_iter, ty_eval, verbose, log, result_file, ref_file, case_file, t)
    return loss,ppl,bleu_score_b


def evaluate(model, data, data_loader_all=None, model_name='trs', ty='valid', writer=None, n_iter=0, ty_eval="before", 
             verbose=False, log=False, result_file="results/results_transformer.txt", ref_file="results/ref_transformer.txt", case_file="results/case_transformer.txt", t=None):

    if log:
        f1 = open(result_file, "w")
        f2 = open(ref_file, "w")
    dial,ref, hyp_b, per= [],[],[], []

    l = []
    p = []
    pbar = tqdm(enumerate(data),total=len(data))
    for j, batch in pbar:

        torch.cuda.empty_cache()
        loss, ppl, _ = model.train_one_batch(batch, data_loader_all, train=False)
        l.append(loss)
        p.append(ppl)

        if ( j < 3 and ty != "test") or ty == "test": 

            sent_b, _ = t.translate_batch(batch, data_loader_all)

            for i in range(len(batch["target_txt"])):
                new_words = []
                for w in sent_b[i][0]:
                    if w==config.EOS_idx:
                        break
                    new_words.append(w)
                    if len(new_words)>2 and (new_words[-2]==w):
                        new_words.pop()
                
                sent_beam_search = ' '.join([model.vocab.index2word[idx] for idx in new_words])
                hyp_b.append(sent_beam_search)
                if log:
                    f1.write(sent_beam_search)
                    f1.write("\n")
                ref.append(batch["target_txt"][i])
                if log:
                    f2.write(batch["target_txt"][i])
                    f2.write("\n")
                dial.append(batch['input_txt'][i])

        pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),np.mean(p)))
        torch.cuda.empty_cache()

        if j > 4 and ty == "train":
            break

    loss = np.mean(l)
    ppl = np.mean(p)

    bleu_score_b = moses_multi_bleu(np.array(hyp_b), np.array(ref), lowercase=True)

    if log:
        f1.close()
        f2.close() 
        log_all(dial,ref,hyp_b, case_file)

    return loss, ppl, bleu_score_b


def print_all(dial, ref, hyp_b, max_print):
    for i in range(len(ref)):
        print(pp.pformat(dial[i]))
        print("Beam: {}".format(hyp_b[i]))
        print("Ref:{}".format(ref[i]))
        print("----------------------------------------------------------------------")
        print("----------------------------------------------------------------------")
        if i > max_print:
            break

def log_all(dial, ref, hyp_b, log_file):
    f = open(log_file, "a")
    for i in range(len(ref)):
        f.write(pp.pformat(dial[i]))
        f.write("\n")
        f.write("generate: {}".format(hyp_b[i]))
        f.write("\n")
        f.write("def:{}".format(ref[i]))
        f.write("\n")
        f.write("----------------------------------------------------------------------")
        f.write("\n")
    f.close()