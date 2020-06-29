
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable
    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu,\
     dicts, embed_size=100, dropout=0.5, code_emb=None):
        super(ConvAttnPool, self).__init__(Y, embed_file, dicts, lmbda, \
                dropout=dropout, gpu=gpu, embed_size=embed_size)
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, \
                kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform_(self.conv.weight)
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform_(self.U.weight)
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform_(self.final.weight)

    def forward(self, x, target):
        x = self.embed(x)
        x = self.embed_drop(x).transpose(1, 2)
        x = torch.tanh(self.conv(x).transpose(1,2))
        alpha = self.U.weight.matmul(x.transpose(1,2))
        m = alpha.matmul(x)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

import gensim.models.word2vec as w2v
def word_embeddings(Y, notes_file, embedding_size, min_count, n_iter):
    filename = "processed_%s.w2v" % (Y)
    iterator = ProcessedIter(Y, notes_file)

    model = w2v.Word2Vec(size=embedding_size, min_count=min_count,\
     workers=4, iter=n_iter)
    
    model.build_vocab(iterator)
    model.train(iterator, total_examples=model.corpus_count, epochs=model.iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    model.save(out_file)
    return out_file

