#%%
sentence = 'Life is shourt, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

#%%
import torch

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
print(sentence_int)

#%%
vocab_size = 50_000

torch.manual_seed(123)
embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)

#%%
d = embedded_sentence.shape[1]
d_q, d_k, d_v = 2, 2, 4

W_query = torch.nn.Parameter(torch.randn(d, d_q))
W_key = torch.nn.Parameter(torch.randn(d, d_k))
W_value = torch.nn.Parameter(torch.randn(d, d_v))   