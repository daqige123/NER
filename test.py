import torch
# memory = {'Thousands': 'O', 'of': 'O', 'demonstrators': 'O', 'have': 'O', 'marched': 'O', 'through': 'O', 'London': 'B-geo', 'to': 'O', 'protest': 'O', 'the': 'O', 'war': 'O', 'in': 'O', 'Iraq': 'B-geo', 'and': 'O'}
# X = ['Thousands', 'of', 'demonstrators', 'have', 'marched', 'through', 'London', 'to', 'protest', 'the', 'war', 'in', 'Iraq', 'and']
# a = [memory.get("iisaij", "O")]
# print(a)

a = torch.randn(2, 2)
print(a)
for ii in a:
    for i in ii:
        print(float(i>0))