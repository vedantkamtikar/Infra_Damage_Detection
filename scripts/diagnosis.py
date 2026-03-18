#using this file to run random code snippets for diagnosis and debugging



import torch
ckpt = torch.load(r'runs/enim_run4/weights/last.pt', map_location='cpu', weights_only=False)
print('Epoch:', ckpt.get('epoch', 'unknown'))
print('Best fitness:', ckpt.get('best_fitness', 'unknown'))


