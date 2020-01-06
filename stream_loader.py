# https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader, IterableDataset
from itertools import islice, cycle

class CustomeDBLoader(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class IterCustomeDBLoader(IterableDataset):
    def __init__(self, data):
        self.data = data
        
    def process_data(self, data):
        for x in data:
            yield x
    
    def get_stream(self, data):
        return cycle(self.process_data(data))
        
    def __iter__(self):
        return self.get_stream(self.data)


data = torch.arange(0,12)    
map_data = IterCustomeDBLoader(data)
loader = DataLoader(map_data, batch_size=4)

for batch in islice(loader, 2):
    print(batch)


class TextIterCustomeDBLoader(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path
        
    def parse_file(self, file_path):
        with open(file_path, 'r') as file_obj:
            for line in file_obj:
                tokens = line.strip('\n').split(' ')
                yield from tokens
                
    def get_stream(self, file_path):
        return cycle(self.parse_file(file_path))
    
    def __iter__(self):
        return self.get_stream(self.file_path)
    
    
dataSet = TextIterCustomeDBLoader('core.txt')
loader = DataLoader(dataSet, batch_size=5)

for batch in islice(loader, 3):
    print(batch)