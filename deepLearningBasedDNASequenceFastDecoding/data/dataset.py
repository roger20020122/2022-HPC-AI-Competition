from collections import defaultdict
from typing import List
import torch

DNA_ALPHABET = ['A', 'T', 'C', 'G']

def get_sequence_str(x):
    return ''.join([DNA_ALPHABET[int(i.item())] for i in x[:,1]+x[:,2]*2+x[:,3]*3])
def get_sequence_list(x):
    return [[0,1,2,3][int(i.item())] for i in x[:,1]+x[:,2]*2+x[:,3]*3]

def get_start_sparse(t):
    nonzeros = torch.nonzero(t.squeeze(-1))
    starts = []
    last_j = -2
    for i,j in nonzeros:
        if j != last_j + 1:
            starts.append((i,j))
        last_j = j
    return starts

def get_start_dense(t:torch.Tensor):
    return torch.clamp(t - torch.cat([torch.zeros(t.shape[0],1,1,device=t.device), t[:,:-1]], dim=1),min=0)
     
def get_seq_feature(x, query, l = 3, shift = 1):
    # x : (batch, seq_len, 4)
    fs = []
    for query_seq in query:
        feature_maps : List[torch.Tensor] = []
        for i in range(l):
            current_shift = shift + i
            feature_map = (x[:,:,query_seq[i]] == 1).int()
            feature_map = torch.roll(feature_map, shifts = -current_shift, dims = 1)
            if current_shift < 0:
                feature_map[:,current_shift:] = 0
            if current_shift > 0:
                feature_map[:,:current_shift] = 0
            
            feature_maps.append(feature_map)
        feature = 1
        for feature_map in feature_maps:
            feature *= feature_map
        fs.append(feature.cpu()) # type: ignore
    return torch.stack(fs, dim=2)

class DNADataset():
    def __init__(self,path,dev = 'cuda') -> None:
        super().__init__()
        print('loading data from',path)
        self.dev = dev
        self.x, self.y = torch.load(path)
        self.x = torch.tensor(self.x,dtype=torch.float32,device=self.dev)
        self.y = torch.tensor(self.y,dtype=torch.float32,device=self.dev)
        self.start_sparse = get_start_sparse(self.y)
        self.start_dense = get_start_dense(self.y)
        self.feature = self.x
        
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.feature[idx], self.y[idx]

    def rank_seq(self,shift = 1,l = 3):
        d = defaultdict(int)
        
        for i,j in self.start_sparse:
            try:
                seq = self.x[i,j+shift:j+shift+l]
                d[tuple(get_sequence_list(seq))] += 1
            except IndexError:
                pass

        rank = sorted(d.items(), key=lambda x: x[1], reverse=True)

        return rank
    
    def get_feature(self,l=3,d=10,shifts=range(0,1), set_as_feature = False):
        features = []
        for i in shifts:
            rank = self.rank_seq(i,l)
            query = [x[0] for x in rank[:d]]
            features.append(get_seq_feature(self.x, query, shift=i))
        feature = torch.cat(features, dim=2)
        feature = torch.cat([feature.float(),self.x[:,:,-1].unsqueeze(-1).cpu()],dim=2)
        if set_as_feature:
            self.feature = feature
            print('using feature:','l',l,'d',d,'shifts',shifts,feature.shape)
        return feature