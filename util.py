import torch

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError
    
def channel_shuffle(x, groups):
    B, N, C = x.shape
    x = x.reshape(B,N,C//groups,groups).permute(0,1,3,2).reshape(B,N,C)
    return x


def calc_tau(start, end, epoch):
    if epoch<start:
        tau = 0
    elif epoch>end:
        tau = 1
    else:
        tau = (epoch-start)/(end-start)
    return tau


