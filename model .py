import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce
from pytorch3d.ops import knn_points
from util import batch_index_select, channel_shuffle


#ARPE: Absolute Relative Position Encoding
class ARPE(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, npoints=1024):
        super(ARPE, self).__init__()

        N0 = 512
        k0 = 32
        self.k = int(k0 * npoints / N0)



        self.lin1 = nn.Linear(2*in_channels, 2*in_channels)
        self.lin2 = nn.Linear(2*in_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(2*in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.max_pooling_layer = Reduce('bn k f -> bn 1 f', 'max')
     
    def forward(self, x):
    
        B, N, C = x.shape  # B: batch size, N: number of points, C: channels

        knn = knn_points(x, x, K=self.k, return_nn=True)[2] # B, N, K, C

        diffs = x.unsqueeze(2) - knn  # B, N, K, C
        x = torch.cat([x.unsqueeze(2).repeat(1, 1, self.k, 1), diffs], dim=-1) # B, N, K, 2*C
        x = F.elu(self.bn1(self.lin1(x.view(B*N, self.k, 2*C)).transpose(1,2)).transpose(1,2)) # B*N, K, 2*C
        x = self.max_pooling_layer(x).squeeze(2) # B*N, 1, 2*C -> B*N, 2*C
        x = F.elu(self.bn2(self.lin2(x.view(B, N, 2*C)).transpose(1,2)).transpose(1,2)) # B, N, out_channels

        return x # B, N, 2*C

#GSA: Group shuffling attention
class GSA(nn.Module):
    def __init__(self, channels, groups=1) -> None:
        super(GSA, self).__init__()

        self.channels = channels
        self.groups = groups
        assert self.channels % self.groups == 0, "C must be divisible by groups"
        self.cg = self.channels // self.groups
        self.linears = [nn.Linear(self.cg, self.cg) for _ in range(self.groups)]
        self.gn = nn.GroupNorm(self.groups, self.channels)

    def forward(self, x, mask=None):

        B, N, C = x.shape

        xin = x # B, N, C

        #grouped_x = x.reshape(B, N, C//self.groups, self.groups) # B, N, C//groups, groups

        #Si può vettorizzare?
        x_g =[]
        for i in range(self.groups):
            x = self.linears[i](xin[:,:,i*self.cg:(i+1)*self.cg]) # B, N, C//groups
            x = F.scaled_dot_product_attention(x,x,F.elu(x),mask=mask)
            x_g.append(x)
        x = torch.cat(x_g, dim=-1) # B, N, C

        x = self.gn((channel_shuffle(x, self.groups) + xin).transpose(1,2)).transpose(1,2) # B, N, C

        return x
    
    
#GSS: Gumbel Softmax Sampling TODO  
class GSS(nn.Module):
    def __init__(self, in_ch) -> None:
        super(GSS).__init__()

        raise NotImplementedError
        linear = nn.Linear(1, 1)

    def forward(self, x, tau=1.0):
        return x

#DropPredictor from dynamicViT
class DropPredictor(nn.Module):
    """ Computes the log-probabilities of dropping a token, adapted from PredictorLG here:
    https://github.com/raoyongming/DynamicViT/blob/48ac52643a637ed5a4cf7c7d429dcf17243794cd/models/dyvit.py#L287 """
    def __init__(self, embed_dim):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / (torch.sum(policy, dim=1, keepdim=True)+1e-20)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)
    
#Classification head
class Classifier(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        lin1 = nn.Linear(in_channels, in_channels//2)
        lin2 = nn.Linear(in_channels//2, n_classes)
        bn1 = nn.BatchNorm1d(in_channels//2)
        dp = nn.Dropout(0.5)
        self.classifier = nn.Sequential(lin1, bn1, nn.GELU(), dp, lin2)

    def forward(self, x):
        return self.classifier(x)

#Whole model
class AdaPT(nn.Module):
    def __init__(self, embed_dim, n_classes, n_blocks = 3, drop_loc=[], drop_target=[]) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.drop_loc = drop_loc
        self.drop_target = drop_target

        self.arpe = ARPE()
        self.blocks = nn.ModuleList([GSA(channels=self.embed_dim) for _ in range(self.n_blocks)])
        self.predictors = nn.ModuleList([DropPredictor(self.embed_dim) for _ in range(len(self.drop_loc))])
        self.classifier = Classifier(self.embed_dim, self.n_classes)
        

    def forward(self, x, drop_temp=1.0):

        B, N, C = x.shape

        x = self.arpe(x)
        
        prev_decision = torch.ones(B, N, 1, dtype=x.dtype, device=x.device)
        p = 0

        for i in range(self.n_blocks):
            if i in self.drop_loc:
                pred_score = self.score_predictor[p](x, prev_decision)
                p+=1
                # Slow warmup
                keepall = torch.cat((torch.zeros_like(pred_score[:,:,0:1]), torch.ones_like(pred_score[:,:,1:2])),2) 
                pred_score = pred_score*drop_temp + keepall*(1-drop_temp)
                if self.training:
                    pred_score = torch.log(pred_score + 1e-8)
                    decision = F.gumbel_softmax(pred_score, tau=1.0, hard=True, dim=-1)[:,:,1:2]*prev_decision
                    prev_decision = decision
                    mask = (decision*decision.transpose(1,2)).bool()


                else:
                    score = pred_score[:,:,1]
                    num_keep_tokens = int((1-self.args.train.drop_ratio[p]) * (N))
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_tokens]
                    x = batch_index_select(x, keep_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    mask = None

            x = self.blocks[i](x, mask=mask)

        x = self.classifier(x)
        return x
    

