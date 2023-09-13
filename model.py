import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce
from pytorch3d.ops import knn_points
from util import batch_index_select, channel_shuffle, calc_tau
import pytorch_lightning as pl
from entmax import sparsemax, entmax15, entmax_bisect
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        self.linears = nn.ModuleList([nn.Linear(self.cg, self.cg) for _ in range(self.groups)])
        self.gn = nn.GroupNorm(self.groups, self.channels)

    def forward(self, x, mask=None):

        B, N, C = x.shape

        xin = x # B, N, C
        #grouped_x = x.reshape(B, N, C//self.groups, self.groups) # B, N, C//groups, groups
        
        #Si può vettorizzare?
        x_g =[]
        for i in range(self.groups):
            x = self.linears[i](xin[:,:,i*self.cg:(i+1)*self.cg]) # B, N, C//groups
            x = F.scaled_dot_product_attention(x,x,F.elu(x),attn_mask=mask)
            x_g.append(x)
        x = torch.cat(x_g, dim=-1) # B, N, C

        x = self.gn((channel_shuffle(x, self.groups) + xin).transpose(1,2)).transpose(1,2) # B, N, C

        return x

#Tansformer block
class TransformerBlock(nn.Module):
  """ A more-or-less standard transformer block. """
  def __init__(self, d_model, n_heads, dropout=0.1):
    super().__init__()
    self.heads = n_heads
    self.sa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.ff = nn.Sequential(
        nn.Linear(d_model, d_model*2),
        nn.GELU(),
        nn.Linear(d_model*2, d_model)
    )

  def forward(self, x, mask=None):
    
    mask = torch.logical_not(mask).repeat_interleave(self.heads, dim=0) if mask is not None else None
    
    x = self.ln1(x)
    x = self.sa(x, x, x, attn_mask=mask)[0] + x
    x = self.ff(self.ln2(x)) + x
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
class Classf_head(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.lin1 = nn.Linear(in_channels, in_channels//2)
        self.lin2 = nn.Linear(in_channels//2, n_classes)
        self.bn1 = nn.BatchNorm1d(in_channels//2)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        
        x = F.gelu(self.bn1(self.lin1(x)))
        x = self.lin2(self.dp(x))
        return x

#Transformer model
class AdaPT(nn.Module):
    def __init__(self, embed_dim, n_points, n_blocks = 3, drop_loc=[], drop_target=[], groups=1, sampling_met= None, entmax_alpha=None) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.n_points = n_points
        self.n_blocks = n_blocks
        self.drop_loc = drop_loc
        self.drop_target = drop_target
        self.groups = groups    
        self.sampling_met = sampling_met
        self.entmax_alpha = entmax_alpha

        self.arpe = ARPE(in_channels=3, out_channels=self.embed_dim, npoints=self.n_points)
        self.blocks = nn.ModuleList([GSA(channels=self.embed_dim, groups=self.groups) for _ in range(self.n_blocks)])
        #self.blocks = nn.ModuleList([TransformerBlock(d_model=self.embed_dim, n_heads=self.groups) for _ in range(self.n_blocks)])
        self.predictors = nn.ModuleList([DropPredictor(self.embed_dim) for _ in range(len(self.drop_loc))])
        

    def forward(self, x, drop_temp=1.0):

        B, N, C = x.shape
        x = self.arpe(x)
        prev_decision = torch.ones(B, N, 1, dtype=x.dtype, device=x.device)
        mask = None
        p = 0
        decisions = []
        for i in range(self.n_blocks):
            if i in self.drop_loc:
                pred_score = self.predictors[p](x, prev_decision)
                # Slow warmup
                keepall = torch.cat((torch.zeros_like(pred_score[:,:,0:1]), torch.ones_like(pred_score[:,:,1:2])),2) 
                pred_score = pred_score*drop_temp + keepall*(1-drop_temp)
                if self.training:
                    pred_score = torch.log(pred_score + 1e-8)
                    if self.sampling_met == 'gumbel':
                        decision = F.gumbel_softmax(pred_score, tau=1.0, hard=True, dim=-1)[:,:,1:2]*prev_decision
                    elif self.sampling_met == 'sparsemax':
                        decision = sparsemax(pred_score, dim=1)[:,:,1:2]*prev_decision
                    elif self.sampling_met == 'entmax15':
                        decision = entmax15(pred_score, dim=1)[:,:,1:2]*prev_decision
                    elif self.sampling_met == 'entmax_bisect' and self.entmax_alpha is not None:
                        decision = entmax_bisect(pred_score, alpha=self.entmax_alpha, dim=1)[:,:,1:2]*prev_decision
                    elif self.sampling_met == None:
                        decision = prev_decision
                    else:
                        print('Sampling method not implemented or alpha not provided for entmax_bisect')
                        raise NotImplementedError
                    decision[decision>0] = 1
                    prev_decision = decision
                    mask = (decision*decision.transpose(1,2) + torch.eye(N,device=self.device).repeat(B,1,1)).bool()
                    decisions.append(decision)
                else:
                    score = pred_score[:,:,1]
                    num_keep_tokens = int((1-self.drop_target[p]) * (N))
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_tokens]
                    x = batch_index_select(x, keep_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    mask = None
                    decisions = None
                p += 1

            x = self.blocks[i](x, mask=mask)
            
        return x, decisions
    
#Classifier model
class Adapt_classf(nn.Module):
    def __init__(self,embed_dim, n_points, n_classes, n_blocks = 3, drop_loc=[], drop_target=[], groups=1, sampling_met=None, entmax_alpha=None):
        super().__init__() 

        self.classifier = Classf_head(embed_dim, n_classes)
        self.adapt = AdaPT(embed_dim, n_points, n_blocks, drop_loc, drop_target, groups, sampling_met, entmax_alpha)

    def forward(self, x, drop_temp=1.0):
        x, decisions = self.adapt(x, drop_temp)
        if decisions is not None and len(decisions) > 0:
            #print(decisions[-1])
            x = torch.sum(x * decisions[-1], dim=1) / (torch.sum(decisions[-1], dim=1) + 1e-20)
        else:
            x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x, decisions

#Pytorch_lightning wrapper
class Adapt_classf_pl(pl.LightningModule):
    def __init__(self, cfg, embed_dim, n_points, n_classes, n_blocks = 3, groups=1):
        super().__init__() 

        self.cfg = cfg
        self.save_hyperparameters()
        self.drop_target = cfg.model.drop_rate
        self.start = cfg.train.warmup_start
        self.end = cfg.train.warmup_end
        self.model = Adapt_classf(embed_dim, n_points, n_classes, n_blocks, cfg.model.drop_loc, self.drop_target, groups=groups, sampling_met=cfg.model.sampling_met, entmax_alpha=cfg.model.entmax_alpha)
        self.lr = cfg.train.lr
        self.weight_decay = cfg.train.weight_decay
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, drop_temp=1.0):
        y, decisions = self.model(x, drop_temp)
        if decisions is not None:
            return y, decisions
        else:
            return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        tau = calc_tau(self.start, self.end, self.current_epoch)
        y_hat, decisions = self.model(x, tau)
        loss = self.loss(y_hat, y)
        for i in range(len(self.drop_target)):
            loss += self.cfg.train.alpha*(self.drop_target[i] -(1- torch.mean(decisions[i], dim=1))).pow(2).mean()/len(self.drop_target)
            self.log(f'drop_rate_{i}', 1-(torch.mean(decisions[i])), on_epoch=True, on_step=False)
        pred = torch.argmax(y_hat, dim=1)
        acc = torch.sum(pred == y).float() / float(y.shape[0])
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True,on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        tau = calc_tau(self.start, self.end, self.current_epoch)
        y_hat, decisions = self.model(x, tau)
        loss = self.loss(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        acc = torch.sum(pred == y).float() / float(y.shape[0])
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True,on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.train.epochs, eta_min=0.0)
        return [optimizer], [scheduler]

    def predict(self, x, drop_temp=1.0):
        return self.model(x, drop_temp)





