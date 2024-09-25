import torch
import torch.nn as nn
import math 

class MagnitudePruner():

    def __init__(self, args, name, layer):
        self.args = args
        self.name = name
        self.layer = layer
        
    def prune(self, sparse_pattern='nmprune', row_b=-1, col_b=128, sparsity=0.5, prunen=2, prunem=4):
        layer = self.layer
        W = layer.weight.data
        W_metric = torch.abs(W)
        if sparse_pattern=='nmprune':
            W_mask = (torch.zeros_like(W)==1)
            for ii in range(W_metric.shape[1]):
                if ii % prunem == 0:
                    tmp = W_metric[:,ii:(ii+prunem)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
        else:
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
            W_mask = (W_metric<=thresh)
        
        if self.args.update_weight:
            W[W_mask] = 0
        layer.mask = ~W_mask
        return layer

class HessianPruner():
    def __init__(self, args, name, layer):
        self.args = args
        self.name = name
        self.layer = layer

        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def prune(self, sparse_pattern='nmprune', row_b=-1, col_b=128, sparsity=0.5, prunen=2, prunem=4, percdamp=0.01):
        W = self.layer.weight.data.clone()
        W = W.float()
        M = torch.ones_like(W)
        #if torch.distributed.get_rank() == 0:
        #    print(self.name, "Before:", self.layer.weight.data[0][:12])
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        blocksize = col_b
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            M1 = M[:, i1:i2].clone()

            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                m = M1[:, i]

                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0
                m[mask1[:, i]] = 0

                Q1[:, i] = q
                M1[:, i] = m
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                if self.args.update_weight:
                    err1 = (w - q) / d 
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            if self.args.update_weight:
                W[:, i1:i2] = Q1
            M[:, i1:i2] = M1
            Losses += torch.sum(Losses1, 1) / 2
            if self.args.update_weight:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
        torch.cuda.synchronize()
    
        if self.args.update_weight:
            self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        #if torch.distributed.get_rank() == 0:
        #    print(self.name, "After:", self.layer.weight.data[0][:12])
        self.layer.mask = M.to(dtype=self.layer.weight.dtype)
        return self.layer
    
    def free(self):
        torch.cuda.empty_cache()

    