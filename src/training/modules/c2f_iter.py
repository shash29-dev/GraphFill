import torch
import torch.nn as nn
import pdb
import torch_geometric.nn as gnn
from torch_geometric.nn import Sequential, GCNConv
from src.utils import DotDict
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
from src.training.modules.ffc import FFCResNetGenerator
from omegaconf import OmegaConf
from src.training.modules.pix2pixhd import MultiDilatedGlobalGenerator
from skimage.segmentation import slic, mark_boundaries
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import scatter

# from torchvision.utils import make_grid

def get_coarse_block(kind):
    if kind.startswith('gcn'):
        return GCNBlock
    
    raise ValueError(f'Unknown coarse model kind {kind}')


class GCNBlock(nn.Module):
    def __init__(self,indim, outdim, gated=True, activation=nn.ReLU(), improved=False, cached=False, add_self_loops=True, bias=False,aggr='mean'):
        super(GCNBlock, self).__init__()
        if gated:
            self.block = Sequential('x, edge_index', [
                                (GCNConv(indim, outdim, improved=improved, add_self_loops=add_self_loops,bias=bias,aggr=aggr, cached=cached), 'x, edge_index -> x1'),
                                activation,
                                (GCNConv(indim, outdim, improved=improved, add_self_loops=add_self_loops,bias=bias,aggr=aggr, cached=cached), 'x, edge_index -> x2'),
                                nn.Sigmoid(),
                                (lambda x1, x2: x1*x2, 'x1, x2 -> xs'),
                            ])
        else:
            self.block = Sequential('x, edge_index', [
                                (GCNConv(indim, outdim, improved=improved, add_self_loops=add_self_loops,bias=bias,aggr=aggr, cached=cached), 'x, edge_index -> x'),
                                activation,
                            ])
        
    
    def forward(self,x,edge_index):
        out= self.block(x,edge_index)
        if x.shape[-1]==out.shape[-1]:
            out+out+x

        return out
        

def init_weights(m):
    if isinstance(m, gnn.Linear):
        m.weight.data.fill_(0.15)

class CoarseNetIter(nn.Module):
    def __init__(self,kind, cfg):
        super(CoarseNetIter, self).__init__()
        cfg = DotDict(lambda : None, cfg)
        block= get_coarse_block(kind)
        channels=32
        chl=[channels* 2**x for x in range(cfg.depth//2+1)]
        chl.extend(chl[::-1][1:])
        chl.insert(0,cfg.indim)
        

        self.layers=nn.ModuleList()
        improved=True
        add_self_loops=True
        bias=False
        aggr='mean'
        cached=False
        for inc,outc in zip(chl[:-1],chl[1:]):
            self.layers.append(block(inc,outc, gated=cfg.gated, activation=nn.ELU(), improved=improved, add_self_loops=add_self_loops,bias=bias,aggr=aggr, cached=cached))
        self.layers.append(block(chl[-1],3, gated=False, activation=nn.Sigmoid()))
        # self.apply(init_weights)

    def get_refined_data(self,currdata, prevout=None, prevdata=None, currseg=None, prevseg=None):
        if prevout==None:
            pass
        else:
            pbi=prevdata.batch
            cbi=currdata.batch
            for bf in range(prevdata.batch.max()+1):
                curr_spix= currdata.x[cbi==bf]
                coarse_im= prevout[pbi==bf][prevseg[bf]]
                H, W, C = coarse_im.size()
                spix_info = scatter(coarse_im.view(H * W, C), currseg[bf].view(H * W), dim=0, reduce='mean')
                masked_pix=(currdata.spix_mask[cbi==bf]==1).squeeze()
                curr_spix[masked_pix] = spix_info[masked_pix]
                currdata.x[cbi==bf]=curr_spix
        return currdata

    def forward(self,batch, concat_mask):
        spin = batch['spix_info']
        spixout={}
        prevout=None
        prevdata=None
        prevseg=None
        for nseg in spin.keys():
            data= spin[nseg]
            if concat_mask:
                x=torch.cat((data.x,data.spix_mask),axis=-1)
            else:
                x= data.x
            eds= data.edge_index
            # eds=SparseTensor(row=eds[0],col=eds[1],sparse_sizes=(x.shape[0],x.shape[0])).t()

            data = self.get_refined_data(currdata=data, prevout=prevout, prevdata=prevdata, currseg=batch['seg'][nseg], prevseg=prevseg)
            out=x
            allout=[]
            # allout.append(out)
            for layer in self.layers:
                out=layer(out,eds)
                for prevblockout in allout:
                    if out.shape==prevblockout.shape:
                        # print('Added Shapes: {},{} for key :{}'.format(out.shape[1],prevblockout.shape[1],nseg))
                        out=out+prevblockout
                allout.append(out)
                # print('Max: {:.4f}, Min: {:.4f} Size: {}'.format(out.max().item(),out.min().item(),out.shape))
            spixout[nseg] = out
            prevout=out
            prevdata=data
            prevseg= batch['seg'][nseg]
            # spixout[nseg] = out
            # print(spixout[nseg].min().item(), spixout[nseg].max().item(),spixout[nseg].mean().item())

        coarse_in,coarse_gt,coarse_pred,coarse_mask= self.post_process_coarse_dict(spixout,batch)
        # coarse_avg=torch.stack(list(coarse_pred.values())).mean(axis=0)
        coarse_avg = coarse_pred[max(list(coarse_pred.keys()))]
        coarse_outdict={
                'coarse_in' : coarse_in, 
                'coarse_gt' : coarse_gt,
                'coarse_pred' : coarse_pred, 
                'coarse_mask': coarse_mask,
                'spixout' : spixout,
                'coarse_avg': coarse_avg,
        }
        return coarse_outdict

    def post_process_coarse_dict(self,spixout,batch, show=False):
        spin= batch['spix_info']
        seg=batch['seg']
        coarse_in= {}
        coarse_gt= {}
        coarse_pred={}
        coarse_mask={}
        for nseg in seg.keys():
            coarse_pred[nseg]=[]
            coarse_gt[nseg]=[]
            coarse_in[nseg]=[]
            coarse_mask[nseg]=[]
        for key in spixout.keys():
            spix_seg = spin[key].x
            spix_seg_pred=spixout[key]
            spix_seg_gt= spin[key].gtspix
            spix_mask = spin[key].spix_mask
            batch_idx= spin[key].batch
            for bf in range(spin[key].batch.max()+1):
                coarse_im_in= spix_seg[batch_idx==bf][seg[key][bf]]
                coarse_im_gt= spix_seg_gt[batch_idx==bf][seg[key][bf]]
                coarse_im_pred= spix_seg_pred[batch_idx==bf][seg[key][bf]]
                c_mask=spix_mask[batch_idx==bf][seg[key][bf]]
                coarse_in[key].append(coarse_im_in.contiguous())
                coarse_gt[key].append(coarse_im_gt.contiguous())
                coarse_pred[key].append(coarse_im_pred.contiguous())
                coarse_mask[key].append(c_mask.contiguous())

        for key in coarse_gt.keys():
            coarse_in[key] = torch.stack(coarse_in[key],axis=0).permute(0,3,1,2).contiguous()
            coarse_gt[key] = torch.stack(coarse_gt[key],axis=0).permute(0,3,1,2).contiguous()
            coarse_pred[key] = torch.stack(coarse_pred[key],axis=0).permute(0,3,1,2).contiguous()
            coarse_mask[key] = torch.stack(coarse_mask[key],axis=0).permute(0,3,1,2).contiguous()
        
        if show:
            for key in spixout.keys():
                for bf in range(spin[key].batch.max()+1):
                    masked_img=coarse_pred[key][bf].detach().permute(1,2,0).cpu().numpy()
                    segments= seg[key][bf].cpu().numpy()
                    segments_ids = np.unique(segments)
                    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
                    imwb=mark_boundaries(masked_img, segments, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
                    eds=np.asarray(spin[key].edge_index.cpu().numpy())
                    plt.imshow(imwb)
                    plt.scatter(centers[:,1],centers[:,0], c='y')
                    for idx,(p1,p2) in enumerate(centers): plt.text(p2,p1, str(idx), c='red')
                    plt.axis('off')
                    plt.show()
        
        return coarse_in,coarse_gt,coarse_pred, coarse_mask

