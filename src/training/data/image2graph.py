from itertools import combinations
import pdb
import networkx as nx
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
# import torch_scatter as ts
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D
from torch_geometric.data import Batch
# from torch_sparse import SparseTensor
from src.training.data.utils import make_graph_util
from torch_geometric.utils import scatter
from scipy import stats
# plt.imsave('tmp.png',xx)
# imwb=mark_boundaries(masked_img, seg, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
# mwb=mark_boundaries(1-mask[0], seg, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
# plt.imsave('tmp.png',np.hstack((imwb,spix_info[seg])))
# src = torch.from_numpy(1-mask[0]).view(-1,1)
# spix_in_mask=ts.scatter_mean(src,index.long(),dim=0)
# spix_mask=spix_in_mask[seg][:,:,0]
# plt.imsave('tmp.png', np.hstack((mwb,np.dstack((spix_mask,spix_mask,spix_mask)))))
# fig=plt.figure()
# ax = fig.add_subplot(111)
# imwb=mark_boundaries(masked_img, seg, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
# eds=np.asarray(G.edges)
# plt.close()
# plt.imshow(imwb)
# plt.scatter(centers[eds][:,:,1],centers[eds][:,:,0], c='y')
# plt.axis('off')
# for p1,p2 in centers[eds]:
#     plt.plot([p1[1],p2[1]],[p1[0],p2[0]])
# plt.savefig('tmp.png')
class Image2Graph(object):
    def __init__(self, generate, segments, segments_mask=None):
        self.segments= segments
        self.segments_mask=segments_mask
    
    def make_graph_delaunay(self, segments, pos):
        # centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in np.unique(segments)])
        # Compute Delaunay triangulation of the indices
        tri = Delaunay(pos)

        # Convert Delaunay triangulation to graph
        graph = nx.Graph()
        graph.add_nodes_from(np.unique(segments))
        for simplex in tri.simplices:
            graph.add_edges_from(combinations(simplex, 2))
        
        return graph, pos

    def make_graph(self, segments):
        segments_ids = np.unique(segments)
        centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
        tri = Delaunay(centers).simplices
        G = nx.Graph()
        G.add_nodes_from(segments_ids)
        eds01= tri[:,:-1]
        eds12= tri[:,1:]
        eds02= tri[:,0::2]
        G.add_edges_from(eds01.tolist())
        G.add_edges_from(eds12.tolist())
        G.add_edges_from(eds02.tolist())

        # pdb.set_trace()
        # for path in tri: 
        #     nx.add_path(G,path)
        return G, centers
    
    def get_data(self, img,mask):
        # mask : 1 means masked
        masked_img= (img*(1-mask)).transpose(1,2,0)
        img=img.transpose(1,2,0)
        seg_dict={}
        pyramid_data={}
        for idx,nseg in enumerate(self.segments):
            if self.segments_mask is None or mask.max()==0:
                seg = slic(masked_img, n_segments=nseg, start_label=0,compactness=10.0, max_num_iter=10, sigma=1, spacing=None, convert2lab=None, enforce_connectivity=True, mask=None)
                seg = torch.from_numpy(seg)
            else:
                # masked_img or img ?
                segkn = slic(masked_img, n_segments=nseg, compactness=10.0, max_num_iter=10, sigma=1, spacing=None, convert2lab=None, enforce_connectivity=True, mask=1-mask[0])
                start_label=segkn.max()+1
                nsegm= self.segments_mask[idx]
                segun = slic(masked_img, n_segments=nsegm, compactness=10.0, max_num_iter=10, sigma=1, spacing=None, convert2lab=None, enforce_connectivity=True, mask=mask[0])
                segun = segun+start_label
                segun[segun==start_label]=0
                seg = segun+segkn

                if seg.min()==0:
                    # print('Correcting Seg...')
                    pixels=np.where(seg==0)
                    xs = pixels[0].tolist()
                    ys = pixels[1].tolist()
                    for x,y in zip(xs,ys):
                        window=1
                        val=0
                        while val==0:
                            slicex_low = max(0,x-window)
                            slicex_high = min(seg.shape[0],x+window+1)
                            slicey_low = max(0,y-window)
                            slicey_high = min(seg.shape[0],y+window+1)
                            xx=seg[slicex_low:slicex_high, slicey_low:slicey_high]
                            val = np.argmax(np.bincount(xx.flatten()))
                            window= window+1
                        seg[x,y]=val
                
                seg=seg-1
                seg = torch.from_numpy(seg)

            # mim= torch.from_numpy(masked_img)
            im= torch.from_numpy(img)
            H, W, C = im.size()
            spix_info = scatter(im.view(H * W, C), seg.view(H * W), dim=0, reduce='mean')

            pos_y = torch.arange(H, dtype=torch.float)
            pos_y = pos_y.view(-1, 1).repeat(1, W).view(H * W)
            pos_x = torch.arange(W, dtype=torch.float)
            pos_x = pos_x.view(1, -1).repeat(H, 1).view(H * W)
            pos = torch.stack([pos_x, pos_y], dim=-1)
            pos = scatter(pos, seg.view(H * W), dim=0, reduce='mean')
            # src= torch.from_numpy(img).view(-1,3)
            # index = seg.view(-1,1)
            # spix_info=ts.scatter_mean(src,index.long(),dim=0)
            gtfs = spix_info.clone()
            
            # src= torch.from_numpy(masked_img).view(-1,3)
            # spix_info_masked=ts.scatter_mean(src,index.long(),dim=0)
            
            # src = torch.from_numpy(mask[0]).view(-1,1)
            # spix_in_mask=ts.scatter_mean(src,index.long(),dim=0)
            mm=torch.from_numpy(mask)
            spix_in_mask=scatter(mm.view(H * W, 1), seg.view(H * W), dim=0, reduce='mean')
            spix_in_mask[spix_in_mask!=0]=1
            spix_in_mask=spix_in_mask.long()

            spix_info[(spix_in_mask==1).squeeze()] = 0
            infs = spix_info
            
            # G,centers=make_graph_util(seg.numpy())
            G,centers=self.make_graph_delaunay(seg.numpy(), pos.numpy())
            # G,centers=self.make_graph(seg)
            adj = nx.to_scipy_sparse_matrix(G, nodelist=list(G.nodes))
            row, col = adj.nonzero()
            eds=np.array([row, col])
            edge_index = torch.tensor(eds, dtype=torch.long)
            seg_data =pyg.data.Data(x=infs, edge_index=edge_index,
                                      gtspix=gtfs, spix_mask=spix_in_mask,
                                      )
            seg_dict[nseg]= seg.long()
            pyramid_data[nseg] = seg_data
        # self.verify_data(pyramid_data,seg_dict, img, masked_img,mask)
        return pyramid_data, seg_dict

    def verify_data(self,data,seg_dict,img,masked_img,maskimo):
        for key in seg_dict.keys():
            data_seg = data[key]
            masked = data_seg.x
            gt = data_seg.gtspix
            mask = data_seg.spix_mask
            eds= data_seg.edge_index
            seg = seg_dict[key]
            seg = seg_dict[key]
            inpim= masked[seg]
            gtim= gt[seg]
            maskim = mask[seg]
            segments=seg.cpu().numpy()
            segments_ids= np.unique(segments)
            H, W= seg.size()
            pos_y = torch.arange(H, dtype=torch.float)
            pos_y = pos_y.view(-1, 1).repeat(1, W).view(H * W)
            pos_x = torch.arange(W, dtype=torch.float)
            pos_x = pos_x.view(1, -1).repeat(H, 1).view(H * W)
            pos = torch.stack([pos_x, pos_y], dim=-1)
            centers = scatter(pos, seg.view(H * W), dim=0, reduce='mean')
            # imwb=mark_boundaries(inpim.numpy(), segments, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
            imwb=mark_boundaries(inpim.numpy(), maskim.numpy()[:,:,0], color=(1, 0, 0), outline_color=None, mode='subpixel', background_label=0)
            # imwb=inpim.numpy()
            plt.imshow(imwb)
            plt.axis('off')
            plt.savefig('./vis_dump/pyramid/inpim_{}.png'.format(key),bbox_inches = 'tight',pad_inches = 0)
            # plt.scatter(centers[eds][:,:,0],centers[eds][:,:,1], c='y')
            # for idx,(p1,p2) in enumerate(centers): plt.text(p1,p2, str(idx), c='red')
            # for p1,p2 in centers[eds.T]: plt.plot([p1[0],p2[0]],[p1[1],p2[1]])
            plt.show(block=False)
            # edsp=SparseTensor(row=eds[0], col=eds[1], sparse_sizes=(masked.shape[0], masked.shape[0])).t()
            plt.figure()
            imwb=mark_boundaries(gtim.numpy(), maskim.numpy()[:,:,0], color=(1, 0, 0), outline_color=None, mode='outer', background_label=0)
            # imwb=mark_boundaries(gtim.numpy(), segments, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
            # imwb=gtim.numpy()
            plt.imshow(imwb)
            # plt.scatter(centers[eds][:,:,0],centers[eds][:,:,1], c='y')
            # for idx,(p1,p2) in enumerate(centers): plt.text(p1,p2, str(idx), c='red')
            # for p1,p2 in centers[eds.T]: plt.plot([p1[0],p2[0]],[p1[1],p2[1]])
            plt.axis('off')
            plt.savefig('./vis_dump/pyramid/gtim_{}.png'.format(key),bbox_inches = 'tight',pad_inches = 0)
            plt.show(block=False)
            if key==100:
                imwb=mark_boundaries(inpim.numpy(), segments*maskim.numpy()[:,:,0], color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
                imwb=mark_boundaries(imwb, segments*(1-maskim.numpy()[:,:,0]), color=(1, 0, 0), outline_color=None, mode='outer', background_label=0)
                plt.figure()
                plt.imshow(imwb)
                plt.axis('off')
                plt.savefig('./vis_dump/pyramid/vis_spix{}.png'.format(key),bbox_inches = 'tight',pad_inches = 0)
                plt.show(block=False)
                print('Here..')

            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.savefig('./vis_dump/pyramid/original.png',bbox_inches = 'tight',pad_inches = 0)
            plt.show(block=False)
            
        pdb.set_trace()
        plt.close('all')

    def verify_data_bkp(self,data,seg_dict):
        for key in seg_dict.keys():
            data_seg = data[key]
            masked = data_seg.x[(data_seg.flagim==key).squeeze()]
            gt = data_seg.gtspix[(data_seg.flagim==key).squeeze()]
            mask = data_seg.spix_mask[(data_seg.flagim==key).squeeze()]
            eds= data_seg.edge_index
            seg = seg_dict[key]
            inpim= masked[seg]
            gtim= gt[seg]
            maskim = mask[seg]
            segments=seg.cpu().numpy()
            segments_ids= np.unique(segments)
            centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
            imwb=mark_boundaries(inpim.numpy(), segments, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
            plt.imshow(imwb)
            plt.scatter(centers[eds][:,:,1],centers[eds][:,:,0], c='y')
            for idx,(p1,p2) in enumerate(centers): plt.text(p2,p1, str(idx), c='red')
            for p1,p2 in centers[eds.T]: plt.plot([p1[1],p2[1]],[p1[0],p2[0]])
            plt.axis('off')
            plt.show(block=False)
            # edsp=SparseTensor(row=eds[0], col=eds[1], sparse_sizes=(masked.shape[0], masked.shape[0])).t()
            plt.figure()
            imwb=mark_boundaries(gtim.numpy(), segments, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)
            plt.imshow(imwb)
            plt.scatter(centers[eds][:,:,1],centers[eds][:,:,0], c='y')
            for idx,(p1,p2) in enumerate(centers): plt.text(p2,p1, str(idx), c='red')
            for p1,p2 in centers[eds.T]: plt.plot([p1[1],p2[1]],[p1[0],p2[0]])
            plt.axis('off')
            plt.show(block=False)

        pdb.set_trace()
        pass
    