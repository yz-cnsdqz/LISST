import torch
import numpy as np
import random
import glob
import os, sys
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import pickle
import cv2
import math
import tqdm
sys.path.append(os.path.join(os.getcwd(), 'lisst'))
sys.path.append(os.getcwd())

from lisst.models.baseops import RotConverter




class BatchGeneratorCMUCanonicalized(object):
    """batch generator for the canonicalized CMU dataset
    
    - The order of joints follow these labels and orders:
        data['joints_names'] = ['root', 'lhipjoint', 'lfemur', 'ltibia', 'lfoot', 
                            'ltoes', 'rhipjoint', 'rfemur', 'rtibia', 'rfoot', 'rtoes', 
                            'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 
                            'head', 'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 
                            'lthumb', 'rclavicle', 'rhumerus', 'rradius', 
                            'rwrist', 'rhand', 'rfingers', 'rthumb']
    - for each npz file, it contains
         data = {
                'global_transl': [], #[t,3]
                'global_rotmat': [], #[t,3,3]
                'joints_locs':[], # #[t,J,3]
                'joints_rotmat': [], #[t,J,3,3] # note that all joint locations are w.r.t. the skeleton template. They need to be transformed globally.
                'joints_length': [], #[t,J+1], incl. the root
            }

    """
    def __init__(self,
                data_path,
                sample_rate=3,
                body_repr='bone_transform', #['joint_location', 'bone_transform', 'bone_transform_localrot' ]
                read_to_ram=True
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.J_locs = []
        self.J_rotmat = []
        # self.J_rotmat_local = []
        self.J_len = []
        self.body_repr = body_repr
        self.read_to_ram = read_to_ram
        self.max_len = 80 if 'x8' in data_path else 10

    def reset(self):
        self.index_rec = 0
        if self.read_to_ram:
            idx_permute = torch.randperm(self.J_locs.shape[1])
            self.J_locs = self.J_locs[:,idx_permute]
            self.J_rotcont = self.J_rotcont[:,idx_permute]
            # self.J_rotcont_local = self.J_rotcont_local[:,idx_permute]
            self.J_len = self.J_len[:,idx_permute]
            

    def has_next_rec(self):
        if self.read_to_ram:
            if self.index_rec < self.J_locs.shape[1]:
                return True
            return False
        else:
            if self.index_rec < len(self.rec_list):
                return True
            return False



      
      
    def get_rec_list(self, 
                    to_gpu=False):
    
        self.rec_list = os.path.join(self.data_path+'.pkl')
        print('[INFO] read all data to RAM from: '+self.data_path)
        all_data = np.load(self.rec_list, allow_pickle=True)
        self.J_locs = all_data['J_locs'] #[b,t,J,3]
        self.J_rotmat = all_data['J_rotmat'] #[b,t, J, 3, 3]
        self.J_len = all_data['J_len'] #[b,t, J]

        if to_gpu:
            self.J_locs = torch.cuda.FloatTensor(self.J_locs).permute(1,0,2,3) #[t,b,J,3]
            self.J_rotmat = torch.cuda.FloatTensor(self.J_rotmat).permute(1,0,2,3,4) #[t,b,J,3,3]
            self.J_len = torch.cuda.FloatTensor(self.J_len).permute(1,0,2) #[t,b,J]
        else:
            raise NotImplementedError('it has to be on gpus.')
        
        ## convert rotation matrix to 6D representations
        nt, nb, nj = self.J_locs.shape[:3]
        self.J_rotcont = self.J_rotmat[:,:,:,:,:-1].contiguous().view(nt,nb,nj,-1)
            


    def next_batch(self, batch_size=64, return_shape=False):
        if self.body_repr == 'joint_location':
            batch_data_ = self.J_locs[:, self.index_rec:self.index_rec+batch_size] #[t,b,J,3]
        elif self.body_repr == 'bone_transform':
            batch_data_locs = self.J_locs[:, self.index_rec:self.index_rec+batch_size]
            batch_data_rotcont = self.J_rotcont[:, self.index_rec:self.index_rec+batch_size]
            batch_data_ = torch.cat([batch_data_locs, batch_data_rotcont],dim=-1)#[t,b,J,3+6]
        
        batch_shape = self.J_len[:, self.index_rec:self.index_rec+batch_size]

        self.index_rec+=batch_size
        nt, nb = batch_data_.shape[:2]
        
        if not return_shape:
            return batch_data_.contiguous().view(nt,nb,-1).detach()
        else:
            return batch_data_.contiguous().view(nt,nb,-1).detach(), batch_shape


    
    def next_sequence(self):
        rec = self.rec_list[self.index_rec]
        with np.load(rec) as data:
            g_transl = np.expand_dims(data['global_transl'], axis=1)[::self.sample_rate]
            g_rotmat = np.expand_dims(data['global_rotmat'], axis=1)[::self.sample_rate]
            j_transl = data['joints_locs'][::self.sample_rate]
            j_rotmat = data['joints_rotmat'][::self.sample_rate]
            transl = np.concatenate([g_transl, j_transl],axis=1)
            rotmat = np.concatenate([g_rotmat, j_rotmat],axis=1)
        
        transl = torch.cuda.FloatTensor(transl) # [t,j,3]
        rotmat = torch.cuda.FloatTensor(rotmat) # [t,j,3,3]
        nt, nj = rotmat.shape[:2]
        rotcont = rotmat[:,:,:,:-1].contiguous().view(nt,nj,-1)

        if self.body_repr == 'joint_location':
            outdata = transl
        elif self.body_repr == 'bone_transform':
            outdata = torch.cat([transl, rotcont],dim=-1)
        
        self.index_rec+=1
        
        return outdata.detach()





if __name__=='__main__':
    pass















