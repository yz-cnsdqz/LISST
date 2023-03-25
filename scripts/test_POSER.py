"""
This script is to train the LISST model. The training has two parts:
    - load the template data and change them into model parameters.
    - Train the body shape PCA space.
"""

import os, sys, glob
import re
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.getcwd())

import time
import numpy as np
import open3d as o3d
import torch
import pdb
import argparse
import pickle


from torch import nn, optim
from torch.nn import functional as F

from lisst.utils.config_creator import ConfigCreator
from lisst.models.body import LISSTPoser, LISST
from lisst.models.baseops import (RotConverter, TestOP, CanonicalCoordinateExtractor)


MAX_DEPTH = 10


class LISSTPoserGenOP(TestOP):

    def build_model(self):
        self.model = LISSTPoser(self.modelconfig)
        self.model.eval()
        self.model.to(self.device)
        self.nj = self.model.nj
        
        ckpt_path = os.path.join(self.testconfig['ckpt_dir'],'epoch-500.ckp')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('[INFO] --load pre-trained pose model: {}'.format(ckpt_path))

        self.body = LISST()
        self.body.eval()
        self.body.to(self.device)
        
        ckpt_path = os.path.join('results/src/LISST_v0/checkpoints', 'epoch-000.ckp')
        self.body.load(ckpt_path)
        print('[INFO] --load pre-trained LISST model: {}'.format(ckpt_path))


    def latent_interpolate(self, n_bodies, n_poses):
        '''
        We first sample n_bodies body shape parameters, and then interpolate in the pose model latent space
        - n_bodies: for each input sequence, how many different sequences to predict
        
        '''
        # load the data
        gen_results = {}
        
        '''generate random body shapes'''
        zs = torch.zeros(n_bodies,self.nj).to(self.device)
        zs[:,:15] = 50*torch.randn(n_bodies, 15)
        bone_length = self.body.decode(zs).unsqueeze(1).repeat(1,n_poses,1).reshape(n_bodies*n_poses, -1)
        
        '''generate random body poses'''
        zp = torch.arange(-3, 3, 6./n_poses).repeat(self.model.z_dim, 1).permute(1,0).to(self.device)
        poses_rotcont = self.model.decode(zp).repeat(n_bodies, 1, 1).reshape(n_bodies*n_poses, -1)
        poses_rotcont_reshape = poses_rotcont.reshape(-1, self.nj, 6).reshape(n_bodies*n_poses*self.nj, 6)
        poses_rotmat = RotConverter.cont2rotmat(poses_rotcont_reshape).reshape(n_bodies*n_poses, self.nj, 3,3)

        x_root = torch.zeros(n_bodies*n_poses, 1,3).to(self.device)
        
        J_locs_fk, _ = self.body.forward_kinematics(x_root, bone_length,poses_rotmat)
        J_locs_fk = J_locs_fk.contiguous().view(n_bodies, n_poses, -1, 3).detach().cpu().numpy()
        gen_results['generated_J_locs'] = J_locs_fk    
            

        ### save to file
        outfilename = os.path.join(
                            self.testconfig['result_dir'],
                            'motion_gen_seed{}'.format(self.testconfig['seed'])
                        )
        if not os.path.exists(outfilename):
            os.makedirs(outfilename)
        outfilename = os.path.join(outfilename,
                        'results_{}.pkl'.format(self.modelconfig['body_repr'])
                        )
        with open(outfilename, 'wb') as f:
            pickle.dump(gen_results, f)
        print('[INFO] --saving results to '+outfilename)



    def sample(self, n_bodies, n_poses):
        '''
        We first sample n_bodies body shape parameters, and then interpolate in the pose model latent space
        - n_bodies: for each input sequence, how many different sequences to predict
        
        '''
        # load the data
        gen_results = {}
        
        '''generate random body shapes'''
        zs = torch.zeros(n_bodies,self.nj).to(self.device)
        zs[:,:15] = 50*torch.randn(n_bodies, 15)
        bone_length = self.body.decode(zs).unsqueeze(1).repeat(1,n_poses,1).reshape(n_bodies*n_poses, -1)
        
        '''generate random body poses'''
        zp = torch.randn(n_bodies*n_poses, self.model.z_dim).to(self.device)
        poses_rotcont = self.model.decode(zp).reshape(n_bodies*n_poses, -1)
        poses_rotcont_reshape = poses_rotcont.reshape(-1, self.nj, 6).reshape(n_bodies*n_poses*self.nj, 6)
        poses_rotmat = RotConverter.cont2rotmat(poses_rotcont_reshape).reshape(n_bodies*n_poses, self.nj, 3,3)

        x_root = torch.zeros(n_bodies*n_poses, 1,3).to(self.device)
        
        J_locs_fk, _ = self.body.forward_kinematics(x_root, bone_length,poses_rotmat)
        J_locs_fk = J_locs_fk.contiguous().view(n_bodies, n_poses, -1, 3).detach().cpu().numpy()
        gen_results['generated_J_locs'] = J_locs_fk    
            

        ### save to file
        outfilename = os.path.join(
                            self.testconfig['result_dir'],
                            'motion_gen_seed{}'.format(self.testconfig['seed'])
                        )
        if not os.path.exists(outfilename):
            os.makedirs(outfilename)
        outfilename = os.path.join(outfilename,
                        'results_{}.pkl'.format(self.modelconfig['body_repr'])
                        )
        with open(outfilename, 'wb') as f:
            pickle.dump(gen_results, f)
        print('[INFO] --saving results to '+outfilename)


from lisst.utils.config_creator import ConfigCreator
from lisst.utils.batch_gen import BatchGeneratorCMUCanonicalized
from lisst.utils.config_env import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_printoptions(sci_mode=False)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    
    cfgall = ConfigCreator(args.cfg)
    modelcfg = cfgall.modelconfig
    losscfg = cfgall.lossconfig
    traincfg = cfgall.trainconfig
    
    testcfg = {}
    testcfg['gpu_index'] = args.gpu_index
    testcfg['ckpt_dir'] = traincfg['save_dir']
    testcfg['result_dir'] = cfgall.cfg_result_dir
    testcfg['seed'] = args.seed
    testcfg['log_dir'] = cfgall.cfg_log_dir
    
    """model and testop"""
    testop = LISSTPoserGenOP(modelcfg,testcfg)
    testop.build_model()
    testop.sample(n_bodies=5, n_poses=30)
    












