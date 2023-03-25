import numpy as np
import os, glob, sys
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from lisst.models.baseops import MLP, VAE





class LISSTCore(nn.Module):
    """LISSTCore: Linear-Shaped Skeleton Template:
      Posed skeleton = LISSTCore(root_transl, root_orient, shape_params, joint_rotations)
        - We employ the CMU skeleton topology. see here for more information: http://graphics.cs.cmu.edu/nsp/course/15-464/Fall05/assignments/StartupCodeDescription.html
        
        - The linear space space is learned by PCA based on the bone length, based on the CMU mocap data.
    
    """
    
    def __init__(self, config=None):
        super(LISSTCore, self).__init__()

        '''about the template'''
        self.joint_names = None
        self.children_table = {}
        self.num_kpts = 31
        self.new_joints = []
        if config.get('add_nose'):
            self.num_kpts += 1
            self.new_joints.append('nose')
        if config.get('add_heels'):
            self.num_kpts += 2
            self.new_joints.append('lheel')
            self.new_joints.append('rheel')


        self.template_joint_directions = torch.nn.Parameter(torch.zeros(self.num_kpts, 3),
                                        requires_grad=False)
        
        '''about the shape space'''
        self.meanshape = torch.nn.Parameter(torch.zeros(1, self.num_kpts),
                                    requires_grad=False)
        self.shape_basis = torch.nn.Parameter(torch.zeros(self.num_kpts, self.num_kpts),
                                    requires_grad=False)
        self.eigvals = torch.nn.Parameter(torch.zeros(self.num_kpts),
                                    requires_grad=False)
        self.shape_basis2 = torch.nn.Parameter(torch.zeros(self.num_kpts, self.num_kpts),
                                    requires_grad=False)
        
        
    def load(self, model_path, verbose=True,
             build_parent_table=False):
        ckpt = torch.load(model_path)
        self.joint_names = ckpt['joint_names']
        self.children_table = ckpt['children_table']
        self.load_state_dict(ckpt['model_state_dict'])
        if verbose:
            print('-- successfully loaded: {}'.format(model_path))
    
        
       
    def get_new_joints(self)->list:
        """return the list of additional joint names beyond the CMU mocap skeleton
        Returns:
            list: joint name list, e.g. ['nose', 'lheel', 'rheel']
        """
        return self.new_joints



    def decode1(self, 
                z:torch.Tensor, 
                use_eigenvals: bool=True,
            )->torch.Tensor:
        
        """Given a latent variable, recover the bone lengths (or body shape).
        Args:
            - z (torch.Tensor): the latent variable [b,d]. Its maximal dimension should not exceed the number of joints.
            - use_eigenvals (bool, optional): if z~N(0,I), we use eigenvals to compensate. If z is encoded bone length, set to False. Defaults to True.
        Returns:
            torch.Tensor: the bone length [b,J]
        """
        
        zdim = z.shape[-1]
        if use_eigenvals:
            eigvals = self.eigvals[:zdim].unsqueeze(0)
            z = z * eigvals.repeat(z.shape[0], 1)
        X = self.meanshape + torch.einsum('ij,bj->bi',self.shape_basis[:,:zdim], z)

        return X
        

    def decode(self, 
                z:torch.Tensor, 
            )->torch.Tensor:
        
        """Given a latent variable, recover the bone lengths (or body shape).
        Args:
            - z (torch.Tensor): the latent variable [b,d]. Its maximal dimension should not exceed the number of joints.
            - use_eigenvals (bool, optional): if z~N(0,I), we use eigenvals to compensate. If z is encoded bone length, set to False. Defaults to True.
        Returns:
            torch.Tensor: the bone length [b,J]
        """
        
        zdim = z.shape[-1]
        X = self.meanshape + torch.einsum('ij,bj->bi',self.shape_basis2[:,:zdim], z)
        return X



    def encode(self, 
                X: torch.Tensor, 
                n_pca_comps: Union[int, None] = None
        )-> torch.Tensor:
        """given a bone length, encode it with the learned PCA space.
        Args:
            X (torch.Tensor): [b,J]
            n_pca_comps (Union[int, None], optional): keep the first n_pca_comps and discard rests. Defaults to None, meaning keeping all.
        Returns:
            torch.Tensor: latent variable z, [b, d]
        """
        
        X_ = X-self.meanshape
        z = torch.einsum('ij,bj->bi', self.shape_basis.T, X_)
        if n_pca_comps is not None:
            z = z[:,:n_pca_comps]
        return z


    def random_skeleton(self,
                        pc: int=0, 
                        n_samples: int=1,
                        device: torch.device=torch.device('cpu')):
        """randomly sample PCA coefficients
        Args:
            pc (int): sample on which principle component? . Defaults to 15.
            n_samples (int): num_samples to draw. Defaults to 1.
            device (torch.device): device type
        Returns:
            jts: joint locations, [n_samaples, J, 3]
        """
        ## random bone length
        z = torch.zeros(n_samples,self.num_kpts)
        z[:,pc] = 100*torch.randn(n_samples)
        z = z.to(device)
        
        bone_length = self.decode(z)
        
        ## T pose body
        transl = torch.zeros(n_samples,1,3).to(device)
        rotmat = torch.eye(3)[None,None,...].repeat(n_samples,self.num_kpts, 1,1).to(device)
        jts = self.forward_kinematics(transl, bone_length,rotmat) #[N,J,3]

        return jts

        
    def forward_kinematics(self, 
                        x_root: torch.Tensor,
                        jl: torch.Tensor,
                        j_rotmat: torch.Tensor,
        )->torch.Tensor:
        """forward kinematics for the LISST body model.
        Note that 
        - the joint lengths and the bone lengths are equivalent. In jl, the first entry is always 0.
        - all rotation matrices are NOT in the local joint coordinate, but in the canonical body space.
        - When the body is globally transformed, `x_root` and ALL `j_rotmat` should be transformed.
        
        Args:
            x_root (torch.Tensor): [b,1,3]
            jl (torch.Tensor): the input joint lengths, [b,J], incl. root
            j_rotmat (torch.Tensor): the input joint rotation matrix, [b,J,3,3], incl. root
        Returns:
            torch.Tensor: all joint locations, [b,J,3], incl. the root
        """
        
        nb = x_root.shape[0]
        x_all = torch.zeros(nb, self.num_kpts, 3).to(x_root.device) ## joint locs placeholder
        x_all[:,0] = x_root[:,0]

        for parent, children in self.children_table.items():            
            idx_p = self.joint_names.index(parent)
            for child in children:
                idx_c = self.joint_names.index(child)
                rotmat = j_rotmat[:,idx_c]
                x_all[:,idx_c] = x_all[:,idx_p] + jl[:,idx_c:idx_c+1] * torch.einsum('bij,j->bi',
                                    rotmat, self.template_joint_directions[idx_c])        
        return x_all


    @staticmethod
    def get_num_kpts():
        return LISSTCore().num_kpts



class PoserEncoder(nn.Module):
    def __init__(self, h_dim, num_blocks, model_type, nj, adj):
        super(PoserEncoder, self).__init__()
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.model_type = model_type
        self.nj = nj
        if model_type == 'MLP':
            self.encoder = nn.ModuleList([MLP(h_dim*nj, [h_dim*nj, h_dim*nj], activation='lrelu')
                                        for _ in range(self.num_blocks)])
        else:
            raise NotImplementedError('The model type is not supported.')

        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """pose encoder
        Args:
            x (torch.Tensor): [b,J,d], [batch, joints, dimension]
        Returns:
            torch.Tensor: n/a
        """
        nb = x.shape[0]
        nj = self.nj
        if self.model_type == 'MLP':
            hx = x.contiguous().view(nb, -1)
            for layer in self.encoder:
                hx = layer(hx)
            hx = hx.contiguous().view(nb, nj, -1)
        
        return hx





class PoserDecoder(nn.Module):
    def __init__(self, h_dim, num_blocks, model_type, nj, adj=None):
        super(PoserDecoder, self).__init__()
        self.h_dim = h_dim
        self.num_blocks = num_blocks
        self.model_type = model_type
        self.nj = nj
        if model_type == 'MLP':
            self.decoder = nn.ModuleList([MLP(h_dim*nj, [h_dim*nj, h_dim*nj], activation='lrelu')
                                        for _ in range(self.num_blocks)])
        else:
            raise NotImplementedError

        
    def forward(self, z:torch.Tensor)->torch.Tensor:
        """pose decoder
        Args:
            z (torch.Tensor): [b,J,d], [batch, joints, dimension]
        Returns:
            torch.Tensor: n/a
        """
        nb = z.shape[0]
        nj = self.nj
        if self.model_type == 'MLP':
            hy = z.contiguous().view(nb, -1)
            for layer in self.decoder:
                hy = layer(hy)
            hy = hy.contiguous().view(nb, nj, -1)
        elif self.model_type == 'TRANSFORMER':
            hy = self.pe(z.permute(1,0,2)).permute(1,0,2)
            hy = self.decoder(hy)
        
        return hy




class LISSTPoser(VAE):
    """generative model of LISST poses
        - Rotations are represented as 6D.
        
        - Pelvis/root is included.
        
        - All rotations are w.r.t. the canonical coordinate
        
    """
    
    def __init__(self, configs):
        super(LISSTPoser, self).__init__()
        self.nj = 31

        in_dim = 6
        self.h_dim = h_dim = configs['h_dim'] #256
        self.z_dim = z_dim = configs['z_dim']
        self.num_blocks = configs.get('num_blocks', 1)
        adj = configs.get('adj', None)
        
        
        '''encoder'''
        self.in_fc = nn.Linear(in_dim, h_dim)
        self.enc = PoserEncoder(h_dim=h_dim, num_blocks=self.num_blocks, 
                                    model_type=configs['mtype'], nj=self.nj,
                                    adj=adj)
        self.e_mu = nn.Linear(h_dim, z_dim)
        self.e_logvar = nn.Linear(h_dim, z_dim)
        

        '''decoder'''
        self.d_z = nn.Linear(z_dim, h_dim)
        self.dec = PoserDecoder(h_dim=h_dim, num_blocks=self.num_blocks, 
                                    model_type=configs['mtype'], nj=self.nj,
                                    adj=adj)    
        
        self.out_fc = nn.Linear(h_dim, in_dim)    


    def load(self, ckpt_path: str, verbose=True):
        """wrapper to load the pre-trained checkpoint.
        Args:
            ckpt_path (str): the absolute path of the checkpoint path
        """
        checkpoint = torch.load(ckpt_path)['model_state_dict']
        self.load_state_dict(checkpoint)
        if verbose:
            print('-- successfully loaded: '+ckpt_path)



    def encode(self, x):
        hx = self.in_fc(x)
        hx = self.enc(hx)
        mu = self.e_mu(hx)
        logvar = self.e_logvar(hx)
        return mu, logvar

    def decode(self, z):
        hz = self.d_z(z)
        hz = self.dec(hz)
        x_rec = self.out_fc(hz)
        return x_rec
            
        

    def forward(self, x):
        '''the forward pass
        Args:
            - x: [b,J,d]. The input batch of data, [batch, joints, dim]
        
        Returns:
            - x_rec: [b,J,d]. The reconstructed version
            - mu, logvar: [b, J, d]. The inference posterior.
        '''
        mu, logvar = self.encode(x)
        z = self._sample(mu, logvar)
        x_rec = self.decode(z)
        
        return x_rec, mu, logvar


    def add_additional_bones(self,
                            J_rotcont: torch.Tensor, joint_names:list,
                            new_joints:list=['nose', 'lheel', 'rheel']):
        """copy existing rotations to add more bones.
        Currently, only nose and heels are supported!
        Args:
            J_rotcont (torch.Tensor): the 6D rotations with 31 joints, [b, nj, 6]
            joint_names (list): the joint names in the employed shaper model.
            new_joints (list): the bones need to add, e.g. ['nose', 'lheel', 'rheel']
            
        Returns:
            J_rotcont: the new joint rotations
        """
        
        for joint in new_joints:        
            if joint == 'nose':
                joint_parent = 'head'
            elif joint == 'lheel':
                joint_parent = 'lfoot'
            elif joint == 'rheel':
                joint_parent = 'rfoot'
            joint_idx = joint_names.index(joint_parent)
            J_rotcont_j = J_rotcont[:,joint_idx:joint_idx+1]
            J_rotcont = torch.cat([J_rotcont, J_rotcont_j],dim=-2)

        return J_rotcont





