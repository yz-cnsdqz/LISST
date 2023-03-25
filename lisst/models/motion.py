import numpy as np
import torch
from torch import nn

from lisst.models.baseops import (MLP, VAE, PositionalEncoding,
                            RotConverter)

from lisst.models.body import LISSTCore



class MotionPredictorRNN(VAE):
    """modified the marker predictor in GAMMA.
    Conditional variational autoencoder
    """
    def __init__(self, configs):
        super(MotionPredictorRNN, self).__init__()
        self.body_repr = body_repr = configs['body_repr']
        self.h_dim = h_dim = configs['h_dim']
        self.num_frames_primitive = configs.get('num_frames_primitive', 10)
        self.num_kpt = 31
        self.num_blocks = configs.get('num_blocks', 1)
        
        if body_repr == 'joint_location':
            self.in_dim = in_dim = self.num_kpt*3 # location
        elif body_repr in ['bone_transform', 'bone_transform_localrot']: 
            self.in_dim = in_dim = self.num_kpt*9 # location and 6D rotcont
        
        self.h_dim = h_dim = configs['h_dim'] #256
        self.z_dim = z_dim = configs['z_dim']
        self.hdims_mlp = hdims_mlp = configs['hdims_mlp'] #[512, 256]

        # encode
        self.x_enc = nn.GRU(in_dim, h_dim)
        self.e_rnn = nn.GRU(in_dim, h_dim)
        self.e_mlp = MLP(2*h_dim, hdims_mlp, activation='tanh')
        self.e_mu = nn.Linear(self.e_mlp.out_dim, z_dim)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, z_dim)

        # decode
        self.dz = nn.Linear(z_dim, h_dim)
        self.drnn_mlp = MLP(h_dim, hdims_mlp, activation='tanh')
        self.d_rnn = nn.GRUCell(in_dim + h_dim + h_dim, h_dim)
        self.d_mlp = MLP(h_dim, hdims_mlp, activation='tanh')
        self.d_out = nn.Linear(self.d_mlp.out_dim, in_dim)


    def encode(self, x, y):
        t_his = x.shape[0]
        _, hy = self.e_rnn(y)
        if t_his > 0:
            _, hx = self.x_enc(x)
            h = torch.cat((hx[-1], hy[-1]), dim=-1)
        else:
            h = torch.cat((hy[-1], hy[-1]), dim=-1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)


    def decode(self, x, z):
        '''decode the motion latent variable
        
        Args:
            - x: the motion seed. [t,b,d]. When t=0, it changes from prediction to generation.
            - z: the latent variable, [b, d]
        
        Returns:
            - y: the generated motion, [t,b,d]
        '''
        t_his = x.shape[0]
        t_pred = self.num_frames_primitive-t_his
        hz = self.dz(z)
        _, hx = self.x_enc(x)
        hx = hx[0] #[b, d]
        h_rnn = hx
        # h_rnn = self.drnn_mlp(hx)
        y = []
        for i in range(t_pred):
            y_p = x[-1] if i==0 else y[-1]
            rnn_in = torch.cat([hx, hz, y_p], dim=-1)
            h_rnn = self.d_rnn(rnn_in, h_rnn)
            hfc = self.d_mlp(h_rnn)
            y_i = self.d_out(hfc) + y_p
            y.append(y_i)
        y = torch.stack(y)
        
        return y


    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self._sample(mu, logvar)
        y_pred = self.decode(x, z)

        return y_pred, mu, logvar






class LISSTGEMP(nn.Module):
    """A generative motion primitive model based on the CMU skeleton template.

    Note:
        - the model only works in the canonical motion space.
        - The joints include the root, i.e. the pelvis.
        - body is represented by the bone transformation, rotation in 6D cont [t,b,J, 3+6]

    """

    def __init__(self, modelcfg):
        super(LISSTGEMP, self).__init__()
        predictorcfg = modelcfg['predictorcfg']
        self.num_frames_primitive = modelcfg.get('num_frames_primitive', 10)
        self.body_repr = modelcfg.get('body_repr', 'bone_transform')
        predictorcfg['body_repr']=self.body_repr
        predictorcfg['num_frames_primitive']=self.num_frames_primitive
        
        if predictorcfg['type'] == 'RNN':
            self.predictor = MotionPredictorRNN(predictorcfg)
        else:
            raise ValueError('Other models are not implemented yet.')  
        
        ## the body template
        self.body_model = LISSTCore({}) # empty dict
        self.nj = self.body_model.num_kpts
        self.z_dim = self.predictor.z_dim
        self.reproj_factor = modelcfg.get('reproj_factor', 1.0)
        
    def _cont2rotmat(self, rotcont):
        '''local process from continuous rotation to rotation matrix
        
        Args:
            - rotcont: [t,b,J,6]

        Returns:
            - rotmat: [t,b,J,3,3]
        '''
        nt, nb, nj = rotcont.shape[:-1]
        rotcont = rotcont.contiguous().view(nt*nb*nj, -1)
        rotmat = RotConverter.cont2rotmat(rotcont).view(nt, nb, nj,3,3)
        return rotmat


    def _rotmat2cont(self, rotmat):
        '''local process from rotatiom matrix to continuous rotation
        
        Args:
            - rotmat: [t,b,J,3,3]

        Returns:
            - rotcont: [t,b,J,6]
        '''
        nt,nb,nj = rotmat.shape[:3]
        rotcont = rotmat[:,:,:,:,:-1].contiguous().view(nt,nb,nj,-1)
        return rotcont


    def transform_bone_transf(self, 
                                X: torch.Tensor, 
                                R: torch.Tensor, 
                                T: torch.Tensor,
                                to_local: bool=True
        )->torch.Tensor:
        '''transform the bone transform to under a new coordinate
        
        Args:
            - X: the input bone transform sequence, [t,b,J*9]
            - R: the new coordinate rotation w.r.t. world origin, [b,3,3]
            - T: the new coordinate translate w.r.t. world origin, [b,1,3]
            - to_local: If true, put X to the coordinate [R,T]. If False, transform X to world
        
        Returns:
            - Y: the transformed bone transform, [t,b,J*9]
        '''
        X = X.contiguous().view(X.shape[0], X.shape[1], self.nj, -1)
        transl = X[:,:,:,:3]
        rotcont = X[:,:,:,3:]
        
        rotmat = self._cont2rotmat(rotcont)

        if to_local:
            transl_new = torch.einsum('bij,tbpj->tbpi', 
                    R.permute(0,2,1), 
                    transl-T.unsqueeze(0)) #transform to current coord

            rotmat_new = torch.einsum('bij,tbpjk->tbpik', 
                    R.permute(0,2,1), 
                    rotmat) #transform to current coord
        else:
            transl_new = torch.einsum('bij,tbpj->tbpi', 
                    R, transl)+T.unsqueeze(0) #transform to current coord

            rotmat_new = torch.einsum('bij,tbpjk->tbpik', 
                    R, rotmat) #transform to current coord

        rotcont_new = self._rotmat2cont(rotmat_new)
        Y = torch.cat([transl_new, rotcont_new], dim=-1)
        Y = Y.contiguous().view(Y.shape[0], Y.shape[1], -1)

        return Y


    
    def forward_test(self,
                bone_length: torch.Tensor,
                z: torch.Tensor, 
                X: torch.Tensor,
                ):
        """the forward pass for generating the body joints, provided the input arguments.

        Args:
            bone_length (torch.Tensor): body shape parameter, [b, J]
            z (torch.Tensor): the latent variable, [b,z_dim] or [t_pred, b, z_dim]
            X (torch.Tensor): the motion seed, [t,b,J*d]

        Returns:
            Y_rec_b: the average of predicted and projected motion, [t,b, J*d], used for recursion
            Y_rec_fk: the reprojected motion with fixed bone length [t,b, J*d], used for visualization/character animation.
            
        Note:
            only during testing, the body model can include the additional bones/joints.
            Since the body model only learns with CMU Mocap data, we first exclude, then inference, and then add new joints back.
        """
        
        # motion generation
        Y_rec = self.predictor.decode(X, z)
        
        # motion projection
        Y_rec_fk, _ = self.reprojection(Y_rec, bone_length)

        # motion blending
        ## the blended motion for the next motion seed, the reprojected motion via FK is to drive avatars.
        Y_rec_b = self.reproj_factor*Y_rec_fk + (1.-self.reproj_factor)*Y_rec

        return Y_rec_b, Y_rec_fk

    
        
    def add_additional_bones(self,
                            J_rotcont: torch.Tensor, joint_names:list,
                            new_joints:list=['nose', 'lheel', 'rheel']):
        """copy existing rotations to add more bones.
        Currently, only nose and heels are supported!

        Args:
            J_rotcont (torch.Tensor): the 6D rotations with 31 joints, [t, b, nj, 6]
            joint_names (list): the joint names in the body model, with the joints to be added.
            new_joints (list): the bones need to add, e.g. ['nose', 'lheel', 'rheel']
            
        Returns:
            J_rotcont: the new joint rotations
        """
        nt, nb = J_rotcont.shape[:2]
        J_rotcont=J_rotcont.contiguous().view(nt*nb, -1, 6)
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

        return J_rotcont.contiguous().view(nt, nb, -1, 6)
        
        


    def reprojection(self, Y_rec, bone_length):
        """reprojection via forward kinematics
        The predicted joint locations are discarded and others are preserved for FK.

        Args:
            Y_rec, [t,b,d]. The predicted bone transform
            bone_length [b,d]. The provided bone length.

        Returns:
            Y_rec_new: [t,b,d] The projected bone transform
            J_locs_fk: [t,b,J,3]. The joint locations via forward kinematics.
        """
        nt, nb = Y_rec.shape[:2]
        Y_rec_ = Y_rec.contiguous().view(nt, nb, self.nj, -1)
        r_locs = Y_rec_[:,:,0,:3]
        rotcont = Y_rec_[:,:,:,3:]
        rotmat = self._cont2rotmat(rotcont)
        rotcont_new = self._rotmat2cont(rotmat)
        bone_length = bone_length.unsqueeze(0).repeat(nt, 1,1)
        
        J_locs_fk = self.body_model.forward_kinematics(r_locs.reshape(nt*nb, 1, 3), 
                                        bone_length.reshape(nt*nb, -1), 
                                        rotmat.reshape(nt*nb, self.nj, 3,3))
        J_locs_fk = J_locs_fk.reshape(nt, nb, self.nj, 3)
        Y_rec_new = torch.cat([J_locs_fk, rotcont_new],dim=-1).contiguous().view(nt, nb, -1)
        
        return Y_rec_new, J_locs_fk


    def forward_train(self, X, Y, 
                bone_length: torch.Tensor,
                ):
        """the forward pass for generating the body joints, provided the input arguments.
        
        Args:
            X: the motion seed, [t,b,d]. When t=0, change from prediction to generation.
            Y: the ground truth prediction, [t,b,d]
            betas: shape parameter, [b,J]
        
        Returns:
            Y_rec: the reconstructed prediction, [t,b,d]
            mu, logvar: the inference posterior
            
        """
        
        '''marker predictor'''
        Y_rec, mu, logvar = self.predictor(X, Y)
        '''reprojection via forward kinematics'''
        Y_rec_fk, _ = self.reprojection(Y_rec, bone_length)
        Y_rec_b = self.reproj_factor*Y_rec_fk + (1.-self.reproj_factor)*Y_rec
        return Y_rec_b, mu, logvar




