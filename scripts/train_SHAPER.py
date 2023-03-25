"""
This script is to train the LISSTCore model. The training has two parts:
    - load the template data and change them into model parameters.
    - Train the body shape PCA space.
"""

import os, sys, glob
sys.path.append(os.getcwd())
import time
import numpy as np
import open3d as o3d
import torch
import pdb
import argparse
from lisst.utils.config_creator import ConfigCreator
from lisst.models.baseops import *
from lisst.models.body import LISSTCore


class LISSTTrainOP(TrainOP):

    def build_model(self):
        self.model = LISSTCore(self.modelconfig)
        self.model.train()
        self.model.to(self.device)
        self.add_nose = self.modelconfig.get('add_nose', False)
        self.add_heels = self.modelconfig.get('add_heels', False)

    def load_template_from_asf(self, asf_path):
        '''
        - extract basic attributes about the CMU body skeleton. 
        - developed based on https://github.com/CalciferZh/AMCParser
        - the asf_path file is directly from the CMU mocap dataset.
        - might be only used once before training.
        - NOTE this script is only used during training. Therefore, remove this function in the released version.

        Args:
          - asf_path: the full filepath. 
        '''
        
        joints = parse_asf(asf_path)

        
        self.model.joint_names = list(joints.keys())
        template_joint_directions = {}
        
        for key, val in joints.items():
            self.model.children_table[key] = [jts.name for jts in val.children]
            template_joint_directions[key] = torch.tensor(val.direction).squeeze() #[3]
        
        template_joint_directions = torch.stack(list(template_joint_directions.values()),dim=0)

        if self.add_nose:
            ## add nose to constrain the head rotation.
            self.model.joint_names.append('nose')
            self.model.children_table['head'].append('nose')
            self.model.children_table['nose'] = []
            nose_direction = torch.tensor([[0,0,1]])
            template_joint_directions = torch.cat([template_joint_directions,nose_direction])

        if self.add_heels:
            ## add lheel and rheel, respectively
            self.model.joint_names.append('lheel')
            self.model.children_table['ltibia'].append('lheel')
            self.model.children_table['lheel'] = []
            lheel_direction = torch.tensor([[ 0.       , -0.8660254, -0.5      ]])
            template_joint_directions = torch.cat([template_joint_directions,lheel_direction])
            ## add lheel and rheel, respectively
            self.model.joint_names.append('rheel')
            self.model.children_table['rtibia'].append('rheel')
            self.model.children_table['rheel'] = []
            rheel_direction = torch.tensor([[ 0.       , -0.8660254, -0.5      ]])
            template_joint_directions = torch.cat([template_joint_directions,rheel_direction])

        self.model.template_joint_directions.copy_(template_joint_directions) #[J,3]
        

    def train(self):
        self.build_model()

        print('-- learn the model from the CMU template')
        asf_path = self.trainconfig['asf_path']
        self.load_template_from_asf(asf_path)
        
        print('-- learn the shape space from CMU data')
        subjects_split = self.trainconfig.get('subjects_split', 'all')
        if subjects_split=='all':
            files = glob.glob(os.path.join(self.trainconfig['subjects_path'],
                                '*/motion_00000.pkl'))
        elif subjects_split=='train':
            subjects_list = ['21', '29', '08', '88', '74', '34', '63', '142', '16', '62', '05', '14', 
                        '36', '104', '138', '106', '26', '114', '122', '136', '19', '01', '120', '94', '54', '135', 
                        '56', '137', '128', '131', '02', '82', '61', '49', '76', '46', '134', '108', '103', '84', '83', '17', '79', 
                        '107', '09', '89', '118', '78', '55', '60', '11', '06', '85', '86', '140', '18', '10', '73', '91', '23', '123', '124', 
                        '77', '132', '87', '70', '31', '125', '39', '127', '126', '139']
            files = []
            for ss in subjects_list:
                files+= glob.glob(os.path.join(self.trainconfig['subjects_path'],ss, 
                                'motion_00000.pkl'))
        elif subjects_split=='test':
            subjects_list = ['43', '32', '45', '90', '38', '03', '22', '15', '07', '111', '30', '40', '117', '113', '81', '75', '13',
                        '102', '20', '69', '41', '33', '24', '133', '42', '115', '27', '35', '12', '47', '28', 
                        '121', '25', '93', '37', '105', '143', '141', '80', '64']
            files = []
            for ss in subjects_list:
                files+= glob.glob(os.path.join(self.trainconfig['subjects_path'],ss,
                                'motion_00000.pkl'))
        else:
            raise ValueError('subjects split is not valid.')
             
             
        ## collect the bone lengths from individual subjects
        X = []
        for file in files:
            data = np.load(file, allow_pickle=True)['joints_length'][0]
            X.append(data)
        X = torch.tensor(np.stack(X)) #[num_subjects, num_joints=31]

        if self.add_nose:
            ## according to the human head statistics, distance from ear center to nose is 0.5*N(mu=20.4cm, sigma=0.778cm)
            X_nose = (torch.randn(X.shape[0],1).to(X.device)*0.778+20.4)*0.5/100 #in meters
            X = torch.cat([X, X_nose],dim=-1)
        
        if self.add_heels:
            ## according to the human foot statistics, distance from ankle to heel N(mu=80mm, sigma=6mm)
            X_heels = (torch.randn(X.shape[0],2).to(X.device)*6.+70.)/1000 #in meters
            X = torch.cat([X, X_heels],dim=-1)

        X_mean = X.mean(dim=0, keepdim=True)
        X_ = X-X_mean

        ## perform PCA
        if self.trainconfig['pca_algo'] == 'usv':
            print('-- -- performing USV...')
            u,s,vh = torch.linalg.svd(X_, full_matrices=False)
            eigval = s**2/(X.shape[0]-1)
            self.model.shape_basis.copy_(vh.T)
            self.model.meanshape.copy_(X_mean)
            self.model.eigvals.copy_(eigval)
            self.model.shape_basis2.copy_(torch.matmul(vh.T, torch.diag_embed(eigval)))
        else:
            raise NotImplemented('other pca algorithms are not implemented')

        torch.save({'model_state_dict': self.model.state_dict(),
                    'joint_names': self.model.joint_names,
                    'children_table': self.model.children_table,
                        }, self.trainconfig['save_dir'] + "/epoch-000.ckp")

        print('-- Training completes!')
        print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None) # specify the model to evaluate
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
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

    """train the body model"""
    trainop = LISSTTrainOP(modelcfg, losscfg, traincfg)
    
    trainop.train()













