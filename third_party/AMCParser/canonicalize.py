'''canonicalize the processed CMU mocap data 

- Like in GAMMA, this one is to train generative motion primitives 
- Since no body model is involved. Here we only use the original bone transforms.
- The body markers are not supported to this end.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob
import numpy as np
from tqdm import tqdm
import torch
import argparse



import json
import csv
import pdb


def test_vis_body(J_locs):
    import open3d as o3d
    #### world coordinate
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    
    ## create body mesh from data
    ball_list = []
    for i in range(31):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        ball.translate(J_locs[i], relative=False)
        ball_list.append(ball)

    o3d.visualization.draw_geometries(ball_list+[coord])



ROT_NEGATIVE_X = torch.tensor([  [1.0000000,  0.0000000,  0.0000000],
                             [0.0000000,  0.0000000,  1.0000000],
                             [0.0000000, -1.0000000,  0.0000000]])

ROT_POSITIVE_X = torch.tensor([  [1.0000000,  0.0000000,  0.0000000],
                             [0.0000000,  0.0000000,  -1.0000000],
                             [0.0000000, 1.0000000,  0.0000000]])


def get_new_coordinate(root, left_hip, right_hip, flipx=True):
    '''
    this function produces transform from body local coordinate to the world coordinate.
    it takes only a single frame.
    local coodinate:
        - located at the root (pelvis?) 
        - x axis: from left hip to the right hip
        - z axis: point up (negative gravity direction)
        - y axis: pointing forward, following right-hand rule
    '''
    x_axis = right_hip-left_hip
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0,0,1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)
    global_ori_new = np.stack([x_axis, y_axis, z_axis], axis=-1)
    transl_new = root # put the local origin to teh root
    if flipx:
        global_ori_new = np.einsum('ij,jk->ik', ROT_NEGATIVE_X, global_ori_new)

    return global_ori_new, transl_new



if __name__=='__main__':
    #### set subsequence length
    ### reference:https://backyardbrains.com/experiments/reactiontime#:~:text=The%20average%20reaction%20time%20for,seconds%20for%20a%20touch%20stimulus.
    # The average reaction time for humans is 0.25 seconds to a visual stimulus, 0.17 for an audio stimulus, and 0.15 seconds for a touch stimulus.
    # len_subseq = 1200 # for each motion primitive, 30frames = 0.25 seconds if 120fps
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_primitives', type=int, default=8) #how many primtives in one file?
    parser.add_argument('--cmu_path', default='/mnt/hdd/datasets/CMU/allasfamc/all_asfamc/subjects') #should we only consider locomotion e.g. walking?
    parser.add_argument('--split', default='train') #should we only consider locomotion e.g. walking?
    args = parser.parse_args()
    
    N_MPS = args.num_primitives
    MP_LENGTH = 30 # 0.25 sec. here we assume all framerates are 120fps, according to the original CMU dataset info
    #### set input output dataset paths
    cmu_dataset_path = args.cmu_path
    output_folder_prefix = '/mnt/hdd/datasets/CMU'
    if N_MPS > 1:
        cmu_path_out = output_folder_prefix+'-canon-MPx{:d}-{}/'.format(N_MPS, args.split)
    else:
        cmu_path_out = output_folder_prefix+'-canon-{}/'.format(args.split)
    if not os.path.exists(cmu_path_out):
        os.makedirs(cmu_path_out)
    
    # subjects = os.listdir(cmu_dataset_path)
    subjects_train = ['21', '29', '08', '88', '74', '34', '63', '142', '16', '62', '05', '14',
    '36', '104', '138', '106', '26', '114', '122', '136', '19', '01', '120', '94', '54', '135', 
    '56', '137', '128', '131', '02', '82', '61', '49', '76', '46', '134', '108', '103', '84', '83', '17', '79', 
    '107', '09', '89', '118', '78', '55', '60', '11', '06', '85', '86', '140', '18', '10', '73', '91', '23', '123', '124', 
    '77', '132', '87', '70', '31', '125', '39', '127', '126', '139']

    subjects_test = ['43', '32', '45', '90', '38', '03', '22', '15', '07', '111', '30', '40', '117', '113', '81', '75', '13',
     '102', '20', '69', '41', '33', '24', '133', '42', '115', '27', '35', '12', '47', '28', 
     '121', '25', '93', '37', '105', '143', '141', '80', '64']

    subjects_all = subjects_train+subjects_test
    
    
    if args.split == 'all':
        seqs = glob.glob(os.path.join(cmu_dataset_path, '*/*.pkl'))
    elif args.split == 'train':
        seqs = []
        for ss in subjects_train:
            seqs+= glob.glob(os.path.join(cmu_dataset_path, ss, '*.pkl'))
    elif args.split == 'test':
        seqs = []
        for ss in subjects_test:
            seqs+= glob.glob(os.path.join(cmu_dataset_path, ss, '*.pkl'))
    else:
        raise ValueError('split should be [train, test, all]')
    
    index_subseq = 0 # index subsequences for subsets separately
    #### main loop to process each sequence
    for seq in tqdm(seqs):
        data = dict(np.load(seq, allow_pickle=True))
        len_subseq = int(MP_LENGTH*N_MPS )
        # motion_parsed = {
        #     'global_transl': [], #[t,3]
        #     'global_rotmat': [], #[t,3,3]
        #     'joints_rotmat': [], #[t,J,3,3]
        #     'joints_length': [], #[t,J+1], incl. the root, w.r.t. parents
        #     'joints_locs':[], # #[t,J,3]
        # }
        
        ## skip too short sequences
        n_frames = data['global_transl'].shape[0]
        if n_frames < len_subseq:
            continue

        t = 0
        while t < n_frames:
            if t+len_subseq >= n_frames:
                break
            
            ## get subsequence and setup IO
            outfilename = os.path.join(cmu_path_out, 'subseq_{:07d}.npz'.format(index_subseq))
            data_sub = {}
            for key, val in data.items():
                data_sub[key] = val[t:t+len_subseq]
            
            # test_vis_body(np.concatenate([data_sub['global_transl'][:1],
            #                     data_sub['joints_locs'][0]]))
            
            ### rotate body about X-axis by 90 degree. from Y-up to Z-up
            data_sub['global_transl'] = np.einsum('ij,tj->ti', ROT_POSITIVE_X, data_sub['global_transl'])
            data_sub['global_rotmat'] = np.einsum('ij,tjk->tik', ROT_POSITIVE_X, data_sub['global_rotmat'])
            # data_sub['global_rotmat_local'] = np.einsum('ij,tjk->tik', ROT_POSITIVE_X, data_sub['global_rotmat_local'])
            data_sub['joints_locs'] = np.einsum('ij,tpj->tpi', ROT_POSITIVE_X, data_sub['joints_locs'])
            data_sub['joints_rotmat'] = np.einsum('ij,tpjk->tpik', ROT_POSITIVE_X, data_sub['joints_rotmat'])
            # data_sub['joints_rotmat_local'] = np.einsum('ij,tpjk->tpik', ROT_POSITIVE_X, data_sub['joints_rotmat_local'])
        
            # test_vis_body(np.concatenate([data_sub['global_transl'][:1],
            #                     data_sub['joints_locs'][0]]))
            
            ### canonicalize the bodies
            global_transl0 = data_sub['global_transl'][0]
            lhip = data_sub['joints_locs'][0,0]
            rhip = data_sub['joints_locs'][0,5]
            
            transf_rotmat, transf_transl = get_new_coordinate(global_transl0, lhip, rhip, flipx=False)
            data_sub['global_transl'] = np.einsum('ij,tj->ti', transf_rotmat.T, data_sub['global_transl']-transf_transl[None,...])
            data_sub['global_rotmat'] = np.einsum('ij,tjk->tik', transf_rotmat.T, data_sub['global_rotmat'])
            # data_sub['global_rotmat_local'] = np.einsum('ij,tjk->tik', transf_rotmat.T, data_sub['global_rotmat_local'])
            data_sub['joints_locs'] = np.einsum('ij,tpj->tpi', transf_rotmat.T, data_sub['joints_locs']-transf_transl[None,None,...])
            data_sub['joints_rotmat'] = np.einsum('ij,tpjk->tpik', transf_rotmat.T, data_sub['joints_rotmat'])
            # data_sub['joints_rotmat_local'] = np.einsum('ij,tpjk->tpik', transf_rotmat.T, data_sub['joints_rotmat_local'])
                
            # test_vis_body(np.concatenate([data_sub['global_transl'][:1],
            #                     data_sub['joints_locs'][0]]))
            
            np.savez(outfilename, **data_sub)
            t = t+len_subseq
            index_subseq = index_subseq+1