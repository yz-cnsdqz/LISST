'''Information
- This script is to format the original CMU dataset into the AMASS format.
- Rather than working on markers, we directly working on bone transforms. I.e. joint locations and joint rotations (w.r.t. rotation matrix)
- Each time run will re-write all previous generated results.
- See the code for more details.
'''

import numpy as np
import os, glob, sys
from amc_parser import parse_amc, parse_asf
import pdb, pickle


if __name__=='__main__':
#   lv0 = './data'
  lv0 = '/mnt/hdd/datasets/CMU/allasfamc/all_asfamc/subjects'
  lv1s = os.listdir(lv0)
#   output_folder = '/mnt/hdd/datasets/CMU_ours'
    
  for lv1 in lv1s:
    lv2s = os.listdir('/'.join([lv0, lv1]))
    asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)
    print('- parsing %s' % asf_path)
    joints = parse_asf(asf_path)
    # ccs = []
    # for _, val in joints.items():
    #     ccs.append(val.C)
    # ccc = np.stack(ccs)
    # print(repr(ccc))
    # for key, val in joints.items():
    #     print('\'{}\':'.format(key))
    #     print('{},'.format(repr(val.C)))
    
    # pdb.set_trace()
    motions_list = sorted(glob.glob(os.path.join(lv0, lv1, lv1+'_*.amc')))
    iidx = -1
    for motion in motions_list:
        iidx += 1
        output_filename = os.path.dirname(asf_path)+'/motion_{:05d}.pkl'.format(iidx)
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))

        motion_data = parse_amc(motion)
        motion_parsed = {
            'global_transl': [], #[t,3]
            'global_rotmat': [], #[t,3,3]
            'joints_rotmat': [], #[t,J,3,3]
            'global_rotmat_local': [], #[t,3,3]
            'joints_rotmat_local': [], #[t,J,3,3]
            'joints_length': [], #[t,J+1], incl. the root, w.r.t. parents
            'joints_locs':[], # #[t,J,3]
        }

        
        for frame_idx in range(len(motion_data)):
            frame_parsed = {
                'global_transl': [],
                'global_rotmat': [],
                'joints_rotmat': [],
                'global_rotmat_local': [], #[t,3,3]
                'joints_rotmat_local': [], #[t,J,3,3]
                'joints_length': [],
                'joints_locs':[],
            }
            joints['root'].set_motion(motion_data[frame_idx])

            for jname in list(joints.keys()):
                if jname == 'root':
                    frame_parsed['global_transl'].append(joints[jname].coordinate.squeeze())
                    frame_parsed['global_rotmat'].append(joints[jname].matrix)
                    frame_parsed['global_rotmat_local'].append(joints[jname].matrix_local)
                    frame_parsed['joints_length'].append(joints[jname].length)
                else:
                    frame_parsed['joints_locs'].append(joints[jname].coordinate.squeeze())
                    frame_parsed['joints_rotmat'].append(joints[jname].matrix)
                    frame_parsed['joints_rotmat_local'].append(joints[jname].matrix_local)
                    frame_parsed['joints_length'].append(joints[jname].length)
            for key in frame_parsed.keys():
                if 'global' in key:
                    frame_parsed[key] = frame_parsed[key][0]
                else:
                    frame_parsed[key] = np.stack(frame_parsed[key])

            for key, val in frame_parsed.items():
                motion_parsed[key].append(val)
            
        for key in frame_parsed.keys():
            motion_parsed[key] = np.stack(motion_parsed[key])
        
        with open(output_filename, 'wb') as f:
            pickle.dump(motion_parsed, f)
