import os
import sys
import numpy as np
import open3d as o3d
import torch
import smplx
import cv2
import pickle
import pdb
import re
import glob

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from lisst.utils.vislib import *


def visualize(jts, outfile_path=None, datatype='kps'):
    ## prepare data
    n_frames, n_jts = jts.shape[:2]
    
    ## prepare visualizer
    np.random.seed(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540,visible=True)
    # vis.create_window(width=480, height=270,visible=True)
    render_opt=vis.get_render_option()
    render_opt.mesh_show_back_face=True
    render_opt.line_width=10
    render_opt.point_size=5
    render_opt.background_color = color_hex2rgb('#1c2434')
    vis.update_renderer()

    ### top lighting
    box = o3d.geometry.TriangleMesh.create_box(width=200, depth=1,height=200)
    box.translate(np.array([-200,-200,6]))
    vis.add_geometry(box)
    vis.poll_events()
    vis.update_renderer()

    #### world coordinate
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    vis.add_geometry(coord)
    vis.poll_events()
    vis.update_renderer()

    ## create body mesh from data
    ball_list = []
    for i in range(n_jts):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        vis.add_geometry(ball)
        vis.poll_events()
        vis.update_renderer()
        ball_list.append(ball)


    
    frame_idx = 0
    cv2.namedWindow('frame2')
    for it in range(0,n_frames):
        for i,b in enumerate(ball_list):
            b.translate(jts[it,i], relative=False)
            vis.update_geometry(b)

        
        ## update camera.
        ctr = vis.get_view_control()
        ctr.set_constant_z_far(10)
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        ### get cam T
        # body_t = np.mean(data[it],axis=0) # let cam follow the body
        body_t = np.array([0,0,0])
        cam_t = body_t + 2.0*np.ones(3)
        ### get cam R
        cam_z =  body_t - cam_t
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.array([cam_z[1], -cam_z[0], 0.0])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.array([cam_z[0], cam_z[1], -(cam_z[0]**2 + cam_z[1]**2)/cam_z[2] ])
        cam_y = cam_y / np.linalg.norm(cam_y)
        cam_r = np.stack([cam_x, -cam_y, cam_z], axis=1)
        ### update render cam
        transf = np.eye(4)
        transf[:3,:3]=cam_r
        transf[:3,-1] = cam_t
        cam_param = update_render_cam(cam_param, transf)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()

        ## capture RGB appearance
        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
        if outfile_path is not None:
            renderimgname = os.path.join(outfile_path, 'img_{:05d}.png'.format(frame_idx))
            frame_idx = frame_idx + 1
            cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        cv2.waitKey(5)



if __name__=='__main__':
    proj_path = os.getcwd()
    exps = [
            'lisst/LISST_POSER_v0'
            ]
    bodies = 5
    
    # for exp in exps:
    #     res_file_list = sorted(glob.glob(proj_path+'/results/{}/results/motion_gen_seed0/results_*.pkl'.format(exp)))
    #     for results_file_name in res_file_list:
    results_file_name = '/home/yzhang/workspaces/LISST-dev/results/CMUIK_20230220-203541/CMU-canon-MPx8-personal.pkl'
    data = np.load(results_file_name, allow_pickle=True)
    jts_pred = data['J_locs_3d'] #list of [t,b, J, d]
    # breakpoint()
    for body in range(bodies):
        joints_locs_all = jts_pred[:,body]
        renderfolder = '/tmp/render_cmu/body_{:d}'.format(body)
        if not os.path.exists(renderfolder):
            os.makedirs(renderfolder)
        visualize(joints_locs_all,
                outfile_path=renderfolder, datatype='kps')


