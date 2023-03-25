"""this script is to sample skeletons from the learned shape pca"""

import os, sys, glob

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'lisst'))
sys.path.append(os.path.join(os.getcwd(), 'lisst','models'))
sys.path.append(os.path.join(os.getcwd(), 'lisst','utils'))
sys.path.append(os.path.join(os.getcwd(), 'lisst','cfg'))



import numpy as np
import open3d as o3d
from lisst.models.body import LISSTCore
from lisst.utils.config_creator import ConfigCreator


'''setup model'''
cfg = ConfigCreator('LISST_SHAPER_v2')
model = LISSTCore(cfg.modelconfig)
ckpt_path = '/home/yzhang/workspaces/LISST-dev/results/lisst/LISST_SHAPER_v2/checkpoints/epoch-000.ckp'
model.load(ckpt_path)


vis = o3d.visualization.Visualizer()
vis.create_window(width=960, height=540,visible=True)
# vis.create_window(width=480, height=270,visible=True)
render_opt=vis.get_render_option()
vis.update_renderer()


coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
vis.add_geometry(coord)
vis.poll_events()
vis.update_renderer()
    
ball_list = []
for i in range(model.num_kpts):
    ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    vis.add_geometry(ball)
    vis.poll_events()
    vis.update_renderer()
    ball_list.append(ball)



for basis in range(0,31):

    X = model.random_skeleton(n_basis=basis, n_samples=15)

    import cv2
    from lisst.utils.vislib import *
    cv2.namedWindow('frame2')
    for ii in range(X.shape[0]):
        for i,b in enumerate(ball_list):
            b.translate(X[ii,i], relative=False)
            vis.update_geometry(b)
        
        ## update camera.
        ctr = vis.get_view_control()
        ctr.set_constant_z_far(10)
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        ### get cam T
        # body_t = np.mean(data[it],axis=0) # let cam follow the body
        body_t = np.array([0,0,0])
        cam_t = body_t + 3*np.array([0,0,1])
        # ### get cam R
        cam_z =  body_t - cam_t
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.array([-1,0,0])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.array([0,-1,0 ])
        cam_y = cam_y / np.linalg.norm(cam_y)
        cam_r = np.stack([cam_x, cam_y, cam_z], axis=1)
        ### update render cam
        transf = np.eye(4)
        transf[:3,:3]=cam_r
        transf[:3,-1] = cam_t
        cam_param = update_render_cam(cam_param, transf)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()

        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
        folder = '/tmp/bodyshape2_b{:d}'.format(basis)
        if not os.path.exists(folder):
            os.makedirs(folder)
        renderimgname = os.path.join(folder, 'img_{:05d}.png'.format(ii))
        cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        cv2.waitKey(5)

    














