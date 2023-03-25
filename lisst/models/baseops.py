"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import os, sys, glob
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter
import torchgeometry as tgm
from torch.optim import lr_scheduler
import logging
import datetime








"""
===============================================================================
basic functionalities
===============================================================================
"""

# Note that python 3.10 has a build-in implementation
from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)







"""
===============================================================================
basic network modules
===============================================================================
"""


class MLP(nn.Module):
    '''wrapper of MLPs, modified from https://github.com/Khrylx/DLow'''
    def __init__(self, 
                in_dim: int,
                h_dims: list=[128,128], 
                activation: str='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU()
        
        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            net = nn.Linear(in_dim_, h_dim)
            self.layers.append(net)
            in_dim_ = h_dim

    def forward(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x










class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def _sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, *args, **kwargs):
        pass

    def decode(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass

    def sample_prior(self, *args, **kwargs): #[t, b, d]
        pass


import math
class PositionalEncoding(nn.Module):
    '''referring to the official implementation in pytorch'''
    def __init__(self, nd, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, nd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nd, 2).float() * (-math.log(10000.0) / nd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [t,b,d]
        '''
        x = x + self.pe[:x.size(0), :]
        return x



class TrainOP:
    def __init__(self, modelconfig, lossconfig, trainconfig):
        self.dtype = torch.float32
        gpu_index = trainconfig.get('gpu_index',0)
        self.device = torch.device('cuda',
                index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.modelconfig = modelconfig
        self.lossconfig = lossconfig
        self.trainconfig = trainconfig
        self.logger = get_logger(self.trainconfig['log_dir'])
        self.writer = SummaryWriter(log_dir=self.trainconfig['log_dir'])

    def build_model(self):
        pass

    def calc_loss(self):
        pass

    def train(self):
        pass


class TestOP:
    def __init__(self, modelconfig, testconfig, *args):
        self.dtype = torch.float32
        gpu_index = testconfig['gpu_index']
        if gpu_index >= 0:
            self.device = torch.device('cuda',
                    index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.modelconfig = modelconfig
        self.testconfig = testconfig

        if not os.path.exists(self.testconfig['ckpt_dir']):
            print('[ERROR]: no model was trained. Program terminates.')
            sys.exit(-1)

    def build_model(self):
        pass

    def visualize(self):
        pass

    def evaluation(self):
        pass











"""
===============================================================================
basic helper functions
===============================================================================
"""

def get_logger(log_dir: str, 
               mode: str='train'):
    """create a logger to help training

    Args:
        log_dir (str): the path to save the log file
        mode (str, optional): train or test mode. Defaults to 'train'.

    Returns:
        _type_: a logger class
    """
    logger = logging.getLogger(log_dir)
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(log_dir, '{}_{}.log'.format(mode, ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def get_scheduler(optimizer, policy, num_epochs_fix=None, num_epochs=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - num_epochs_fix) / float(num_epochs - num_epochs_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)
    else:
        return NotImplementedError('scheduler with {} is not implemented'.format(policy))
    return scheduler





# ===============================================================================
# geometric transformations
# ===============================================================================


class RotConverter(nn.Module):
    """Rotation conversion between different representations

        - requires torch tensors
        
        - all functions only support data_in with [N, num_joints, D].
            - N can be n_batch, or n_batch*n_time

    """
    
    def __init__(self):
        super(RotConverter, self).__init__()

    @staticmethod
    def cont2rotmat(rotcont: torch.Tensor)->torch.Tensor:
        """Conversion from 6D representation to rotation matrix 3x3.
        
        Args:
            rotcont (torch.Tensor): [b, 6]

        Returns:
            torch.Tensor: rotation matrix, [b,3,3]
        """
        '''
        - data_in bx6
        - return: pose_matrot: bx3x3
        '''
        rotcont_ = rotcont.contiguous().view(-1, 3, 2)
        b1 = F.normalize(rotcont_[:, :, 0], dim=-1)
        dot_prod = torch.sum(b1 * rotcont_[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(rotcont_[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)#[b,3,3]


    @staticmethod
    def cont2rotmat_jacobian(rotcont: torch.Tensor, 
                ):
        """cont2rotmat and return analytical jacobian matrix, same to autodiff result

        Args:
            rotcont (torch.Tensor): [b,6]
            
        Returns:
            rotmat: torch.tensor, [b,3,3]
            jacobian matrix: torch.Tensor, [b,9,6]
        """
        rotcont_ = rotcont.contiguous().view(-1, 3, 2)
        a1 = rotcont_[:,:,0] #[b,3]
        a2 = rotcont_[:,:,1]
        nb, nd = a1.shape
        idty = torch.eye(nd).unsqueeze(0).to(a1.device)

        def ff(x):
            """batched derivative of l2 normalization

            Args:
                x (torch.Tensor): [b,d]
            """
            bb = idty- \
                torch.matmul(x.unsqueeze(-1), x.unsqueeze(1))/torch.matmul(x.unsqueeze(1), x.unsqueeze(-1))
            return bb/torch.linalg.norm(x,dim=-1,keepdim=True).unsqueeze(-1)
        
        def gg(x,y):
            """batched derivative of complement of linear projection to 
                unit vector x w.r.t. x, i.e. d((I-x*x.T)*y) / dx

            Args:
                x (torch.Tensor): [b,d]
                y (torch.Tensor): [b,d]
            """
            aa = torch.matmul(x.unsqueeze(1), y.unsqueeze(-1))*idty
            bb = torch.matmul(x.unsqueeze(-1), y.unsqueeze(1))
            return -(aa+bb)
        
        def skew(x):
            """obtain the skew-synmetric matrix of a 3D vector, batched.
            Part of the cross product representation

            Args:
                x (torch.Tensor): [b,d]
            """
            aa = torch.zeros(nb, nd, nd).to(x.device)
            aa[:,0,1] = -x[:,2]
            aa[:,0,2] = x[:,1]
            aa[:,1,0] = x[:,2]
            aa[:,1,2] = -x[:,0]
            aa[:,2,0] = -x[:,1]
            aa[:,2,1] = x[:,0]
            
            return aa
            
        
        # forward pass
        b1 = F.normalize(a1, dim=-1)
        dot_prod = torch.sum(b1 * a2, dim=1, keepdim=True)
        b2_tilde = a2 - dot_prod * b1
        b2 = F.normalize(b2_tilde, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)        
        rotmat = torch.stack([b1, b2, b3], dim=-1)#[b,3,3]
        
        ## jacobian blocks
        db1_da1 = ff(a1.detach()) #[b,3,3]
        db1_da2 = torch.zeros(nb,3,3).to(rotcont.device)
        db2_da1 = ff(b2_tilde.detach()) @ gg(b1.detach(),a2.detach()) @ db1_da1
        db2_da2 = ff(b2_tilde.detach()) @ \
            (torch.eye(nd).unsqueeze(0).to(rotcont.device)-torch.matmul(b1.unsqueeze(-1), b1.unsqueeze(1)))
        db3_da1 = skew(b2.detach()).permute(0,2,1) @ db1_da1 + skew(b1.detach())@db2_da1
        db3_da2 = skew(b2.detach()).permute(0,2,1) @ db1_da2 + skew(b1.detach())@db2_da2
        
        jacobian1 = torch.cat([db1_da1, db2_da1, db3_da1],dim=1)
        jacobian2 = torch.cat([db1_da2, db2_da2, db3_da2],dim=1)
        jacobian = torch.cat([jacobian1, jacobian2],dim=-1)
        
        # permute to fit tensor.reshape(), which cat entries row by row
        jacobian = jacobian[:,:,[0,3,1,4,2,5]] 
        jacobian = jacobian[:,[0,3,6,1,4,7,2,5,8],:] 
        
        return rotmat, jacobian # permute to fit tensor.reshape()



    @staticmethod
    def rotmat2cont(rotmat: torch.Tensor)->torch.Tensor:
        """conversion from rotation matrix to 6D.

        Args:
            rotmat (torch.Tensor): [b,3,3]

        Returns:
            torch.Tensor: [b,6]
        """
        rotmat = rotmat[:,:,:-1]
        rotmat_ = rotmat.contiguous().view(-1, 3, 2)
        return rotmat_.view(-1, 3*2)


    @staticmethod
    def aa2cont(rotaa:torch.Tensor)->torch.Tensor:
        """axis-angle to 6D rot

        Args:
            rotaa (torch.Tensor): axis angles, [b, num_joints, 3]

        Returns:
            torch.Tensor: 6D representations, [b, num_joints, 6]
        """
        
        nb = rotaa.shape[0]
        rotcont = tgm.angle_axis_to_rotation_matrix(rotaa.reshape(-1, 3))[:, :3, :2].contiguous().view(nb, -1, 6)
        
        return rotcont


    @staticmethod
    def cont2aa(rotcont: torch.Tensor)->torch.Tensor:
        """6D continuous rotation to axis-angle

        Args:
            rotcont (torch.Tensor): [b, num_joints, 6]

        Returns:
            torch.Tensor: [b, num_joints, 3]
        """
        
        batch_size = rotcont.shape[0]
        x_matrot_9d = RotConverter.cont2rotmat(rotcont).view(batch_size,-1,9)
        x_aa = RotConverter.rotmat2aa(x_matrot_9d).contiguous().view(batch_size, -1, 3)
        return x_aa


    @staticmethod
    def rotmat2aa(rotmat: torch.Tensor)->torch.Tensor:
        """from rotation matrix to axis angle

        Args:
            rotmat (torch.Tensor): [b, num_joints, 9] or [b, num_joints, 3,3]

        Returns:
            torch.Tensor: [b, num_joints, 3]
        """
        if rotmat.ndim==5:
            nt,nb = rotmat.shape[:2]
        
        homogen_rotmat = F.pad(rotmat.contiguous().view(-1, 3, 3), [0,1])
        rotaa = tgm.rotation_matrix_to_angle_axis(homogen_rotmat).contiguous().view(-1, 3)
        if rotmat.ndim==5:
            rotaa = rotaa.contiguous().view(nt, nb, -1,3)
        
        return rotaa


    @staticmethod
    def aa2rotmat(rotaa:torch.Tensor)->torch.Tensor:
        """axis angle to rotation matrix

        Args:
            rotaa (torch.Tensor): [b, num_joints, 3]

        Returns:
            torch.Tensor: [b, num_joints, 9]
        """
        
        nb = rotaa.shape[0]
        rotmat = tgm.angle_axis_to_rotation_matrix(rotaa.reshape(-1, 3))[:, :3, :3].contiguous().view(nb, -1, 9)

        return rotmat




class CanonicalCoordinateExtractor:
    """Summary of class here.

    - Motion canonicalization as in MOJO and GAMMA
    
    - When the model runs recursively, we need to reset the coordinate and perform canonicalization on the fly.
    This class provides such functionality.
    
    - When specifying the joint locations of the motion primitive, it produces a new canonical coordinate, according to
    the reference frame.
    
    - Only torch is supported.
    
    Attributes:
        device: torch.device, to specify the device when the input is torch.tensor
    """

    def __init__(self, device=torch.device('cuda:0')):
        self.device = device

    def get_new_coordinate(self,
                            pelvis: torch.Tensor, #[b,3]
                            lhip: torch.Tensor, #[b,3]
                            rhip: torch.Tensor, #[b,3]
        ) -> Tuple[torch.Tensor]:
        x_axis = rhip - lhip #[b,3]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis,dim=-1, keepdim=True)
        z_axis = torch.FloatTensor([[0,0,1]]).to(self.device).repeat(x_axis.shape[0], 1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis/torch.norm(y_axis,dim=-1, keepdim=True)
        new_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=-1) #[b,3,3]
        new_transl = pelvis.unsqueeze(1) #[b,1,3]
        return new_rotmat.detach(), new_transl.detach()
















# SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
# the following functions are from the repository: 

# https://github.com/CalciferZh/AMCParser
# SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS


from transforms3d.euler import euler2mat

class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone. 
      What are the lengths in CMU's asf/amc files? 
      ASF files in CMU mocap database have ":units->length" set to 0.45. That is because all the values are multiplied by 0.45 before they are stored to file (I am not sure why). 
      Also ASF files are stored in inches, so to convert to meters you need to multiply all length values by the following scale=(1.0/0.45)*2.54/100.0.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
    self.name = name
    self.direction = np.reshape(direction, [3, 1])
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    self.scale = scale = (1.0/0.45)*2.54/100.0
    # scale=1
    self.length = length*scale
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None

def read_line(stream, idx):
    if idx >= len(stream):
        return None, idx
    line = stream[idx].strip().split()
    idx += 1
    return line, idx

def parse_asf(file_path):

  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])
    # print(direction)

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints
