"""
This script is to train the LISST model. The training has two parts:
    - load the template data and change them into model parameters.
    - Train the body shape PCA space.
"""
import os, sys, glob
sys.path.append(os.path.join(os.getcwd(), 'lisst'))
sys.path.append(os.getcwd())

import time
import numpy as np
import torch
import argparse

from torch import nn, optim
from torch.nn import functional as F

from lisst.utils.config_creator import ConfigCreator
from lisst.utils.config_creator import ConfigCreator
from lisst.utils.batch_gen import BatchGeneratorCMUCanonicalized
from lisst.utils.config_env import *

from lisst.models.body import LISSTPoser, LISSTPoserLite, LISSTCore
from lisst.models.baseops import (TrainOP, get_logger, get_scheduler, RotConverter)




class LISSTPoserTrainOP(TrainOP):

    def build_model(self):
        self.model = LISSTPoser(self.modelconfig)
        self.model.train()
        self.model.to(self.device)
        self.nj = self.model.nj
        
        
    def _calc_loss_rec(self, Y, Y_rec):
        '''
        - to calculate the reconstruction loss of markers, including the 1-st order derivative
        - Y and Y_rec should be in the format of [time, batch, dim]
        '''
        loss_rec = F.l1_loss(Y, Y_rec)

        return loss_rec


    def _calc_loss_kld(self, mu, logvar):
        loss_kld = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())            
        loss_kld = torch.sqrt(1 + loss_kld**2)-1
    
        return loss_kld


    def calc_loss(self, J_btransf,  epoch):
        nt, nb = J_btransf.shape[:2]
        
        '''prepare the input data'''
        J_btransf = J_btransf.contiguous().view(nt, nb, self.nj,-1)
        J_rotcont = J_btransf[:,:,:,3:].contiguous().view(nt, nb, -1)
        J_rotcont = J_rotcont.view(nt*nb, -1)
        Y = J_rotcont[torch.randperm(nt*nb)]
        
        if self.trainconfig['data_aug'] == 'canonical':
            # relatively rotation about the root rotation
            Y_rotmat = RotConverter.cont2rotmat(Y.contiguous().view(nt*nb*self.nj, -1))
            Y_rotmat = Y_rotmat.contiguous().view(nt*nb, self.nj, 3,3)
            Y_rotmat_new = torch.einsum('bpij,bpjk->bpik', Y_rotmat[:,:1].permute(0,1,3,2), Y_rotmat)
            Y = RotConverter.rotmat2cont(Y_rotmat_new.contiguous().view(nt*nb*self.nj, 3,3))
        else:
            raise ValueError('data augmentation should be *canonical*')
        
        # forward pass
        Y = Y.contiguous().view(nt*nb, self.nj, -1).detach()
        [Y_rec, mu, logvar] = self.model(Y)

        # rec loss
        loss_rec = self._calc_loss_rec(Y, Y_rec)
        
        # kl-divergence
        loss_kld = self._calc_loss_kld(mu, logvar)
        ## kl loss annealing
        weight_kld = self.lossconfig['weight_kld']
        if self.lossconfig['annealing_kld']:
            weight_kld = min( ( float(epoch) / (0.9*self.trainconfig['num_epochs']) ), 1.0) * self.lossconfig['weight_kld']
        
        loss = loss_rec + weight_kld * loss_kld #+ self.lossconfig['weight_fk']*loss_fk
        loss_info = np.array([loss.item(), loss_rec.item(), loss_kld.item()])

        return loss, loss_info


    def train(self, batch_gen):
        self.build_model()

        starting_epoch = 0
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.trainconfig['learning_rate'])
        scheduler = get_scheduler(optimizer, policy='lambda',
                                    num_epochs_fix=self.trainconfig['num_epochs_fix'],
                                    num_epochs=self.trainconfig['num_epochs'])

        if self.trainconfig['resume_training']: # can also be used for fine-tuning
            ckp_list = sorted(glob.glob(os.path.join(self.trainconfig['save_dir'],
                                        'epoch-*.ckp')),
                                        key=os.path.getmtime)
            if len(ckp_list)>0:
                checkpoint = torch.load(ckp_list[-1])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if not self.trainconfig.get('fine_tune', False):
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    starting_epoch = checkpoint['epoch']
                    print('[INFO] --resume training from {}'.format(ckp_list[-1]))
                else:
                    print('[INFO] --fine-tune from {}'.format(ckp_list[-1]))
            else:
                raise FileExistsError('the pre-trained checkpoint does not exist.')

        # training main loop
        loss_names = ['ALL', 'REC', 'KLD']
        for epoch in range(starting_epoch, self.trainconfig['num_epochs']):
            epoch_losses = 0
            epoch_nsamples = 0
            stime = time.time()
            ## training subloop for each epoch
            while batch_gen.has_next_rec():
                J_btransf= batch_gen.next_batch(self.trainconfig['batch_size'])
                optimizer.zero_grad()
                loss, losses_items = self.calc_loss(J_btransf, epoch)
                loss.backward(retain_graph=False)
                optimizer.step()
            epoch_losses += losses_items
            epoch_nsamples += 1
            
            batch_gen.reset()            
            scheduler.step()
            
            ## logging
            epoch_losses /= epoch_nsamples
            eps_time = time.time()-stime
            lr = optimizer.param_groups[0]['lr']
            info_str = '[epoch {:d}]:'.format(epoch+1)
            for name, val in zip(loss_names, epoch_losses):
                self.writer.add_scalar(name, val, epoch+1)
                info_str += '{}={:04f}, '.format(name, val)
            info_str += 'time={:04f}, lr={:f}'.format(eps_time, lr)

            self.logger.info(info_str)

            if ((1+epoch) % self.trainconfig['saving_per_X_ep']==0) :
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            # 'scheduler_state_dict': scheduler.state_dict()
                            }, self.trainconfig['save_dir'] + "/epoch-" + str(epoch + 1) + ".ckp")
            
        if self.trainconfig['verbose']:
            print('[INFO]: Training completes!')
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, required=True)
    parser.add_argument('--resume_training', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
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
    traincfg['resume_training'] = True if args.resume_training==1 else False
    traincfg['verbose'] = True if args.verbose==1 else False
    traincfg['gpu_index'] = args.gpu_index
    traincfg['data_aug'] = traincfg.get('data_aug', False)

    """data"""
    data_path = get_cmu_canonicalizedx8_path(split=traincfg['cmu_canon_split'])
    batch_gen = BatchGeneratorCMUCanonicalized(data_path=data_path,
                                            sample_rate=3, # fps=40
                                            body_repr=modelcfg['body_repr'])
    batch_gen.get_rec_list(to_gpu=True)
    

    """model and trainop"""
    print('--training {}'.format(args.cfg))
    trainop = LISSTPoserTrainOP(modelcfg, losscfg, traincfg)
    trainop.train(batch_gen)
