from IPython.display import clear_output
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim,compare_psnr, compare_mse
from utils import compute_psnr, plot_test

from pathlib import Path
from dataset import *
from tqdm import tqdm


def test(u,alpha,x_test,n_test,n_batch,n_steps,noise_type='Poisson', noise_snr=20,plot_loss=False):
    torch.cuda.set_device(0)
    N_mask = u.shape[1]
    x_test = np.expand_dims(x_test, axis=3)
    _, height, width, nc = x_test.shape
    x_test = x_test[:n_test,:,:,:].reshape(-1,nc,height,width)

    N_iter = int(np.ceil(n_test/n_batch))
    x_test_rec = np.zeros_like(x_test)

    eps_tensor = torch.cuda.FloatTensor([1e-15])
    epoch_idx = np.arange(n_test)

    for iters in tqdm(range(N_iter)):
#     for iters in range(N_iter):
        x = x_test[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:]
        x_gt = torch.cuda.FloatTensor(x).view(-1, 1, nc, height, width).cuda()
        uk = torch.cuda.FloatTensor(u).view(-1,N_mask,nc,height,width)

        # z = x * u, multiplicative masks
        z = x_gt * uk
        dummy_zeros = torch.zeros_like(z).cuda()
        z_complex = torch.cat((z.unsqueeze(5), dummy_zeros.unsqueeze(5)), 5)

        Fz = torch.fft(z_complex, 2, normalized=True)
        # y = |F(x*u)| = |Fz|
        y = torch.norm(Fz, dim=5)
        
        if noise_type=='Poisson':
            true_meas=y.cpu().detach()
            noise=np.random.normal(0,true_meas,(y.shape))

            noise_tensor=torch.cuda.FloatTensor(noise)
            noise_coeff=(y.pow(2).mean()/noise_tensor.pow(2).mean()/np.power(10,noise_snr/10.0)).pow(0.5)
            y=y+noise_coeff*noise_tensor  
            y=torch.relu(y)
        elif noise_type=='Gaussian':
            noise=np.random.normal(0,0.1,(y.shape))

            noise_tensor=torch.cuda.FloatTensor(noise)
            noise_coeff=(y.pow(2).mean()/noise_tensor.pow(2).mean()/np.power(10,noise_snr/10.0)).pow(0.5)
            y_=y+noise_coeff*noise_tensor
            y=torch.relu(y)
        else:
            print('Unsupported noise type')


        y_dual = torch.cat((y.unsqueeze(5), y.unsqueeze(5)), 5)

        x_est = x_test_rec[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:]
        x_est = torch.cuda.FloatTensor(x_est.reshape(-1,1,nc,height,width)).cuda()

        # image loss and measurement loss
        loss_x_pr=[]
        loss_y_pr=[]
        for kx in range(n_steps):

            z_est = x_est * uk + eps_tensor
            z_est_complex = torch.cat((z_est.unsqueeze(5), dummy_zeros.unsqueeze(5)), 5)
            Fz_est = torch.fft(z_est_complex,2, normalized=True)
            y_est = torch.norm(Fz_est,dim=5)
            y_est_dual = torch.cat((y_est.unsqueeze(5), y_est.unsqueeze(5)), 5)
            # angle Fz
            Fz_est_phase = Fz_est / (y_est_dual + eps_tensor)
            # update x
            x_grad_complex = torch.ifft( Fz_est - torch.mul(Fz_est_phase, y_dual), 2, normalized=True)
            x_grad = uk * x_grad_complex[:,:,:,:,:,0]
            x_grad = torch.sum(x_grad,dim=1)
            x_est = x_est - alpha * x_grad.view(x_est.shape)
            x_est = torch.clamp(x_est, 0, 1)

            # loss_x is image reconstruction loss, loss_y is the measurement loss
            loss_x_pr.append(np.mean((x-x_est.cpu().detach().numpy())**2))
            loss_y_pr.append(height*width*np.mean((y.cpu().detach().numpy() - y_est.cpu().detach().numpy())**2))

        x_test_rec[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:] = x_est.cpu().detach().numpy().reshape(-1,nc,height,width)

        if plot_loss:
            plt.figure(figsize = (12,4))
            plt.subplot(121)
            plt.plot(loss_x_pr)
            plt.yscale('log')
            plt.title(f'loss x @ iter {iters}')
            plt.subplot(122)
            plt.plot(loss_y_pr)
            plt.yscale('log')
            plt.title(f'loss y @ iter {iters}')
            plt.show()


    mse_list = [compare_mse(x_test[i,0,:,:],x_test_rec[i,0,:,:]) for i in range(n_test)]
    psnr_list = [compute_psnr(x_test[i,0,:,:],x_test_rec[i,0,:,:]) for i in range(n_test)]
    ssim_list = [compare_ssim(x_test[i,0,:,:],x_test_rec[i,0,:,:]) for i in range(n_test)]
    mean_of_psnr = np.mean(psnr_list) 
    psnr_of_mean = 20*np.log10((np.max(x_test)-np.min(x_test))/np.sqrt(np.mean(mse_list)))

    return x_test_rec,mse_list,psnr_list
