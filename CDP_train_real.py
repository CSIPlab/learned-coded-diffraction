from IPython.display import clear_output
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr

from pathlib import Path
from dataset import *
from utils import np_sigmoid
from CDP_test_real import test


def train(n_masks,lr_u,alpha,x_train,x_valid,n_train,n_batch,n_steps,n_epoch,dataset,test_id=0):
    device_id = 0
    torch.cuda.set_device(device_id)

    logit = lambda x: np.log(x/(1-x))
    x_train = np.expand_dims(x_train, axis=3)
    N_valid = 128
    N_batch_valid = 32
    # m,n are height and width of every image
    _, height, width, nc = x_train.shape

    sigmoid = nn.Sigmoid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    output_base_dir = './output/'
    output_dir = f"{output_base_dir}{n_masks}masks_{alpha}_{dataset}_{n_train}samples_{n_steps}steps_{lr_u}_id{test_id}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # run it again restores the previous status
    u_iter = len(list(output_dir.glob('u_*.npy')))

    x_train = x_train[:n_train,:,:,:].reshape(-1,nc,height,width)


    if u_iter == 0:
        theta = np.random.uniform(0, 1, size=(n_masks,nc,height,width))
    else:
        u = np.load(output_dir / f'u_{u_iter}.npy')
        theta = logit(u)
        

    N_iter = np.int(np.ceil(n_train/np.float(n_batch)))
    idx = np.arange(x_train.shape[0])
    l2_loss_all_epochs = []
    psnr_train_all_epochs = []
    psnr_test_all_epochs = []


    torch.autograd.set_detect_anomaly(True)
    eps_tensor = torch.cuda.FloatTensor([1e-15])    
    pi_tensor = torch.cuda.FloatTensor([np.pi])

    best_epoch = 0
    best_loss = float('inf')
    for epoch in range (u_iter,n_epoch+u_iter):
        if epoch % 10 == 9: clear_output()
        print(epoch)
        x_train_rec = np.zeros_like(x_train) # reconstruction of x_train

        loss_epoch = []
        epoch_idx = idx
        np.random.shuffle(epoch_idx)

        for iters in range(N_iter):

            x = x_train[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_train])],:,:,:]
            x_gt = torch.cuda.FloatTensor(x).view(-1, 1, nc, height, width).cuda()
            theta_k = torch.autograd.Variable(torch.cuda.FloatTensor(theta).view(-1,n_masks,nc,height,width),requires_grad=True)
            optimizer = torch.optim.Adam([theta_k], lr=lr_u)
            optimizer.zero_grad()

            # z = x * u, multiplicative masks
            uk = sigmoid(theta_k)
            z = x_gt * uk
            dummy_zeros = torch.zeros_like(z).cuda()
            z_complex = torch.cat((z.unsqueeze(5), dummy_zeros.unsqueeze(5)), 5)

            Fz = torch.fft(z_complex, 2, normalized=True)
            # y = |F(x*u)| = |Fz|
            y = torch.norm(Fz, dim=5)
            y_dual = torch.cat((y.unsqueeze(5), y.unsqueeze(5)), 5)

            x_est = x_train_rec[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_train])],:,:,:]
            x_est = torch.cuda.FloatTensor(x_est.reshape(-1,1,nc,height,width)).cuda()
            
            
            loss_pr=[]
            meas_loss_pr=[]
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
                x_grad = uk * x_grad_complex[:,:,:,:,:, 0]
                x_grad = torch.sum(x_grad,dim=1)
                
                x_est = x_est - alpha * x_grad.view(x_est.shape)
                x_est = torch.clamp(x_est, 0, 1)
                
                loss_pr.append(np.mean((x-x_est.cpu().detach().numpy())**2))
                meas_loss_pr.append(height*width*np.mean((y.cpu().detach().numpy().reshape(-1,height,width)-
                    np.abs(np.fft.fft2(z_est.cpu().detach().numpy().reshape(-1,height,width), norm="ortho")))**2))
            
            x_train_rec[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_train])],:,:,:] = x_est.cpu().detach().numpy().reshape(-1,nc,height,width)


            # update u
            loss_u = (x_gt - x_est).pow(2).mean() * height * width
            loss_epoch.append(loss_u.item())
            loss_u.backward()
            optimizer.step()


        # save u at every epoch_epoch
        theta = theta_k.cpu().detach().numpy()
        u = np_sigmoid(theta)
        u_iter = u_iter + 1
        np.save(output_dir / f'u_{u_iter}', u)

        # mean loss over all iters
        mean_loss = np.mean(loss_epoch)
        l2_loss_all_epochs.append(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = u_iter
        print(f'best so far {best_epoch}' )

        
        mse = np.mean((x_train_rec-x_train)**2)
        psnr = 20*np.log10((np.max(x_train)-np.min(x_train))/np.sqrt(mse))
        print(f'psnr {psnr}')
        train_psnr_list = [compare_psnr(x_train[i,0,:,:],x_train_rec[i,0,:,:]) for i in range(n_train)]
        
        
        x_test_rec,_,test_psnr_list = test(u,alpha,x_valid,N_valid,N_batch_valid,n_steps)
        psnr_train_all_epochs.append(np.mean(train_psnr_list))
        psnr_test_all_epochs.append(np.mean(test_psnr_list))


        # plot loss and psnrs
        plt.figure(figsize=(15,4))
        plt.subplot(121)
        plt.semilogy(np.array(l2_loss_all_epochs).flatten())
        plt.title(f'loss per epoch {l2_loss_all_epochs[-1]}')
        plt.subplot(122)
        plt.plot(psnr_train_all_epochs,label='psnr train')
        plt.plot(psnr_test_all_epochs,label='psnr test')
        plt.title(f'psnr train, validate: {psnr_train_all_epochs[-1]:.2f}, {psnr_test_all_epochs[-1]:.2f}')
        plt.legend()
        plt.show()


        # plot masks
        fig, axs = plt.subplots(1,n_masks,figsize=(15,4))
        for idx_mask in range(n_masks):
            if n_masks == 1:
                u_show = np.squeeze(u)
                axs.imshow(u_show, cmap='gray')
                axs.set_title(f'u [{u_show.min():.2f}, {u_show.max():.2f}]')
            else:
                u_show = np.squeeze(u)[idx_mask]
                axs[idx_mask].imshow(u_show, cmap='gray')
                axs[idx_mask].set_title(f'u [{u_show.min():.2f}, {u_show.max():.2f}]')
        plt.show()

        
        # plot GT and reconstructions
        n = np.min([100,n_train])
        figset = range(0,n)
        plt.figure(figsize=(n*2, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_train[figset[i]].reshape(height, width))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(2, n, i + 1 +n)
            plt.imshow(x_train_rec[figset[i]].reshape(height, width))
            plt.title(f'{train_psnr_list[i]:.2f}')
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    print(f'best_epoch {best_epoch}' )
    return 