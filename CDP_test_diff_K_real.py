from IPython.display import clear_output
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim,compare_psnr, compare_mse
from utils import compute_psnr, plot_test
from tqdm import tqdm


def test_diff_k(mask,alpha,x_test,n_test,n_batch,n_steps,rec_save_step = 50):
    torch.cuda.set_device(0)

    n_masks = mask.shape[1]
    x_test = np.expand_dims(x_test, axis=3)
    _, height, width, nc = x_test.shape

    x_test = x_test[:n_test,:,:,:].reshape(-1,nc,height,width)
    x_test_rec = np.zeros([n_steps//rec_save_step,*x_test.shape])
    n_iter = int(np.ceil(n_test/n_batch))
    eps_tensor = torch.cuda.FloatTensor([1e-15])
    epoch_idx = np.arange(n_test)

    # image loss and measurement loss
    loss_x = np.zeros([n_test,n_steps])
    loss_y = np.zeros([n_test,n_steps])
    psnr_x = np.zeros([n_test,n_steps])

    for iters in tqdm(range(n_iter)):
    # for iters in range(n_iter):
        x = x_test[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:]
        x_gt = torch.cuda.FloatTensor(x).view(-1, 1, nc, height, width).cuda()
        mask_k = torch.cuda.FloatTensor(mask).view(-1,n_masks,nc,height,width)

        # masked signal z = mask * x
        z = x_gt * mask_k
        z_complex = F.pad(z.unsqueeze(5), (0,1), mode="constant") # pad last dim on the right
        Fz = torch.fft(z_complex, 2, normalized=True)
        # measurement y = |Fz|
        y = torch.norm(Fz, dim=5)

        x_est = x_test_rec[0,epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:]
        x_est = torch.cuda.FloatTensor(x_est.reshape(-1,1,nc,height,width)).cuda()
        
        for k in range(n_steps):
            # z_est = x_est * mask_k  # would fail without eps_tensor
            z_est = x_est * mask_k + eps_tensor
            z_est_complex = F.pad(z_est.unsqueeze(5), (0,1), mode="constant")
            Fz_est = torch.fft(z_est_complex,2, normalized=True)
            y_est = torch.norm(Fz_est,dim=5)
            # angle Fz
            Fz_est_phase = Fz_est / (y_est.unsqueeze(5) + eps_tensor)

            # update x
            x_grad = mask_k * torch.ifft( Fz_est - torch.mul(Fz_est_phase, y.unsqueeze(5)), 2, normalized=True )[:,:,:,:,:,0]
            x_grad = torch.sum(x_grad,dim=1)
            x_est = x_est - alpha * x_grad.view(x_est.shape)
            x_est = torch.clamp(x_est, 0, 1)

            x_est_np = x_est.cpu().detach().numpy().reshape(-1,nc,height,width)
            y_np = y.cpu().detach().numpy().reshape(-1,n_masks,height,width)
            y_est_np = y_est.cpu().detach().numpy().reshape(-1,n_masks,height,width)
            
            # loss_x is image reconstruction loss, loss_y is the measurement loss (MSE)
            loss_x[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],k] = np.array([compare_mse(x1,x2) for x1,x2 in zip(x,x_est_np)])
            psnr_x[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],k] = np.array([compute_psnr(x1,x2) for x1,x2 in zip(x,x_est_np)])
            loss_y[epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],k] = np.array([compare_mse(y1,y2) for y1,y2 in zip(y_np,y_est_np)])
            
            if (k+1)%rec_save_step == 0:
                x_test_rec[k//rec_save_step,epoch_idx[iters*n_batch:np.min([(iters+1)*n_batch,n_test])],:,:,:] = x_est.cpu().detach().numpy().reshape(-1,nc,height,width)

    return loss_x,psnr_x,loss_y,x_test_rec