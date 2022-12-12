from pathlib import Path
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)


def angle(x):
    a = np.angle(x)
    check_pos = a >= 0
    return check_pos*a + (1-check_pos)*(2*np.pi+a)

def np_sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def to_complex_tensor(x):
    # input [batch,h,w]
    batch = x.shape[0]
    m,n = x.shape[1],x.shape[2]
    x = torch.reshape(x,(batch,m,n,1))
    x = torch.cat((x, torch.zeros_like(x).to(dev)), dim=3)
    return x

def complex_mm(x,y):
    # x and y are [batch,h,w,2]
    # (a+bj) * (c+dj) = ac-bd + (ad+bc)j
    ac = x[:,:,:,0:1]*y[:,:,:,0:1]
    bd = x[:,:,:,1:]*y[:,:,:,1:]
    ad = x[:,:,:,0:1]*y[:,:,:,1:]
    bc = x[:,:,:,1:]*y[:,:,:,0:1]
    xy = torch.cat((ac-bd, ad+bc), dim=3)
    return xy

def zero_pad(x):
    # x is input signal, size (m x n x c)
    # F.pad - (c1, c2, n1, n2, m1, m2)
    #  (front, back, left, right, up, down)
    # pad around the input signal
    m,n = x.shape[0],x.shape[1]
    return F.pad(x,(0,0,n,n,m,m), mode='constant', value=0)

def zero_pad_right_down(x,m,n):
    # pad in 2 directions (right and down)
    return F.pad(x,(0,0,0,n,0,m), mode='constant', value=0)

def zero_pad_4sides(x,m,n):
    # pad in 2 directions (right and down)
    return F.pad(x,(0,0,n,n,m,m), mode='constant', value=0)

def center_crop(x):
    if len(x.shape) == 4:
        m3,n3 = x.shape[1],x.shape[2]
        m, n = m3//3, n3//3
        return x[:,m:2*m,n:2*n,0]
    elif len(x.shape) == 3:
        m3,n3 = x.shape[0],x.shape[1]
        m, n = m3//3, n3//3
        return x[m:2*m,n:2*n,0]
    elif len(x.shape) == 2:
        m3,n3 = x.shape[0],x.shape[1]
        m, n = m3//3, n3//3
        return x[m:2*m,n:2*n]

def make_real_mask(m,n):
    real_mask = torch.cat((torch.ones(m,n,1),torch.zeros(m,n,1)),dim=2)
    return real_mask
    
def make_imag_mask(m,n):
    imag_mask = torch.cat((torch.zeros(m,n,1),torch.ones(m,n,1)),dim=2)
    return imag_mask

def make_pad_mask(m,n):
    # m,n are the shape of the input image
    pad_mask = torch.ones([m,n,2])
    # pad_mask = torch.ones([corner_size,corner_size,2])
    # pad_mask = zero_pad_right_down(pad_mask,m-corner_size,n-corner_size)
    pad_mask = zero_pad(pad_mask)
    return pad_mask

def make_pad_mask_u_corner(corner_size,m,n):
    # m,n are the shape of the input image
    # pad_mask = torch.ones([m,n,2])
    pad_mask = torch.ones([corner_size,corner_size,2])
    pad_mask = zero_pad_right_down(pad_mask,m-corner_size,n-corner_size)
    pad_mask = zero_pad(pad_mask)
    return pad_mask

def make_pad_mask_u_corner_apart(corner_size,m,n):
    # m,n are the shape of the input image
    # pad_mask = torch.ones([m,n,2])
    pad_mask = torch.ones([corner_size,corner_size,2])
    pad_mask = zero_pad_right_down(pad_mask,3*m-corner_size,3*n-corner_size)
    return pad_mask

def make_pad_mask_u_center(center_size,m,n):
    # m,n are the shape of the input image
    # pad_mask = torch.ones([m,n,2])
    pad_mask = torch.ones([center_size,center_size,2])
    pad_mask = zero_pad_4sides(pad_mask,(m-center_size)//2,(n-center_size)//2)
    pad_mask = zero_pad(pad_mask)
    return pad_mask


A = lambda x : torch.fft(x,2,normalized=True)
B = lambda x : torch.fft(x,2,normalized=True)
Aconj = lambda x : torch.ifft(x,2,normalized=True)
Mag =  lambda x : to_complex_tensor(torch.norm(x,dim=3))
Mag2 = lambda x : to_complex_tensor(torch.pow(torch.norm(x,dim=3),2))


def load_u_trained(base_path,nth_iter,ku):
    disk_dir = Path(base_path)
    u = np.load(disk_dir / f"u_{nth_iter}_{ku}.npy")
    u = torch.from_numpy(u)
    return u


def prepare_u(x):
    m,n = x.shape[0],x.shape[1]
    x = torch.from_numpy(x)
    x = torch.reshape(x,(m,n,1))
    x = torch.cat((x, torch.zeros_like(x)), dim=2)
    x = F.pad(x,(0,0,n,n,m,m), mode='constant', value=0)
    return x


def init_constant_corner(c,m,n,M,N):
    # m,n is the corner size
    # M,N is the full size
    u = c*torch.ones([m,n,1])
    u = F.pad(u,(0,0,0,N-n,0,M-m), mode='constant', value=0)
    u = zero_pad(u)
    u = torch.cat((u, torch.zeros_like(u)), dim=2)
    return u

def load_batches(dataset,nth_iter,Batch):
    m,n = dataset[0].shape[0],dataset[0].shape[1]
    x_batch = torch.zeros([Batch,3*m,3*n,2])
    idx = 0
    for i in range(nth_iter*Batch,nth_iter*Batch+Batch):
        x = torch.from_numpy(np.expand_dims(dataset[i], axis=2))
        x = torch.cat((x, torch.zeros_like(x)), dim=2)
        x = zero_pad(x)
        x_batch[idx] = x
        idx += 1
    return x_batch.to(dev)


def load_batches_with_noise(dataset,nth_iter,Batch):
    m,n = dataset[0].shape[0],dataset[0].shape[1]
    x_batch = torch.zeros([Batch,3*m,3*n,2])
    idx = 0
    for i in range(nth_iter*Batch,nth_iter*Batch+Batch):
        noisy_data = dataset[i] + np.random.normal(0,0.01,dataset[i].shape)
        x = torch.from_numpy(np.expand_dims(noisy_data, axis=2))
        x = torch.cat((x, torch.zeros_like(x)), dim=2)
        x = zero_pad(x)
        x_batch[idx] = x
        idx += 1
    return x_batch.to(dev)


def load_batch_at_idx(dataset,idx):
    m,n = dataset[0].shape[0],dataset[0].shape[1]
    N_batch = len(idx)
    x_batch = torch.zeros([N_batch,3*m,3*n,2])
    for i in range(N_batch):
        x = torch.from_numpy(np.expand_dims(dataset[idx[i]], axis=2))
        x = torch.cat((x, torch.zeros_like(x)), dim=2)
        x = zero_pad(x)
        x_batch[i] = x
    return x_batch.to(dev)


def plot_dataset(dataset,nth_iter,batch_size):
    if batch_size <= 16:
        column = batch_size
        row = 1
    else:
        column = 16
        row = batch_size//16
    for r in range(row):
        fig, ax = plt.subplots(1, column,figsize=(20, 1))
        plt.gray()
        for c in range(column):
            i = r*column+c
            image = dataset[i+batch_size*nth_iter]
            title = f"{i+batch_size*nth_iter}"
            ax[c].set_title(title)
            ax[c].imshow(image)
        [axi.set_axis_off() for axi in ax.ravel()]
        plt.show()

from skimage.measure import compare_ssim,compare_psnr


def l2(x,y):
    if x.ptp() != 0:
        x_norm = (x-x.min())/(x.ptp())
    else:
        x_norm = x
    if y.ptp() != 0:
        y_norm = (y-y.min())/(y.ptp())
    else:
        y_norm = y
    loss = np.sqrt(np.mean(np.power(x_norm-y_norm, 2)))
    return loss

def compute_psnr(x, y):
    if x.ptp() != 0:
        x_norm = (x-x.min())/(x.ptp())
    else:
        x_norm = x
    if y.ptp() != 0:
        y_norm = (y-y.min())/(y.ptp())
    else:
        y_norm = y
    mse = np.mean(np.power(x_norm-y_norm,2))
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))


def SNR(image,noise_level):
    """
    Given signal and noise level, generate gaussian noise
    Return: noisy image
    """
    relu = lambda x : np.maximum(0, x)

    row,col= image.shape
    gauss = np.random.normal(0,0.1,(row,col))
    
    
    pow_signal = np.sum(np.square(image))
    pow_noise = np.sum(np.square(gauss))
    # gauss = relu(gauss)
    k = np.sqrt(pow_signal/pow_noise/np.power(10,noise_level/10))
    noise = k*gauss
    noise = relu(noise)
    pow_noise = np.sum(np.square(noise))
    snr = 10*np.log10(pow_signal/pow_noise)
    
    # SNR is frequently defined as the ratio of the signal power and the noise power
    #     plt.figure(figsize=(15,5))
    #     plt.subplot(1,3,1)
    #     plt.imshow(image)
    #     plt.subplot(1,3,2)
    #     plt.imshow(noise)
    #     plt.subplot(1,3,3)
    #     plt.imshow(image+noise)
    #     plt.show()
    
    return image+noise

def plot_recovery(x,x_train,nth_iter,batch_size):
    gt_data = load_batches(x_train,nth_iter,batch_size).cpu()
    if batch_size <= 16:
        column = batch_size
        row = 1
    else:
        column = 16
        row = batch_size//16
    for r in range(row):
        fig, ax = plt.subplots(1, column,figsize=(20, 1))
        for c in range(column):
            i = r*column+c
            image = center_crop(x[i])
            gt = center_crop(gt_data[i].numpy())
            loss_psnr = psnr(image,gt)
            title = f"${loss_psnr:.2f}$"
            ax[c].set_title(title)
            ax[c].imshow(image)
        [axi.set_axis_off() for axi in ax.ravel()]
        plt.show()


def plot_img_list(x_list,N_column=16,height_row=1):
    batch_size = len(x_list)
    if batch_size <= N_column:
        column = batch_size
        row = 1
    else:
        column = N_column
        row = np.ceil(batch_size/N_column).astype(np.int)
    fig, ax = plt.subplots(row, column,figsize=(20, height_row*row))
    plt.gray()
    for i in range(batch_size):
        if x_list[i].requires_grad:
            x = x_list[i].detach().numpy()
        else:
            x = x_list[i]
        if row == 1:
            ax[i].imshow(center_crop(x))
        else:
            ax[i//column,i%column].imshow(center_crop(x))
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()


def plot_img_with_title(imgs,titles):
    assert len(imgs) == len(titles),"len(imgs) and len(titles) don't match"
    fig, ax = plt.subplots(1, len(imgs),figsize=(20, 2))
    plt.gray()
    for i in range(len(imgs)):
        ax[i].imshow(imgs[i])
        ax[i].set_title(titles[i])
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()


def plot_test(x_test_rec,x_test,mse_list,psnr_list,N_test,u):
    N_mask = u.shape[1]

    x_test = np.squeeze(x_test)
    _, height, width = x_test.shape
    nc = 1
    
    plt.figure()
    mean_of_psnr = np.mean(psnr_list)
    psnr_of_mean = 20*np.log10((np.max(x_test)-np.min(x_test))/np.sqrt(np.mean(mse_list)))
    # plt.hist(psnr_list,bins=100,range=(0,150))
    plt.hist(psnr_list,bins=100)
    plt.title(f"PSNR {mean_of_psnr:.2f}({psnr_of_mean:.2f}) SD {np.std(psnr_list):.2f}")
    
    fig, axs = plt.subplots(1,N_mask,figsize=(20,4))
    if N_mask == 1:
        u_show = np.squeeze(u)
        plt.imshow(u_show, cmap='gray')
        plt.title(f'u [{u_show.min():.2f}, {u_show.max():.2f}]')
    else:
        for idx_mask in range(N_mask):
            u_show = np.squeeze(u)[idx_mask]
            axs[idx_mask].imshow(u_show, cmap='gray')
            axs[idx_mask].set_title(f'u [{u_show.min():.2f}, {u_show.max():.2f}]')
    plt.show()
    

    n = np.min([10,N_test])
    figset = range(0,n)
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[figset[i]].reshape(height, width))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 +n)
        plt.imshow(x_test_rec[figset[i]].reshape(height, width))
        plt.title(f'{psnr_list[i]:.2f}')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_test_rand(x_test_rec,x_test,psnr_list,N_test):

    x_test = np.squeeze(x_test)
    x_test_rec = np.squeeze(x_test_rec)
    if x_test.shape[-1] == 2:
        _, height, width, _ = x_test.shape
        x_test = LA.norm(x_test, axis=3)
        x_test_rec = LA.norm(x_test_rec, axis=3)
    else: 
        _, height, width = x_test.shape
    nc = 1

    
    plt.figure()
    mean_of_psnr = np.mean(psnr_list)
    # plt.hist(psnr_list,bins=100,range=(0,150))
    plt.hist(psnr_list,bins=100)
    plt.title(f"PSNR {mean_of_psnr:.2f} SD {np.std(psnr_list):.2f}")
    
    n = np.min([10,N_test])
    figset = range(0,n)
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[figset[i]].reshape(height, width))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 +n)
        plt.imshow(x_test_rec[figset[i]].reshape(height, width))
        plt.title(f'{psnr_list[i]:.2f}')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    

def rand_best_each(alpha,N_mask,size,N_test = 1000,N_batch_test = 32,n_trials = 10):
    
    best_psnr_list = np.zeros(N_test)
    best_mse_list = np.ones(N_test) * float('inf')
    best_x_test_rec = np.zeros([N_test,1,size,size])
    
    
    changes_per_trial = []
    for i in range(n_trials):
        
        u = uniform_mask(N_mask,size)

        x_test_rec,mse_list,psnr_list = test(N_test,N_batch_test,N_kx,alpha,u,x_test)
        plot_test(best_x_test_rec,x_test,best_mse_list,best_psnr_list,N_test,u)

        change_list = psnr_list > best_psnr_list
        print(f'num of changes {change_list.sum()}')
        changes_per_trial.append(change_list.sum())
        plt.figure()
        plt.plot(changes_per_trial)
        plt.show()

        best_psnr_list = np.maximum(best_psnr_list,psnr_list)
        best_mse_list = np.minimum(best_mse_list,mse_list)
        best_x_test_rec[change_list] = x_test_rec[change_list]

        if i>0 and i%10 ==0:
            clear_output()
    return best_psnr_list,best_x_test_rec


def rand_best_phase_each(alpha,N_mask,size,N_test = 1000,N_batch_test = 32,n_trials = 10):
    
    best_psnr_list = np.zeros(N_test)
    best_mse_list = np.ones(N_test) * float('inf')
    best_x_test_rec = np.zeros([N_test,1,size,size])
    
    
    changes_per_trial = []
    for i in range(n_trials):
        
        u = phase_mask(N_mask,size)

        x_test_rec,mse_list,psnr_list = test_complex(N_test,N_batch_test,N_kx,alpha,u,x_test)
        plot_test(best_x_test_rec,x_test,best_mse_list,best_psnr_list,N_test,u)

        change_list = psnr_list > best_psnr_list
        print(f'num of changes {change_list.sum()}')
        changes_per_trial.append(change_list.sum())
        plt.figure()
        plt.plot(changes_per_trial)
        plt.show()

        best_psnr_list = np.maximum(best_psnr_list,psnr_list)
        best_mse_list = np.minimum(best_mse_list,mse_list)
        best_x_test_rec[change_list] = x_test_rec[change_list]

        if i>0 and i%10 ==0:
            clear_output()
    return best_psnr_list,best_x_test_rec


def normal_mask(N_mask,size):
    mask = np.random.normal(loc=0, scale=1, size=(1,N_mask,1,size,size)) # mu, sigma
    return mask

def uniform_mask(N_mask,size):
    mask = np.random.uniform(low=0, high=1, size=(1,N_mask,1,size,size))
    return mask

def phase_mask(N_mask,size):
#     mask = np.random.uniform(low=0, high=2*np.pi, size=(1,N_mask,1,size,size))
    mask = np.random.uniform(low=0, high=1, size=(1,N_mask,1,size,size))
    return mask

# plot masks
def plot_mask(mask):
    N_mask = mask.shape[1]
    fig, axs = plt.subplots(1,N_mask,figsize=(15,4))
    for idx_mask in range(N_mask):
        if N_mask == 1:
            u_show = np.squeeze(mask)
            axs.imshow(u_show, cmap='gray')
            axs.set_title(f'u [{u_show.min():.2f}, {u_show.max():.2f}]')
        else:
            u_show = np.squeeze(mask)[idx_mask]
            axs[idx_mask].imshow(u_show, cmap='gray')
            axs[idx_mask].set_title(f'u [{u_show.min():.2f}, {u_show.max():.2f}]')
    plt.show()


def plot_rec_complex(x_rec,x_test,psnr_real,psnr_mag):
    n_test, _, _, height, width, _ = x_rec.shape
    x_rec = np.squeeze(x_rec)
    x_test = x_test[:n_test]
    
    x_test_mag = LA.norm(x_test, axis=3)
    x_test_real = x_test[:,:,:,0]
    x_test_imag = x_test[:,:,:,1]
    x_test_ang = np.angle(x_test_real+1j*x_test_imag)
    
    x_rec_mag = LA.norm(x_rec, axis=3)
    x_rec_real = x_rec[:,:,:,0]
    x_rec_imag = x_rec[:,:,:,1]
    x_rec_ang = np.angle(x_rec_real+1j*x_rec_imag)
    
    print('GT Mag & Ang')
    # display GT    
    n = np.min([10,n_test])
    fig, ax = plt.subplots(2, n,figsize=(n*2, 4))
    plt.gray()
    for c in range(n):
        img_mag = np.squeeze(x_test_mag[c])
        ax[0,c].imshow(img_mag)
        ax[0,c].set_title(f'{img_mag.min():.2f},{img_mag.max():.2f}')
        img_ang = np.squeeze(x_test_ang[c])
        ax[1,c].imshow(img_ang)
        ax[1,c].set_title(f'{img_ang.min():.2f},{img_ang.max():.2f}')
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()

    
    # display REC
    print('Mag,Ang,real,imag')
    psnr_real = psnr_real[:,-1]
    psnr_mag = psnr_mag[:,-1]
    
    n = np.min([10,n_test])
    fig, ax = plt.subplots(4, n,figsize=(n*2, 8))
    plt.gray()
    for c in range(n):
        img_mag = np.squeeze(x_rec_mag[c])
        ax[0,c].imshow(img_mag)
        ax[0,c].set_title(f'{psnr_mag[c]:.2f}')
        img_ang = np.squeeze(x_rec_ang[c])
        ax[1,c].imshow(img_ang)
        ax[1,c].set_title(f'{img_ang.min():.2f},{img_ang.max():.2f}')
        img_real = np.squeeze(x_rec_real[c])
        ax[2,c].imshow(img_mag)
        ax[2,c].set_title(f'{psnr_real[c]:.2f}')
        img_imag = np.squeeze(x_rec_imag[c])
        ax[3,c].imshow(img_imag)
        ax[3,c].set_title(f'{img_imag.min():.2f},{img_imag.max():.2f}')
        
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()

    
def plot_loss(loss_x,psnr_x,loss_y,n_steps,point_steps = 100, n_curves = 1000):
    plt.figure(figsize=(20,4))
    number_size = 14
    text_size = 16    
    plt.rc('xtick',labelsize=number_size)
    plt.rc('ytick',labelsize=number_size)

    plt.subplot(131)
    plt.plot(loss_x[:n_curves].T,color='silver',lw=1)
    plt.plot(np.mean(loss_x,axis=0),color='orangered',lw=3)
    plt.yscale('log')
    plt.xlabel("Number of Steps",size=text_size)
    plt.ylabel("Reconstruction loss",size=text_size)
    plt.subplot(132)
    plt.plot(loss_y[:n_curves].T,color='silver',lw=1)
    plt.plot(np.mean(loss_y,axis=0),color='orangered',lw=3)
    plt.yscale('log')
    plt.xlabel("Number of Steps",size=text_size)
    plt.ylabel("Measurement loss",size=text_size)
    plt.subplot(133)
    plt.plot(psnr_x[:n_curves].T,color='silver',lw=1)
    avg_psnr_x = np.mean(psnr_x,axis=0)
    x_list = np.arange(n_steps)[point_steps-1::point_steps]
    y_list = avg_psnr_x[point_steps-1::point_steps]    
    plt.plot(avg_psnr_x,color='orangered',lw=3)
    plt.scatter(x_list,y_list,color='red', zorder=3)
    for x,y in zip(x_list,y_list):
        plt.text(x-20,y+5,f'{y:.0f}',color='red',fontsize=number_size)
    plt.xlabel("Number of Steps",size=text_size)
    plt.ylabel("Reconstruction PSNR (dB)",size=text_size)
    plt.show()

    
def make_complex_data_1phase(x_test):
    print(f'before x.shape {x_test.shape}')
    x_test_mag = np.expand_dims(x_test, axis=-1)
    x_test_ang = np.zeros_like(x_test_mag)
    x_test_phase = np.exp(1j*x_test_ang)
    x_test_complex = x_test_mag*x_test_phase
    x_test = np.concatenate((np.real(x_test_complex), np.imag(x_test_complex)), axis=-1)
    print(f'after x.shape {x_test.shape}')
    return x_test

def make_complex_data_rand_phase(x_test,seed=1000):
    print(f'before x.shape {x_test.shape}')
    x_test_mag = np.expand_dims(x_test, axis=-1)
    np.random.seed(seed)
    x_test_ang = np.random.uniform(0,2*np.pi,x_test_mag.shape)
    x_test_phase = np.exp(1j*x_test_ang)
    x_test_complex = x_test_mag*x_test_phase
    x_test = np.concatenate((np.real(x_test_complex), np.imag(x_test_complex)), axis=-1)
    print(f'after x.shape {x_test.shape}')
    return x_test