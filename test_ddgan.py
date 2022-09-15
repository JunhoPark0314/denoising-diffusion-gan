# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np

import os

import torchvision
from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch.distributed as dist
import shutil

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps // args.step_size
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, config, device):
                
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(device)
        self.num_timesteps = args.num_timesteps
        self.step_size = args.step_size
    
    def compute_alpha(self, t):
        beta = torch.cat([torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t+1).view(-1, 1, 1, 1)
        return a
    
def q_sample(coeff, x_start, t, netTrg, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
    
    a = coeff.compute_alpha(t)
      
    x_t = a.sqrt() * x_start + (1 - a).sqrt() * noise
    eps_trg = netTrg(x_t, t)
    x_0 = (x_t - (1 - a).sqrt() * eps_trg) / a.sqrt()
    
    return x_0, a, x_t, noise

def q_sample_st(coeff, x_start, t, netTrg, a_next, noised_next, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
    
    a = coeff.compute_alpha(t)
      
    # x_t = a.sqrt() * x_start + (1 - a).sqrt() * noise
    a_ts = (a / a_next)
    var_ts = ((1-a) - (1-a_next) * a_ts)
    x_t = a_ts.sqrt() * noised_next + var_ts.sqrt() * noise

    eps_trg = netTrg(x_t, t)
    x_0 = (x_t - (1 - a).sqrt() * eps_trg) / a.sqrt()
    
    return x_0

def q_sample_pairs(coeff, x_start, t, netTrg, fix=False):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    # noise = torch.randn_like(x_start)
    x_t, a, x_noised, noise = q_sample(coeff, x_start, t, netTrg)
    prev_t = (t + coeff.step_size).clip(max=coeff.num_timesteps-1)
    x_t_plus_one = q_sample_st(coeff, x_start, prev_t, netTrg, a, x_noised, noise=noise if fix else None)
    # x_t_plus_one = extract(coeff.a_s, prev_t, x_start.shape) * x_t + \
    #                extract(coeff.sigmas, prev_t, x_start.shape) * noise
    
    return x_t.detach(), x_t_plus_one.detach()

def sample_posterior(coeff, x_0, x_t, t, next_t, **kwargs):
    at = coeff.compute_alpha(t)
    at_next = coeff.compute_alpha(next_t)
    et = (x_t - x_0 * at.sqrt()) / (1 - at).sqrt()

    c1 = (
        kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    )
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    xt_next = at_next.sqrt() * x_0 + c1 * torch.randn_like(x_0) + c2 * et
    
    return xt_next

def sample_from_model(coeff, generator, n_time, x_init, T, opt, netTrg, netEnc, real_data):
    x = x_init
    x0_list = []
    with torch.no_grad():
        t = torch.full((x.size(0),), 999, dtype=torch.int64).to(x.device)
        a = coeff.compute_alpha(t)
        eps_pred = netTrg(x, t)
        x = (x - (1 - a).sqrt() * eps_pred) / a.sqrt()
        x0_list.append(x)

        for i in reversed(range(0, n_time)):
            t = torch.full((x.size(0),), i * coeff.step_size, dtype=torch.int64).to(x.device)
            x_enc = netEnc(real_data, None)
          
            # t_time = t
            # next_time = (t - coeff.step_size).clip(min=0)
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device) + x_enc
            x = generator(x, t, latent_z)
            x0_list.append(x)
            # x_new = sample_posterior(coeff, x_0, x, t)
            # x = x_new.detach()
        
    return (torch.cat(x0_list) + 1) * 0.5

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

#%%
def sample_and_test(args):
    torch.manual_seed(42)
    device = 'cuda:0'
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    
    netG = NCSNpp(args).to(device)
    g_ckpt = torch.load('./saved_info/dd_gan/{}/{}/netG_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    enc_ckpt = torch.load('./saved_info/dd_gan/{}/{}/netEnc_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    
    #loading weights from ddp in single gpu
    for key in list(g_ckpt.keys()):
        g_ckpt[key[7:]] = g_ckpt.pop(key)
    netG.load_state_dict(g_ckpt)
    netG.eval()

    import yaml
    from ddim.models.diffusion import Model as DDIM
    from ddim.models.diffusion import Encoder
    with open(args.ddim, "r") as f:
        ddim_config = dict2namespace(yaml.safe_load(f))
    netTrg = DDIM(ddim_config).to(device)
    netTrg.load_state_dict(torch.load(ddim_config.model.pretrained))
    coeff = Diffusion_Coefficients(args, ddim_config, device)
    T = get_time_schedule(args, device)

    from copy import deepcopy

    enc_config = deepcopy(ddim_config)
    enc_config.model.out_ch = args.nz
    enc_config.model.zero_out = False
    enc_config.model.temb_ch = 0
    netEnc = Encoder(enc_config).to(device)

    for key in list(enc_ckpt.keys()):
        enc_ckpt[key[7:]] = enc_ckpt.pop(key)
    netEnc.load_state_dict(enc_ckpt)
    netEnc.eval()


    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=1,
                                                                    rank=0)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = False)
    
    data_iter = iter(data_loader)
        
    iters_needed = 50000 //args.batch_size
    base_dir = 'saved_info/dd_gan/{}/{}'.format(args.dataset, args.exp)
    
    save_dir = f"{base_dir}/generated_samples"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/real_samples", exist_ok=True)
    
    if args.compute_fid:
        for i in range(iters_needed):
            with torch.no_grad():
                real_data = next(data_iter)[0]
                x_t_1 = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)
                fake_sample = sample_from_model(coeff, netG, args.num_timesteps//args.step_size, x_t_1,T,  args, netTrg, netEnc, real_data.to(device)).cpu()
                fake_sample = fake_sample[-len(real_data):]
                
                # fake_sample = to_range_0_1(fake_sample)
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j 
                    torchvision.utils.save_image(x, f'{base_dir}/generated_samples/{index}.jpg')
                    torchvision.utils.save_image(to_range_0_1(real_data[j]), f'{base_dir}/real_samples/{index}.jpg')
                print('generating batch ', i, iters_needed)
        
        paths = [save_dir, real_img_dir]
    
        kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
        
    parser.add_argument('--ddim', type=str, default=None,
                        help='Target ddim model config')
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--step_size', type=int, default=166)



   
    args = parser.parse_args()
    
    sample_and_test(args)
    
   
                