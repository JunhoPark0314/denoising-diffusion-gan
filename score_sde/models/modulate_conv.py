import torch
import numpy as np

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = torch.nn.SiLU()(x) + b
        return  x

class SynthesisGroupConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        sampling_rate,
        fdim_base = 8,
        fdim_max = 512,
        kernel_size = 3,
        padding = 1,
        act_clamp = 255,
        init_scale = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sampling_rate = sampling_rate
        self.kernel_size = kernel_size
        self.bandwidth = self.sampling_rate * (2 ** 0.1) #* (2 ** -0.9)
        self.freq_dim = int(np.clip(self.sampling_rate * fdim_base, a_min=128, a_max=fdim_max))
        self.num_freq = 16
        self.padding = padding
        self.act_clamp = act_clamp
        self.dyn_channel = w_dim
        self.init_scale = init_scale
        self.init_weight_gen('random_dynami', kernel_size, in_channels, out_channels)

    def init_weight_gen(self, freq_dist, kernel_size, in_channels, out_channels):
        magnitude = torch.randn([in_channels, self.freq_dim])
        out_linear = torch.randn([out_channels, self.freq_dim])

        if freq_dist == "random_train":
            self.basis = torch.nn.Parameter(torch.randn([self.in_channels, 3, 3, self.freq_dim]))
        elif freq_dist == "random_fixed":
            self.register_buffer("basis", torch.randn([self.in_channels, 3, 3, self.freq_dim]))
        elif freq_dist == "random_dynamic":
            self.basis = torch.nn.Parameter(torch.randn([self.in_channels, 3, 3, self.freq_dim]))
            self.affine = FullyConnectedLayer(self.dyn_channel, self.num_freq, bias=True, bias_init=1, lr_multiplier=1/self.sampling_rate, weight_init=1/self.sampling_rate)
            magnitude = torch.randn([self.num_freq, in_channels, self.freq_dim])

        elif freq_dist == "four_train":
            freqs = torch.rand([self.in_channels, self.freq_dim, 2])
            self.register_buffer("freqs", freqs)

            # Prebuild grid
            theta = torch.eye(2, 3)
            grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, kernel_size, kernel_size], align_corners=True)
            self.register_buffer("grids", grids.squeeze(0))

            phases = torch.rand([self.in_channels, 1, 1, self.freq_dim]) - 0.5
            self.phases = torch.nn.Parameter(phases)
            self.weight_bias = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1, 1]))

        elif freq_dist == "four_fixed":
            freqs = torch.rand([self.in_channels, self.freq_dim, 2])
            self.register_buffer("freqs", freqs)

            # Prebuild grid
            theta = torch.eye(2, 3)
            grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, kernel_size, kernel_size], align_corners=True)
            self.register_buffer("grids", grids.squeeze(0))

            phases = torch.rand([self.in_channels, 1, 1, self.freq_dim]) - 0.5
            self.register_buffer("phases", phases)
            self.weight_bias = torch.nn.Parameter(torch.randn([1, out_channels, in_channels, 1, 1]))

        self.magnitude = torch.nn.Parameter(magnitude * self.init_scale)
        self.out_linear = torch.nn.Parameter(out_linear)

        self.gain = float(np.sqrt(1 / (in_channels * (kernel_size ** 2))))
        self.freq_gain = float(np.sqrt(1 / (self.freq_dim * 2)))

        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))

    def get_basis(self):
        if self.freq_dist in ["random_train", "random_fixed", "random_dynamic"]:
            return self.basis

        ix = torch.einsum('hwr,ifr->ihwf', self.grids, self.freqs)
        ix = ix + self.phases
        ix = torch.sin(ix * (np.pi * 2))
        return ix
    
    def weight_gen(self, x, temb):
        ks = self.kernel_size
        basis  = self.get_basis()
        magnitude = self.magnitude.unsqueeze(0)
        out_linear = self.out_linear

        if self.freq_dist == "random_dynamic":
            s = self.affine(temb)
            # s = s * s.square().mean(dim=[1,2], keepdim=True).rsqrt()
            # magnitude = magnitude * magnitude.square().mean(dim=[1,2], keepdim=True).rsqrt()
            magnitude = torch.einsum('bkif,bik->bif', magnitude, s) / np.sqrt(self.num_freq)
            
        kernel = torch.einsum('ihwf,bif,of->boihw', basis, magnitude, out_linear) * self.freq_gain
        if hasattr(self, 'weight_bias'):		
            kernel = kernel + self.weight_bias * np.sqrt(1 / (2 * ks))
        kernel = kernel * self.gain

        return kernel.squeeze(0)

    def forward(self, x):

        kernel = self.weight_gen(x).to(dtype=x.dtype)
        batch_size = x.shape[0]
        group_size = 1
        if self.freq_dist == "random_dynamic":
            kernel = kernel.reshape(-1, self.in_channels, self.kernel_size, self.kernel_size)
            x = x.reshape(1, -1, *x.shape[2:])
            group_size = batch_size

        x = torch.nn.functional.conv2d(x, kernel, padding=self.padding, groups=group_size)
        x = x.reshape(batch_size, -1, *x.shape[2:])
        if self.bias is not None:
            x = torch.nn.SiLU()(x) + self.bias
        return x