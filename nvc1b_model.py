import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmcv.utils import get_logger
from torchvision import transforms



class NVC1B(nn.Module):
    def __init__(self):
        super(NVC1B, self).__init__()
        self.motion_estimation = MotionEstimation()
        self.motion_encoder_decoder = MotionEncoderDecoder()
        self.temporal_context_mining = TemporalContextMining()
        self.contextual_encoder_decoder = ContextualEncoderDecoder()
        self.motion_entropy_model = EntropyModel()
        self.contextual_entropy_model = EntropyModel()

    def forward(self, x_t, x_t_minus_1):
        print(f"NVC1B input shapes: x_t {x_t.shape}, x_t_minus_1 {x_t_minus_1.shape}")
        
        vs_t, vd_t = self.motion_estimation(x_t, x_t_minus_1)
        print(f"Motion estimation output shapes: vs_t {vs_t.shape}, vd_t {vd_t.shape}")
        
        vs_t_hat, vd_t_hat, m_t_hat = self.motion_encoder_decoder(vs_t, vd_t)
        print(f"Motion encoder-decoder output shapes: vs_t_hat {vs_t_hat.shape}, vd_t_hat {vd_t_hat.shape}, m_t_hat {m_t_hat.shape}")
        
        m_t_compressed = self.motion_entropy_model(m_t_hat)
        print(f"Motion entropy model output: m_t_compressed {m_t_compressed}")
        
        C_t = self.temporal_context_mining(x_t_minus_1, vs_t_hat, vd_t_hat)
        print(f"Temporal context mining output shape: C_t {C_t.shape}")
        
        x_t_hat, y_t_hat = self.contextual_encoder_decoder(x_t, C_t)
        print(f"Contextual encoder-decoder output shapes: x_t_hat {x_t_hat.shape}, y_t_hat {y_t_hat.shape}")
        
        y_t_compressed = self.contextual_entropy_model(y_t_hat, C_t)
        print(f"Contextual entropy model output: y_t_compressed {y_t_compressed}")
        
        return x_t_hat, m_t_compressed, y_t_compressed

class MotionEstimation(nn.Module):
    def __init__(self, spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'):
        super(MotionEstimation, self).__init__()
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        self.gaussian = transforms.GaussianBlur(kernel_size=5, sigma=1.0)

    def forward(self, x_t, x_t_minus_1):
        print(f"MotionEstimation input shapes: x_t {x_t.shape}, x_t_minus_1 {x_t_minus_1.shape}")
        
        xs_t, xd_t = self.decompose_frame(x_t)
        xs_t_minus_1, xd_t_minus_1 = self.decompose_frame(x_t_minus_1)
        print(f"Decomposed frame shapes: xs_t {xs_t.shape}, xd_t {xd_t.shape}")
        
        vs_t = self.spynet(xs_t, xs_t_minus_1)
        vd_t = self.spynet(xd_t, xd_t_minus_1)
        print(f"SPyNet output shapes: vs_t {vs_t.shape}, vd_t {vd_t.shape}")
        
        return vs_t, vd_t

    def decompose_frame(self, frame):
        structure = self.gaussian(frame)
        detail = frame - structure
        return structure, detail

class MotionEncoderDecoder(nn.Module):
    def __init__(self, in_channels=4, latent_channels=256):
        super(MotionEncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, vs_t, vd_t):
        print(f"MotionEncoderDecoder input shapes: vs_t {vs_t.shape}, vd_t {vd_t.shape}")
        
        v_t = torch.cat((vs_t, vd_t), dim=1)
        print(f"Concatenated input shape: v_t {v_t.shape}")
        
        m_t = self.encoder(v_t)
        print(f"Encoder output shape: m_t {m_t.shape}")
        
        m_t_hat = torch.round(m_t)
        v_t_hat = self.decoder(m_t_hat)
        print(f"Decoder output shape: v_t_hat {v_t_hat.shape}")
        
        vs_t_hat, vd_t_hat = torch.split(v_t_hat, 2, dim=1)
        print(f"Split output shapes: vs_t_hat {vs_t_hat.shape}, vd_t_hat {vd_t_hat.shape}")
        
        return vs_t_hat, vd_t_hat, m_t_hat

class TemporalContextMining(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(TemporalContextMining, self).__init__()
        self.feature_pyramid = FeaturePyramidNetwork(input_dim, hidden_dim)
        self.convlstm = ConvLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=(3,3), num_layers=1)

    def forward(self, x_t_minus_1, vs_t_hat, vd_t_hat):
        print(f"TemporalContextMining input shapes: x_t_minus_1 {x_t_minus_1.shape}, vs_t_hat {vs_t_hat.shape}, vd_t_hat {vd_t_hat.shape}")
        
        features = self.feature_pyramid(x_t_minus_1)
        print(f"Feature pyramid output shapes: level0 {features['level0'].shape}, level1 {features['level1'].shape}, level2 {features['level2'].shape}")
        
        C0_t = self.motion_compensation(features['level0'], vs_t_hat)
        C1_t = self.motion_compensation(features['level1'], vd_t_hat)
        C2_t = self.motion_compensation(features['level2'], vs_t_hat)
        print(f"Motion compensation output shapes: C0_t {C0_t.shape}, C1_t {C1_t.shape}, C2_t {C2_t.shape}")
        
        H_t, _ = self.convlstm(C0_t.unsqueeze(0))
        H_t = H_t.squeeze(0)
        print(f"ConvLSTM output shape: H_t {H_t.shape}")
        
        C_t = self.fuse_contexts(C0_t, C1_t, C2_t, H_t)
        print(f"Fused context shape: C_t {C_t.shape}")
        
        return C_t

    def motion_compensation(self, feature, motion):
        grid = self.get_grid(motion)
        return F.grid_sample(feature, grid, mode='bilinear', padding_mode='border')

    def get_grid(self, flow):
        N, C, H, W = flow.size()
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(N,1,1,1)
        yy = yy.view(1,1,H,W).repeat(N,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        grid = grid.to(flow.device)
        vgrid = grid + flow
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
        vgrid = vgrid.permute(0,2,3,1)
        return vgrid

    def fuse_contexts(self, C0_t, C1_t, C2_t, H_t):
        return torch.cat([C0_t, C1_t, C2_t, H_t], dim=1)

class ContextualEncoderDecoder(nn.Module):
    def __init__(self, in_channels=3, context_channels=256, latent_channels=256):
        super(ContextualEncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + context_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels + context_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x_t, C_t):
        print(f"ContextualEncoderDecoder input shapes: x_t {x_t.shape}, C_t {C_t.shape}")
        
        x_t_context = torch.cat([x_t, C_t], dim=1)
        print(f"Concatenated input shape: x_t_context {x_t_context.shape}")
        
        y_t = self.encoder(x_t_context)
        print(f"Encoder output shape: y_t {y_t.shape}")
        
        y_t_hat = torch.round(y_t)
        x_t_hat = self.decoder(torch.cat([y_t_hat, C_t], dim=1))
        print(f"Decoder output shape: x_t_hat {x_t_hat.shape}")
        
        return x_t_hat, y_t_hat

class EntropyModel(nn.Module):
    def __init__(self, in_channels=256):
        super(EntropyModel, self).__init__()
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=5, stride=2, padding=2),
        )
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 192, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 192, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.context_model = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, y_t_hat, C_t=None):
        print(f"EntropyModel input shapes: y_t_hat {y_t_hat.shape}, C_t {C_t.shape if C_t is not None else None}")
        
        z_t = self.hyper_encoder(y_t_hat)
        print(f"Hyper-encoder output shape: z_t {z_t.shape}")
        
        z_t_hat = torch.round(z_t)
        sigma = self.hyper_decoder(z_t_hat)
        print(f"Hyper-decoder output shape: sigma {sigma.shape}")
        
        if C_t is not None:
            mu = self.context_model(C_t)
            print(f"Context model output shape: mu {mu.shape}")
        else:
            mu = torch.zeros_like(sigma)
        
        scale = torch.exp(sigma)
        probs = self.laplace_distribution(y_t_hat, mu, scale)
        
        bits = -torch.log2(probs)
        total_bits = bits.sum()
        print(f"Total bits: {total_bits}")
        
        return total_bits

    def laplace_distribution(self, y, mu, scale):
        return 0.5 * torch.exp(-torch.abs(y - mu) / scale) / scale
    


class SPyNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        
        if isinstance(pretrained, str):
            logger = get_logger('basicsr')
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        print(f"SPyNet compute_flow input shapes: ref {ref.shape}, supp {supp.shape}")
        
        n, _, h, w = ref.size()

        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        for level in range(5):
            ref.append(F.avg_pool2d(input=ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(input=supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            flow = flow_up + self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode='border'),
                flow_up
            ], 1))
            print(f"SPyNet compute_flow level {level} output shape: flow {flow.shape}")

        return flow

    def forward(self, ref, supp):
        print(f"SPyNet forward input shapes: ref {ref.shape}, supp {supp.shape}")
        
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_up, w_up), mode='bilinear', align_corners=False)

        flow = F.interpolate(input=self.compute_flow(ref, supp), size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        print(f"SPyNet forward output shape: flow {flow.shape}")
        return flow

class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvModule(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=None)
        )

    def forward(self, tensor_input):
        print(f"SPyNetBasicModule input shape: {tensor_input.shape}")
        output = self.basic_module(tensor_input)
        print(f"SPyNetBasicModule output shape: {output.shape}")
        return output

def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    print(f"flow_warp input shapes: x {x.shape}, flow {flow.shape}")
    
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    
    print(f"flow_warp output shape: {output.shape}")
    return output

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(FeaturePyramidNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        self.lateral1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        print(f"FeaturePyramidNetwork input shape: {x.shape}")
        
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        
        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')
        
        print(f"FeaturePyramidNetwork output shapes: level0 {p1.shape}, level1 {p2.shape}, level2 {p3.shape}")
        return {'level0': p1, 'level1': p2, 'level2': p3}

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=True)

    def forward(self, input_tensor, cur_state):
        print(f"ConvLSTMCell input shape: {input_tensor.shape}")
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        print(f"ConvLSTMCell output shapes: h_next {h_next.shape}, c_next {c_next.shape}")
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        print(f"ConvLSTM input shape: {input_tensor.shape}")
        b, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        
        internal_state = []
        output = input_tensor
        for i in range(self.num_layers):
            if hidden_state[i] is None:
                h, c = (torch.zeros(b, self.hidden_dim, h, w).to(input_tensor.device),
                        torch.zeros(b, self.hidden_dim, h, w).to(input_tensor.device))
            else:
                h, c = hidden_state[i]
            output, new_c = self.cell_list[i](output, (h, c))
            internal_state.append((output, new_c))

        print(f"ConvLSTM output shape: {output.shape}")
        return output, internal_state