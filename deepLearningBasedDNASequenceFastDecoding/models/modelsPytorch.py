import torch
from torch import nn
import numpy as np

Nonlinearity = nn.ReLU

class SkipConnection(nn.Sequential):
    def forward(self, input):
        return input + super().forward(input)

class Sequential(nn.Sequential):
    """ Class to combine multiple models. Sequential allowing multiple inputs."""

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, x, *args, **kwargs):
        for i, module in enumerate(self):
            if i == 0:
                x = module(x, *args, **kwargs)
            else:
                x = module(*x, **kwargs)
            if not isinstance(x, tuple) and i != len(self) - 1:
                x = (x,)
        return x


class FFTransformer(nn.Module):
    def __init__(self,d,n_blocks = 4, n_heads = 8, learn_sigma = False) -> None:
        super().__init__()
        self.learn_sigma=learn_sigma
        # time embbeding
        self.d_time_emb = 16
        self.time_emb = nn.Sequential(nn.Linear(1, self.d_time_emb),Nonlinearity())

        # positional embedding
        self.d_pos_emb = 11#+32
        pos = np.arange(0,4096)
        pos_emb = [(pos//(2**i))%2 for i in range(11)]#+[pos%32 == i for i in range(32)]
        pos_emb = np.stack(pos_emb)
        pos_emb = torch.tensor(pos_emb).transpose(0,1)
        self.register_buffer('pos_emb',pos_emb)

        # build model blocks
        self.in_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(88+self.d_time_emb+self.d_pos_emb,d),
            SkipConnection(
                Nonlinearity(),
                nn.Linear(d,d),
            )
        )

        '''
        self.down1 = nn.Sequential(
            *[TransformerBlock(d, n_heads) for i in range(n_blocks)]
        )
        self.up1 = nn.Sequential(
            *[TransformerBlock(d, n_heads) for i in range(n_blocks)]
        )
        '''

        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=n_blocks)

        self.out_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(d,88)
        )

        if learn_sigma:
            self.out_sigma_block = nn.Sequential(# [B,L,P,D]
                nn.Linear(d,88)
            )

        #print(self)
        #print(f'param count: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1_000_000}M')
            
    
    def forward(self,x,t):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]
        B = x.shape[0]
        L = x.shape[1]

        t_emb = self.time_emb(t.view(-1,1)) # [B, 16]
        t_emb = t_emb.view(B,1,-1).expand(B,L,-1) # [B, L, 16]

        pos_emb = self.pos_emb[:L] # [L, 43]
        pos_emb = pos_emb.view(1,L,-1).expand(B,L,-1) # [B, L, 43]

        x = torch.cat([x,t_emb,pos_emb],dim = -1) # [B, L, D]

        x = self.in_block(x)

        '''
        x = self.down1(x) # [B, L, D]
        x = self.up1(x) # [B, L, D]
        '''
        x = self.transformer(x)

        mu = self.out_block(x)
        #out = mu
        
        if self.learn_sigma:
            sigma = self.out_sigma_block(x)
            out = torch.cat([mu,sigma],dim=1) # that's the time dim, but the diffusion model code requires to cat on dim 1
        else:
            out = mu
        
        return out

class TransformerWithSE(nn.Module):
    def __init__(self,wrapped, d, learn_se_input = False, output_se = False):
        super().__init__()
        self.wrapped = wrapped
        if learn_se_input:
            self.se_input = nn.Parameter(torch.zeros([d],dtype=torch.float))
        self.output_se = output_se

    def forward(self, x, se_input = None):
        if se_input == None:
            target_shape = x.shape[:-2]+(1,-1) # [...,t=1,d]
            se_input = self.se_input.expand(target_shape)
        x = torch.cat([x,se_input],-2) # cat time dim
        y = self.wrapped.forward(x)
        if self.output_se:
            return y[...,:-1,:], y[...,-1:,:]
        else:
            return y[...,:-1,:]


#https://www.cnblogs.com/wevolf/p/15188846.html
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy

    def get_position_angle_vec(position):
        # this part calculate the position In brackets
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # [:, 0::2] are all even subscripts, is dim_2i
    sinusoid_table[:, 0::2] = np.sin(np.pi*sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(np.pi*sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class TransformerUnet(nn.Module):
    def __init__(self,d1,d2,d3, depth1 = 2,depth2=2, n_heads = 8) -> None:
        super().__init__()

        # positional embedding
        self.d_local_pos_emb = 11#+32
        pos = np.arange(0,256)
        local_pos_emb = [(pos//(2**i))%2 for i in range(11)]#+[pos%32 == i for i in range(32)]
        local_pos_emb = np.stack(local_pos_emb)
        local_pos_emb = torch.tensor(local_pos_emb).transpose(0,1)
        self.register_buffer('local_pos_emb',local_pos_emb)

        self.d_global_pos_emb = 32
        global_pos_emb = get_sinusoid_encoding_table(400,self.d_global_pos_emb)
        self.register_buffer('global_pos_emb',global_pos_emb)

        # build model blocks
        self.in_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(88+self.d_time_emb+self.d_local_pos_emb,d1),
            SkipConnection(
                Nonlinearity(),
                nn.Linear(d1,d1),
            )
        )

        encoder_layer1 = nn.TransformerEncoderLayer(d_model=d1, nhead=n_heads,batch_first=True)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=d2, nhead=n_heads,batch_first=True)

        self.down1 = TransformerWithSE(nn.TransformerEncoder(encoder_layer1,num_layers=depth1),d1,True,True)

        self.down2 = nn.Sequential( 
            nn.Linear(d1+self.d_global_pos_emb,d2),
            TransformerWithSE(nn.TransformerEncoder(encoder_layer2,num_layers=depth2),d2,True,True)
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(d2,d3),
            Nonlinearity(),
            nn.Linear(d3,d2),
            Nonlinearity(),
        )

        self.up2 = Sequential( 
            TransformerWithSE(nn.TransformerEncoder(encoder_layer2,num_layers=depth2),d2,False,False),
            nn.Linear(d2,d1),
        )

        self.up1 = TransformerWithSE(nn.TransformerEncoder(encoder_layer1,num_layers=depth1),d1,False,False)

        self.out_block = nn.Sequential(# [B,L,P,D]
            nn.Linear(d1,88)
        )
        print(self)
        print(f'param count: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1_000_000}M')
            
    
    def forward(self,x,t):
        # x : [b, l=n_bar*32, p=88]
        # t : [b]
        B = x.shape[0]
        L = x.shape[1]
        N_BAR = L//32

        pos_emb = self.local_pos_emb[:32] # [32, 43]
        pos_emb = pos_emb.view(1,1,32,-1).expand(B,N_BAR,32,-1).reshape(B,L,-1) # [B, L, 43]

        x = torch.cat([x,pos_emb],dim = -1) # [B, L, D]

        '''
        Down path
        '''

        x = self.in_block(x) # [B, L, d1]

        # Split bars
        x = x.view(B*N_BAR,32,-1) # [B*n_bar, 32, d1]
        skip1, x = self.down1(x) # skip1: [B*n_bar, 32, d1], x: [B*n_bar, 1, d1]

        # Bar embedding
        x = x.view(B,N_BAR,-1) # [B, N_BAR, d1]
        global_pos_emb = self.global_pos_emb[:,:N_BAR] # [N_BAR, d1 + d_global_pos_emb]
        global_pos_emb = global_pos_emb.expand(B,N_BAR,-1) # [B, N_BAR, d1 + d_global_pos_emb]
        x = torch.cat([x,global_pos_emb],dim = -1) # [B, N_BAR, d1 + d_global_pos_emb]
        skip2, x = self.down2(x) # skip2: [B, n_bar, d2], x: [B, 1, d2]

        x = self.bottleneck(x) # [B, 1, d2]

        '''
        Up path
        '''

        x= self.up2(skip2,x) # [B, n_bar, d1]
        x = x.view(B*N_BAR,1,-1) # [B*n_bar, 1, d1]

        x= self.up1(skip1,se_input = x) # [B*n_bar, 32, d1]

        x = x.reshape(B,L,-1) # [B, L, d1]

        x = self.out_block(x)
        return x

class GeneticUnet(nn.Module):
    def __init__(self,downs,ups) -> None:
        super().__init__()
        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = 0
        for skip, up in zip(skips[::-1], list(self.ups)[::-1]):
            x = x + skip
            x = up(x)

        return x



def unet1():
    model = GeneticUnet(
    downs=[
        nn.Sequential(
                nn.Conv1d(51, 64, 3, padding=1),
                nn.LeakyReLU(),
            ),
        ]
    )
    return model