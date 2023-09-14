import torch.nn as nn
import torch
import math
import gin

###Code from TIMM:
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Transpose(nn.Module):
    def __init__(self, axis0, axis1):
        super().__init__()
        self.axis0=axis0
        self.axis1=axis1
        
    def forward(self,x):
        return torch.transpose(x,self.axis0,self.axis1)

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, embedding_dim,learn_scale=True,scale=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = torch.nn.Parameter(data=torch.tensor(scale),requires_grad=learn_scale)

    def forward(self,x):
        pe = torch.zeros_like(x)
        position = torch.arange(0, pe.shape[1]).unsqueeze(1).unsqueeze(0)
        div_term = torch.exp((torch.arange(0, pe.shape[2], 2, dtype=torch.float) *
                             -(math.log(10000.0) / pe.shape[2])))
        pe[:,:, 0::2] = torch.sin(position.float() * div_term)
        pe[:,:, 1::2] = torch.cos(position.float() * div_term)
        return x+self.scale*pe

### Transformer implementation with ResiDual normalization. See 'ResiDual: Transformer with Dual Residual Connections' - Xie et al.
class TransformerLayer(nn.Module):
    def __init__(self, model_dim, attention_layer, ff_layer, norm_layer, 
                 norm_type='ResiDual', cross_attention_layer=None, drop_path=0, init_values=None):
        super().__init__()
        self.att_layer = attention_layer
        self.ff_layer = ff_layer
        self.norm1 = norm_layer(model_dim)
        self.norm2 = norm_layer(model_dim)
        self.norm_type = norm_type
        self.xatt_layer = cross_attention_layer
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        if cross_attention_layer is not None:
            self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.norm3 = norm_layer(model_dim)
            self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            
    def forward(self, x, 
                key_mask=None, att_mask=None, 
                att_bias=None, k_mem=None, v_mem=None, 
                key_mem_mask=None, mem_att_mask=None, mem_att_bias=None):
        
        if self.norm_type=='prenorm':
            xnorm = self.norm1(x)
            att_out, att_matrix = self.att_layer(xnorm, xnorm, xnorm, key_mask=key_mask, att_mask=att_mask, att_bias=att_bias)
            x = x + self.drop_path1(self.ls1(att_out))
            if self.xatt_layer is not None:
                xnorm = self.norm3(x)
                xatt_out, xatt_matrix = self.xatt_layer(xnorm, k_mem, v_mem, key_mask=key_mem_mask, att_mask=mem_att_mask, att_bias=mem_att_bias)
                x = x + self.drop_path3(self.ls3(xatt_out))    
            else:
                xatt_matrix = None
            x = x + self.drop_path2(self.ls2(self.ff_layer(self.norm2(x))))
            
        elif self.norm_type=='postnorm':
            att_out, att_matrix = self.att_layer(x,x,x,key_mask=key_mask, att_mask=att_mask, att_bias=att_bias)
            x = self.norm1(x+self.drop_path1(self.ls1(att_out)))
            if self.xatt_layer is not None:
                xatt_out, xatt_matrix = self.xatt_layer(x, k_mem, v_mem, key_mask=key_mem_mask, att_mask=mem_att_mask, att_bias=mem_att_bias)
                x = self.norm3(x + self.drop_path3(self.ls3(xatt_out)))
            else:
                xatt_matrix = None
            x = self.norm2(x+self.drop_path2(self.ls2(self.ff_layer(x))))

        elif self.norm_type=='ResiDual':
            #Here 2 IO are managed. x[0]: ln_out x[1]: unnormalized
            #See https://arxiv.org/pdf/2304.14802.pdf

            if not isinstance(x,tuple):
                x = (x,x)
            att_out, att_matrix = self.att_layer(x[0],x[0],x[0],key_mask=key_mask, att_mask=att_mask, att_bias=att_bias)
            x0 = self.norm1(x[0]+self.drop_path1(self.ls1(att_out)))
            x1 = x[1]+self.drop_path1(self.ls1(att_out))
            if self.xatt_layer is not None:
                xatt_out, xatt_matrix = self.xatt_layer(x0, k_mem, v_mem, key_mask=key_mem_mask, att_mask=mem_att_mask, att_bias=mem_att_bias)
                x0 = self.norm3(x0+self.drop_path3(self.ls3(xatt_out)))
                x1 = x1+self.drop_path1(self.ls3(xatt_out))
            else:
                xatt_matrix=None
            ff_out = self.ff_layer(x0)
            x0 = self.norm2(x0+self.drop_path2(self.ls2(ff_out)))
            x1 = x1+self.drop_path2(self.ls2(ff_out))
            x=(x0,x1)

        return x, att_matrix, xatt_matrix

class TransformerEncoder(nn.Module):
    def __init__(self, model_dim,
                 num_layers,
                 attention_layer, 
                 ff_layer=None,
                 ff_dim=4096,
                 norm_layer=torch.nn.LayerNorm, 
                 norm_type='ResiDual', 
                 cross_attention_layer=None, 
                 drop_path=0, 
                 init_values=None,
                 positional_encoder=None,
                 compile=True,
                 return_activations=False):
        
        super().__init__()
        if ff_layer is None:
            ff_layer =  torch.nn.Sequential(torch.nn.Linear(model_dim, ff_dim),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(ff_dim, model_dim))
        else:
            ff_layer = ff_layer()
        if cross_attention_layer is not None:
            cross_attention_layer = cross_attention_layer()
        self.encoder_layers = torch.nn.ModuleList([TransformerLayer(model_dim, 
                                                    attention_layer(), 
                                                    ff_layer, 
                                                    norm_layer, 
                                                    norm_type, 
                                                    cross_attention_layer,
                                                    drop_path,
                                                    init_values) for i in range(num_layers)])
        if positional_encoder is not None:
            self.positional_encoder = positional_encoder(model_dim)
        else:
            self.positional_encoder = positional_encoder
        if norm_type in ['ResiDual','prenorm']:
            self.final_norm = norm_layer(model_dim)
        elif norm_type == 'postnorm':
            self.final_norm = torch.nn.Identity()
        self.model_dim = model_dim
        self.compile = compile
        #self.compile = False
        self.norm_type = norm_type
        self.return_activations = return_activations
        if not self.return_activations:
            self.return_activations = []
        elif self.return_activations == 'all':
            self.return_activations = [i for i in range(num_layers)]

    def run_through_encoder(self,x):
        x, layers, padding_mask = x
        for l in layers:
            x,_,_=l(x,key_mask=padding_mask)
        return x
    
    def extract_activations(self,x):
        activations = []
        x, layers, padding_mask = x
        for l in layers:
            x,_,_=l(x,key_mask=padding_mask)
            activations.append(x[0])
        return activations

    @torch.compile
    def compiled_run_through_encoder(self,x):
        return self.run_through_encoder(x)

    @torch.compile
    def compiled_get_activations(self,x):
        return self.extract_activations(x)

    def get_activations(self,x, padding_mask):
        if self.positional_encoder is not None:
            x = self.positional_encoder(x)        
        if self.compile:
            return self.compiled_get_activations((x, self.encoder_layers, padding_mask))
        else:
            return self.extract_activations((x, self.encoder_layers, padding_mask))

    def forward(self,x, padding_mask):
        if self.positional_encoder is not None:
            x = self.positional_encoder(x)

        if self.compile:
            x = self.compiled_run_through_encoder((x,self.encoder_layers,padding_mask))
        else:
            x = self.run_through_encoder((x,self.encoder_layers,padding_mask))

        if self.norm_type == 'ResiDual':
            x = x[0] + self.final_norm(x[1])
        elif self.norm_type == 'prenorm':
            x = self.final_norm(x)
        else:
            pass
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 model_dim,
                 num_heads,
                 qk_proj_dim=None,
                 v_proj_dim=None,
                 q_bias=False,
                 k_bias=False,
                 v_bias=False,
                 out_bias=False,
                 att_drop=0,
                 proj_drop=0,
                 att_scale=None,
                 mask_value=-1e9):
        
        super().__init__()
        if qk_proj_dim is None:
            qk_proj_dim = model_dim//num_heads
        if v_proj_dim is None:
            v_proj_dim = qk_proj_dim
        
        self.qk_proj_dim = qk_proj_dim
        self.num_heads = num_heads
        self.v_proj_dim = v_proj_dim
        
        self.wq = nn.Linear(model_dim, qk_proj_dim*num_heads, bias=q_bias)
        self.wk = nn.Linear(model_dim, qk_proj_dim*num_heads, bias=k_bias)
        self.wv = nn.Linear(model_dim, v_proj_dim*num_heads, bias=v_bias)
        
        if att_scale is None:
            self.att_scale = qk_proj_dim ** -0.5
        else:
            self.att_scale = att_scale
        self.mask_value = mask_value
        
        self.att_drop = nn.Dropout(att_drop)
        self.wo = nn.Linear(v_proj_dim*num_heads, model_dim, bias=out_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, q, k, v, key_mask=None, att_mask=None, att_bias=None):
        N,Tq,C = q.shape
        N,Tk,C = k.shape
        Q = self.wq(q) #NxTxD.H
        K = self.wk(k) #NxTxD.H
        V = self.wv(v) #NxTxD.H
        
        Q = Q.view(N,Tq,self.qk_proj_dim,self.num_heads).permute(0,3,1,2).reshape(N*self.num_heads,Tq,self.qk_proj_dim) #N.HxTqxDq
        K = K.view(N,Tk,self.qk_proj_dim,self.num_heads).permute(0,3,1,2).reshape(N*self.num_heads,Tk,self.qk_proj_dim) #N.HxTkxDk
        V = V.view(N,Tk,self.v_proj_dim,self.num_heads).permute(0,3,1,2).reshape(N*self.num_heads,Tk,self.v_proj_dim) #N.HxTkxDv
        
        kv_mask = torch.zeros((N*self.num_heads,Tq,Tk), dtype=torch.bool, device=Q.device)
        if key_mask is not None:
            key_mask = torch.tile(key_mask[:,None,None,:],(1,self.num_heads,Tq,1)).reshape(N*self.num_heads,Tq,Tk).to(dtype=torch.bool)
            kv_mask += key_mask
        if att_mask is not None:
            kv_mask += att_mask
        
        att = Q @ K.transpose(-2, -1) * self.att_scale #N.HxTqxTk
        if att_bias is not None:
            att += att_bias
        att += kv_mask*self.mask_value
        att = att.softmax(dim=-1) #N.HxTqxTk
        att = self.att_drop(att) #N.HxTqxTk
        
        x = att @ V #N.HxTqxDv
        x = x.view(N,self.num_heads,Tq,self.v_proj_dim).permute(0,2,3,1).reshape(N,Tq,self.v_proj_dim*self.num_heads) #NxTqxDvxH
        O = self.wo(x)
        O = self.proj_drop(O)
        
        return O, att
