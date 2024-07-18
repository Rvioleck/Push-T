## 数据预处理

#### 信息含义
```python
# agent_pos, block_pos, block_angle  
self.observation_space = spaces.Box(  
    low=np.array([0, 0, 0, 0, 0], dtype=np.float64),  
    high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float64),  
    shape=(5,),  
    dtype=np.float64  
)  
  
# positional goal for agent  
self.action_space = spaces.Box(  
    low=np.array([0, 0], dtype=np.float64),  
    high=np.array([ws, ws], dtype=np.float64),  
    shape=(2,),  
    dtype=np.float64  
)
```

#### 数据scale
```python
with np.printoptions(precision=4, suppress=True, threshold=5):  
    print("obs['image'].shape:", obs['image'].shape, "float32, [0,1]")  
    print("obs['agent_pos'].shape:", obs['agent_pos'].shape, "float32, [0,512]")  
    print("action.shape: ", action.shape, "float32, [0,512]")


$> obs['image'].shape: (3, 96, 96) float32, [0,1]
$> obs['agent_pos'].shape: (2,) float32, [0,512]
$> action.shape:  (2,) float32, [0,512]
```
对于环境中的观测数据`obs`而言，数据的尺度并不一样，`image`是取值为$0\sim1$之间的$3\times96\times96$的$RGB$三通道图像数据，而`agent_pos`是取值为$0\sim512$的$2$维数据（$x$值和$y$值）

- `image`的尺度其实不需要与仿真space的长宽尺度`512`相同，图片信息的尺度压缩到长宽为$96$仍然能够很好的work

#### 数据集数据
```python
train_image_data = dataset_root['data']['img'][:]
train_data = {  
    # first two dims of state vector are agent (i.e. gripper) locations  
    'agent_pos': dataset_root['data']['state'][:, :2],  
    'action': dataset_root['data']['action'][:]  
}
```
数据集中存在25650组数据，每组数据包含：$3\times96\times96$的图片`image`数据，`agent_pos`的$2$个值$x,y$，`action`的$2$个值$x,y$

```python
episode_ends = dataset_root['meta']['episode_ends'][:]

$> [  161   279   420  ,..., 25601 25650]
```
总共$206$个数据的`episode_ends`，表示共$25650$组数据，其实是由$206$次完整的`pushT`过程组成的，其中`episode_ends[i]`表示第`i`次`pushT`过程的结束帧
根据原始数据`train_image_data`，`train_data`和`episode_ends`可以制作实际输入神经网络的训练数据

```python
indices = create_sample_indices(  
    episode_ends=episode_ends,  
    sequence_length=pred_horizon,  
    pad_before=obs_horizon - 1,  
    pad_after=action_horizon - 1)


buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx
```

`indices`的维度为`24208, 4`，其中`len(dataset)`也为$24208$，即原本的$25650$帧数据，$206$次完整的`pushT`过程被加工成了$24208$个训练数据

| buffer_start_idx | buffer_end_idx | sample_start_idx | sample_end_idx |
| ---------------- | -------------- | ---------------- | -------------- |
| 0                | 15             | 1                | 16             |
| 0                | 16             | 0                | 16             |
| 1                | 17             | 0                | 16             |
| 2                | 18             | 0                | 16             |
| ...              | ...            | ...              | ...            |
| 144              | 160            | 0                | 16             |
| 145              | 161            | 0                | 16             |
| 146              | 161            | 0                | 15             |
| 147              | 161            | 0                | 14             |
| 148              | 161            | 0                | 13             |
| 149              | 161            | 0                | 12             |
| 150              | 161            | 0                | 11             |
| 151              | 161            | 0                | 10             |
| 152              | 161            | 0                | 9              |
| 161              | 176            | 1                | 16             |
设置好了每个数据的开始和结束的相对帧与绝对帧，在采样的时候就可以根据这些帧索引，进行帧的`padding`

```python
nsample = sample_sequence(  
    train_data=self.normalized_train_data,  
    sequence_length=self.pred_horizon,  
    buffer_start_idx=buffer_start_idx,  
    buffer_end_idx=buffer_end_idx,  
    sample_start_idx=sample_start_idx,  
    sample_end_idx=sample_end_idx  
)
```

对于数据：$0, 15, 1, 16$而言，需要在开头padding$1$帧，而对于数据：$152, 161, 0, 9$而言，需要在结尾padding$7$帧
```python
if sample_start_idx > 0:  
    data[:sample_start_idx] = sample[0]  
if sample_end_idx < sequence_length:  
    data[sample_end_idx:] = sample[-1]
```
因为列表切片取不到第buffer_end_idx上的元素，所以实际上将所有padding过后的data都同一控制在长度为16

```python
sample = input_arr[buffer_start_idx:buffer_end_idx]
```

当取`dataset[idx]`时，就要经过上述的`sample_sequence`的过程，其后的`nsample`结果如下：
```python
'agent_pos' = {ndarray: (16, 2)} 
'action' = {ndarray: (16, 2)}
'image' = {ndarray: (16, 3, 96, 96)} 
```
在返回之前，还要截取`image`和`agent_pos`的可视范围，即仅返回前`obs_horizon`个元素
```python
nsample['image'] = nsample['image'][:self.obs_horizon, :]  
nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon, :]
```
这意味着，在取`dataset[0]`的时候，其实取的是时刻$0\sim15$共$16$帧`action`数据，和$0\sim1$共$2$帧`agent_pos`和`image`观测

```python
|o|o|                             observations: 2  
  |a|a|a|a|a|a|a|a|               actions executed: 8  
|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
```



## 网络结构
#### 时序嵌入模块
```python
class SinusoidalPosEmb(nn.Module):  
    def __init__(self, dim):  
        super().__init__()  
        self.dim = dim  
  
    def forward(self, x):  
        device = x.device  
        half_dim = self.dim // 2  
        emb = math.log(10000) / (half_dim - 1)  
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  
        emb = x[:, None] * emb[None, :]  
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  
        return emb
```
- 用于嵌入unet的timestep条件
#### 上下采样模块

上采样模块和下采样模块实际上就用了一个不改变通道数，仅改变特征图大小的卷积/反卷积
有两组常见的特征图减半/增倍的参数组合：
- `kernel_size=4,stride=2,padding=1`
- `kernel_size=3,stride=2,padding=1`
```python
class Downsample1d(nn.Module):  
    def __init__(self, dim):  
        super().__init__()  
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)  
  
    def forward(self, x):  
        return self.conv(x)  
  
  
class Upsample1d(nn.Module):  
    def __init__(self, dim):  
        super().__init__()  
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)  
  
    def forward(self, x):  
        return self.conv(x)
```
- 未使用pooling是因为池化操作没有可训练参数

#### 特征提取模块
```python
class Conv1dBlock(nn.Module):  
    '''  Conv1d --> GroupNorm --> Mish  '''  
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):  
        super().__init__()  
  
        self.block = nn.Sequential(  
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),  
            nn.GroupNorm(n_groups, out_channels),  
            nn.Mish(),  
        )  
  
    def forward(self, x):  
        return self.block(x)
```
-  `kernel_size,stride=1,padding=kernel_size//2`是常见的不改变特征图大小的参数方式

#### 条件残差模块
```python
class ConditionalResidualBlock1D(nn.Module):  
    def __init__(self,  
                 in_channels,  
                 out_channels,  
                 cond_dim,  
                 kernel_size=3,  
                 n_groups=8):  
        super().__init__()  
  
        self.blocks = nn.ModuleList([  
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),  
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),  
        ])  

        self.out_channels = out_channels  
        self.cond_encoder = nn.Sequential(  
            nn.Mish(),  
            nn.Linear(cond_dim, out_channels * 2),  
            nn.Unflatten(-1, (-1, 1))  
        )  
  
        # make sure dimensions compatible  
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \  
            if in_channels != out_channels else nn.Identity()  
  
    def forward(self, x, cond):  
        out = self.blocks[0](x)  
        embed = self.cond_encoder(cond)  
  
        embed = embed.reshape(  
            embed.shape[0], 2, self.out_channels, 1)  
        scale = embed[:, 0, ...]  
        bias = embed[:, 1, ...]  
        out = scale * out + bias  
  
        out = self.blocks[1](out)  
        out = out + self.residual_conv(x)  
        return out
```
- FiLM modulation https://arxiv.org/abs/1709.07871  
- 此处的条件机制，用条件信息预测和输入`x`同维度的一个`scale`和一个`bias`，再使用条件将`action`进行偏移
- 残差连接`out = out + self.residual_conv(x)`确保了引入条件机制之后不会让整体变得更差

#### Conditional U-Net
```python

def __init__(self,  
             input_dim,  
             global_cond_dim,  
             diffusion_step_embed_dim=256,  
             down_dims=[256, 512, 1024],  
             kernel_size=5,  
             n_groups=8  
             ):
    pass
    
def forward(self,  
            sample: torch.Tensor,  
            timestep: Union[torch.Tensor, float, int],  
            global_cond=None):  
    global_feature = self.diffusion_step_encoder(timesteps)  
    if global_cond is not None:  
        global_feature = torch.cat([  
            global_feature, global_cond  
        ], axis=-1)  
    x = sample  
    h = []  
    for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):  
        x = resnet(x, global_feature)  
        x = resnet2(x, global_feature)  
        h.append(x)  
        x = downsample(x)  
    for mid_module in self.mid_modules:  
        x = mid_module(x, global_feature)  
    for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):  
        x = torch.cat((x, h.pop()), dim=1)  
        x = resnet(x, global_feature)  
        x = resnet2(x, global_feature)  
        x = upsample(x)  
    x = self.final_conv(x)  # (B,C,T)  
    x = x.moveaxis(-1, -2)  # (B,T,C)
    return x
```

1. `diffusion_step_encoder`模块如下：
```python
diffusion_step_encoder = nn.Sequential(  
    SinusoidalPosEmb(diffusion_step_embed_dim),  
    nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),  
    nn.Mish(),  
    nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),  
)
```
相当于将原本的`timestep`提取特征，模块定义时，传入了`diffusion_step_embed_dim=256`，此处即相当于将原本的`timestep`转换为$256$维的条件特征

2. `global_feature = torch.cat([global_feature, global_cond], axis=-1)`直接把处理后的`timestep`的特征和传入的`global_cond`特征直接在特征维度拼接，其中`global_cond`在传入前已经处理过了，如下：
```python
image = torch.zeros((1, obs_horizon, 3, 96, 96))  
agent_pos = torch.zeros((1, obs_horizon, 2))  
# vision encoder  
image_features = nets['vision_encoder'](image.flatten(end_dim=1))  
# (2,512)  
image_features = image_features.reshape(*image.shape[:2], -1)  
# (1,2,512)  
obs = torch.cat([image_features, agent_pos], dim=-1)
```
- 其中`nets['vision_encoder']`是将所有的`batchNorm`层替换为了`groupnorm`层之后的`resnet18`
- `resnet18`将原本的`(1, obs_horizon, 3, 96, 96)`提取特征为`(obs_horizon, 512)`再`reshape`回到`(1, obs_horizon, 512)`，最后完整的`obs`就是`image_features`和`(1, obs_horizon, 2)`的 `agent_pos`进行`concat`起来的结果，实际上就是`(1, obs_horizon, 514)`
- 输入到网络的`global_cond`实际上是`obs.flatten(start_dim=1)`，也就是`(1, obs_horizon*514)`的张量
- 整体来看，假设`obs_horizon=2`时，输入到`Conditional U-Net`的条件信息其实是：`[256维timestep特征，obs1-512维image特征，obs1-2维agent-pos，obs2-512维image特征，obs2-2维agent-pos]`

3. `down_modules`的设计如下：
```python
down_modules = nn.ModuleList([])  
for ind, (dim_in, dim_out) in enumerate(in_out):  
    is_last = ind >= (len(in_out) - 1)  
    down_modules.append(nn.ModuleList([  
        ConditionalResidualBlock1D(  
            dim_in, dim_out, cond_dim=cond_dim,  
            kernel_size=kernel_size, n_groups=n_groups),  
        ConditionalResidualBlock1D(  
            dim_out, dim_out, cond_dim=cond_dim,  
            kernel_size=kernel_size, n_groups=n_groups),  
        Downsample1d(dim_out) if not is_last else nn.Identity()  
    ]))
```
- 每个`down_module`由两个`ConditionalResidualBlock`构和一个`Downsample`构成

4. `up_modules`的设计如下：
```python
for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):  
    is_last = ind >= (len(in_out) - 1)  
    up_modules.append(nn.ModuleList([  
        ConditionalResidualBlock1D(  
            dim_out * 2, dim_in, cond_dim=cond_dim,  
            kernel_size=kernel_size, n_groups=n_groups),  
        ConditionalResidualBlock1D(  
            dim_in, dim_in, cond_dim=cond_dim,  
            kernel_size=kernel_size, n_groups=n_groups),  
        Upsample1d(dim_in) if not is_last else nn.Identity()  
    ]))
```
- `up_module`的第一个`ConditionalResidualBlock`的通道输入是`dim_out * 2`，因为实际上由于`Unet`的残差连接，需要把`down_module`模块的输出同时也输入到`up_module`进行拼接，两个相同维度的数据进行拼接，输入维度就需要`*2`了（也可以直接相加，就无需`*2`）

5. `mid_module`的设计如下：
```python
self.mid_modules = nn.ModuleList([  
    ConditionalResidualBlock1D(  
        mid_dim, mid_dim, cond_dim=cond_dim,  
        kernel_size=kernel_size, n_groups=n_groups  
    ),  
    ConditionalResidualBlock1D(  
        mid_dim, mid_dim, cond_dim=cond_dim,  
        kernel_size=kernel_size, n_groups=n_groups  
    ),  
])
```
- 可见三者的设计基本一致，并且在上采样，下采样和中间的所有部分都引入了条件信息。

