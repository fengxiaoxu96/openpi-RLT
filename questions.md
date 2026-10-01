# 1. VLA模型是如何定义的？

本仓库里的 VLA 是 openpi 的 **π0 / π0.5（`Pi0`）**。它把视觉、语言和机器人状态编码成 prefix，再用一个较小的 action expert 以 flow matching 生成一段动作序列（action chunk）。RLT 并没有换掉这个骨干，而是在它的 prefix embedding 上再挂一个轻量的 RL-token 编解码器；真正在机器人上做在线修正的 actor / critic 是另一套小网络，定义在第 2、4 问。

### 1.1 统一接口：观测、动作、基类

所有模型都吃同一种结构化观测。图像固定三路、分辨率 224，状态是低维向量，语言是 token id。动作是形状 `[batch, action_horizon, action_dim]` 的 chunk。

```50:142:src/openpi/models/model.py
# Data format
# {
#     "image": { "base_0_rgb": (float32|uint8)[*b, h, w, 3], ... },
#     "image_mask": { "base_0_rgb": bool[*b], ... },
#     "state": float32[*b, s],
#     "tokenized_prompt": int32[*b, l],
#     "actions": float32[*b ah ad]
# }

@struct.dataclass
class Observation(Generic[ArrayT]):
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    ...

Actions = at.Float[ArrayT, "*b ah ad"]
```

`BaseModel` 要求每个具体模型实现两件事：`compute_loss`（训练）和 `sample_actions`（推理）。`Pi0` 继承这个基类。

```262:283:src/openpi/models/model.py
class BaseModel(nnx.Module, abc.ABC):
    action_dim: int
    action_horizon: int
    max_token_len: int

    def compute_loss(...) -> at.Float[at.Array, "*b ah"]: ...
    def sample_actions(self, rng, observation, **kwargs) -> Actions: ...
```

### 1.2 π0.5 的配置：两个 Gemma expert

当前真实机器人实验用的是 `pi05=True`。和 π0 相比，代码里写明了两处结构差别：

- 机器人状态不再作为连续 suffix token，而是离散化后写进语言 token（`discrete_state_input`）。
- action expert 用 adaRMSNorm 注入 flow-matching 的时间步，而不是把时间 embedding 和动作拼在一起过 MLP。

骨干是 PaliGemma 的 Gemma 2B（hidden 2048）加一个 Gemma 300M action expert（hidden 1024）。视觉编码器是 SigLIP So400m/14。默认动作维被 pad 到 32、horizon 为 50；Agilex 真机只有 7 维，多出来的维度是 padding。

```18:55:src/openpi/models/pi0_config.py
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    discrete_state_input: bool = None
```

```79:90:src/openpi/models/gemma.py
if variant == "gemma_2b":
    return Config(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
```

```78:112:src/openpi/models/pi0.py
class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        ...
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(...)
            self.time_mlp_out = nnx.Linear(...)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            ...
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
```

Gemma 被建成**两个 expert、共享注意力**：expert 0 吃图像和语言，expert 1 吃动作（以及 π0 下的连续状态）。`adarms=[False, True]` 表示只有 action expert 接收时间条件。

### 1.3 Prefix / suffix 与注意力

`embed_prefix` 把每路图像过 SigLIP，再把语言 token（π0.5 里已经含离散化的 state）拼在后面。图像 token 之间、图像和语言之间都是全注意力（`ar_mask` 全 False）。三路 224 图像、patch 14，每路 16×16=256 个 token，三路共 768 个图像 token。RL-token 默认就压这 768 个图像 token。

```117:149:src/openpi/models/pi0.py
def embed_prefix(self, obs):
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        # image tokens attend to each other
        ar_mask += [False] * image_tokens.shape[1]
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        ar_mask += [False] * tokenized_inputs.shape[1]
    return tokens, input_mask, ar_mask
```

`embed_suffix` 把带噪动作投影进 action expert。π0.5 把时间步过 MLP，作为 adaRMS 条件；π0 则把 state 做成一个独立 token，再把时间 embedding 和动作 token 拼起来。suffix 的第一个 token 把 `ar_mask` 置 True，所以 prefix 不能回看动作，动作 token 之间除第一个外可以互相看。

```152:198:src/openpi/models/pi0.py
def embed_suffix(self, obs, noisy_actions, timestep):
    if not self.pi05:
        state_token = self.state_proj(obs.state)[:, None, :]
        ar_mask += [True]   # image/language do not attend to state or actions
    ...
    if self.pi05:
        time_emb = self.time_mlp_out(nnx.swish(self.time_mlp_in(time_emb)))
        adarms_cond = nnx.swish(time_emb)
    else:
        action_time_tokens = concat(action_tokens, time_tokens)
        action_expert_tokens = action_time_mlp(...)
    # image/language/state inputs do not attend to action tokens
    ar_mask += [True] + ([False] * (self.action_horizon - 1))
```

### 1.4 训练目标：flow matching

训练时从 Beta(1.5, 1) 采样时间 `t ∈ (0.001, 1)`，把干净动作和噪声线性混合，网络预测速度场 `u_t = noise - actions`。损失是动作维上的 MSE，返回形状 `[batch, action_horizon]`。

```201:226:src/openpi/models/pi0.py
def compute_loss(self, rng, observation, actions, *, train=False):
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    ...
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

`preprocess_observation` 在训练时对非腕部相机做随机裁剪、旋转和颜色抖动，并把图像统一到 224、数值范围 `[-1, 1]`。

推理从纯噪声 `t=1` 出发，默认 10 步欧拉积分走到 `t=0`。prefix 只跑一次并缓存 KV，后面每步只跑 suffix。

```395:417:src/openpi/models/pi0.py
def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
    # t=1 is noise and t=0 is the target distribution
    dt = -1.0 / num_steps
    ...
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

### 1.5 挂在 VLA 上的 RL-token

RL-token 不生成动作。它用可学习 query 对 VLA prefix embedding 做交叉注意力，压成少量 token，再用 decoder 把 prefix 重建回来。默认 1 个 token、2 层、内部维和输入维都是 2048（Gemma 2B 的 hidden size）。

```15:63:src/openpi/models/rl_token.py
class RLTokenConfig:
    num_rl_tokens: int = 1
    num_layers: int = 2
    embed_dim: int = 512
    input_dim: int = 2048   # VLA prefix embedding dimension (Gemma 2B hidden size)

class RLTokenEncoder(nn.Module):
    """Compresses VLA prefix embeddings [b, seq, input_dim] into RL tokens [b, num_rl_tokens, embed_dim]."""
    def __call__(self, prefix_embs, mask=None, *, train=True):
        x = self.param("q_embed", sinusoidal_pe_init, (cfg.num_rl_tokens, cfg.embed_dim))
        ...
        for _ in range(cfg.num_layers):
            x = CrossAttentionLayer(...)(x, y, train=train, mask_cross=attn_mask)
        return x
```

重建损失对 VLA embedding 做 `stop_gradient`，所以这份 MSE 只训练 RL-token，不把梯度送回 VLA。推理时线上用的 `z_rl` 就是 encoder 输出展平后的 2048 维向量（1×2048）。

```130:152:src/openpi/models/rl_token.py
def loss(self, prefix_embs, mask=None, *, train=True):
    rl_tokens = self.encode(prefix_embs, mask, train=train)
    reconstructed = self.decode(rl_tokens, target_seq_len, train=train)
    target = jax.lax.stop_gradient(prefix_embs)
    ...
    return mse, {"mse": mse}
```

固定指令任务会丢掉语言 token，只把图像 token 送给 RL-token。这是 `extract_prefix_embeddings(..., image_only=True)` 和 `compute_loss_with_prefix(..., image_only=True)` 的行为，见 `src/openpi/models/pi0.py` 第 280–313 行。

Agilex 三相机怎样映射进这三路图像，见第 2 问的 `AgilexBagImageInputs`。

---

# 2. 训练过程是如何定义的？VLA和附属策略使用怎样的数据进行训练？

训练分成两段，数据完全不同。

| 阶段 | 入口 | 被训练的参数 | 数据 |
| --- | --- | --- | --- |
| Stage 1 | `scripts/train_rlt.py` | RL-token；`rlt_alpha>0` 时再加上整网 VLA | 离线 LeRobot 示教（Agilex 网口插入） |
| Stage 2 | `rlt_online_rl` 的 `LearnerService` | 轻量 chunk actor 和 twin critic。VLA 与 RL-token 已冻结 | 机器人在线 rollout 写进 replay 的 transition |

上游 `scripts/train.py` 仍是纯 VLA 的 flow-matching 微调（例如 `pi05_agilexbag_image_delta`）。RLT 实验走的是 `scripts/train_rlt.py`。

### 2.1 Stage 1：复合模型与损失

`RLTTrainModel` 把已经创建好的 `Pi0` 放在 `self.vla` 下，再挂一个 Linen 实现的 `RLTokenModel`（经 `nnx_bridge.ToNNX` 包一层）。预训练 VLA 权重从 checkpoint 载入，并被重新挂到 `vla/` 前缀下；RL-token 随机初始化。

```131:186:scripts/train_rlt.py
class RLTTrainModel(nnx.Module):
    def compute_rlt_loss(self, rng, observation, actions, alpha, *, train=False):
        if alpha > 0.0:
            vla_per_sample_loss, prefix_embs, prefix_mask = self.vla.compute_loss_with_prefix(
                rng, observation, actions, train=train, image_only=True
            )
            vla_loss = jnp.mean(vla_per_sample_loss)
        else:
            prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(
                rng, observation, train=train, image_only=True
            )
            vla_loss = None

        prefix_embs_sg = jax.lax.stop_gradient(prefix_embs)
        rlt_loss, rlt_info = self.rlt_module(prefix_embs_sg.astype(jnp.float32), None, train=train)
        total_loss = rlt_loss
        if vla_loss is not None:
            total_loss = rlt_loss + alpha * vla_loss
        return total_loss, info
```

因此：

- `rlt_alpha == 0`：VLA 冻结，只做 prefix 前向，可训练参数过滤为 `.*rlt_module.*`。损失只有重建 MSE。
- `rlt_alpha > 0`：同一次前向同时算 flow-matching 损失和 prefix。总损失是 `L_rlt + alpha * L_vla`。重建项仍然 stop-gradient，VLA 的梯度只来自动作损失。此时 `trainable_filter = nnx.Param`，VLA 和 RL-token 一起更新。

对应配置在 `src/openpi/training/config.py`：

- `rlt_pi05_agilexbag_image_delta`：`rlt_alpha=0.0`，冻结 VLA，只训 RL-token，动作是 delta。
- `rlt_pi05_agilexbag_image_delta_joint`：`rlt_alpha=1.0`，VLA 与 RL-token 联合训练。README 里的启动命令用的就是这一份。

两者都从 `AGILEX_PI05_BASE_CKPT`（默认 `gs://openpi-assets/checkpoints/pi05_base/params`）加载，训 5000 step，batch 32。模型超参是 1 个 RL token、2 层、embed/input 都是 2048。

优化器沿用 openpi 的 AdamW + cosine decay，并对参数做 EMA（`ema_decay=0.99`）。每 `save_interval`（默认 1000）步用 Orbax 存 `TrainState`。

### 2.2 Stage 1 的数据：LeRobot 示教

Agilex 数据工厂把数据集键重打包成推理时也会出现的名字：三路相机、`observation.state`、`action`，以及任务字符串作为 prompt。

```365:430:src/openpi/training/config.py
class LerobotAgilexBagImageDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = False
    repack_transforms = RepackTransform({
        "images": {
            "global_camera": "observation.images.global_camera",
            "pikaGripperFisheyeCamera": "observation.images.pikaGripperFisheyeCamera",
            "pikaGripperDepthCamera": "observation.images.pikaGripperDepthCamera",
        },
        "state": "observation.state",
        "actions": "action",
    })
    action_sequence_keys = ("action",)
```

`AgilexBagImageInputs` 再映射到模型的三路槽位：`global_camera → base_0_rgb`，鱼眼腕部相机 → `left_wrist_0_rgb`，夹爪侧 RealSense 的 RGB → `right_wrist_0_rgb`。

```76:91:src/openpi/policies/agilexbag_image_policy.py
parsed_images = {
    "base_0_rgb": base_image,
    "left_wrist_0_rgb": left_image,
    "right_wrist_0_rgb": right_image,
}
```

DataLoader 用 LeRobot。动作序列按 `action_horizon`（这里是 50）和数据集 fps 取未来时间戳，所以每个样本的监督目标是从当前帧起的 50 步动作。

```140:145:src/openpi/training/data_loader.py
dataset = lerobot_dataset.LeRobotDataset(
    data_config.repo_id,
    delta_timestamps={
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    },
)
```

送进模型之前的变换顺序是：

1. `RepackTransform`：对齐键名。`prompt_from_task=True` 时再从 LeRobot task 填 prompt。
2. `AgilexBagImageInputs`：三相机 + state。
3. `DeltaActions`（仅 `use_delta_joint_actions=True`）：前 6 个关节减去当前 state，夹爪（第 7 维）保持绝对值。mask 是 `make_bool_mask(6, -1)`。
4. 分位数归一化（π0.5 的 `use_quantile_norm=True`）。
5. `ModelTransformFactory`：注入 prompt、图像缩到 224、PaliGemma tokenizer 把离散 state 写进语言 token、state/action pad 到 32 维。

```204:220:src/openpi/transforms.py
class DeltaActions:
    def __call__(self, data):
        state, actions = data["state"], data["actions"]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
```

数据集 id 来自环境变量 `AGILEX_LEROBOT_REPO`，默认占位符是 `your_hf_username/agilex_ethernet_lerobot`。归一化统计需要事先用 `scripts/compute_norm_stats.py` 算好，放在 `assets/<config_name>/` 下。

这条管线里**没有奖励、没有在线交互**。VLA 学的是示教动作 chunk 的速度场，RL-token 学的是把同一批图像 prefix 压紧再还原。

### 2.3 Stage 2：附属策略是 chunk actor 和 twin critic

在线阶段不再更新 VLA / RL-token。附属策略定义在 `rlt_online_rl/src/rlt_online_rl/networks.py`。

Actor 输入三项：`z_rl`（2048）、`proprio`（7 维关节状态）、`ref_chunk`（VLA 参考动作，Ethernet 配置是 10×7）。它们分别投影到 256 / 64 / 256，拼接后过 LayerNorm + GELU 的 MLP，输出整个 chunk 的高斯均值。标准差是常数 `fixed_std`，Ethernet 配置里是 `0.002`。

```54:127:rlt_online_rl/src/rlt_online_rl/networks.py
class ChunkActor:
    def _encode_inputs(self, params, z_rl, proprio, ref_chunk):
        z_feat = layer_norm(z_rl @ Wz + bz)          # 2048 -> 256
        proprio_feat = tanh(layer_norm(...))         # 7 -> 64
        ref_feat = tanh(layer_norm(...))             # chunk_len*action_dim -> 256
        return concat([z_feat, proprio_feat, ref_feat])

    def sample_action(..., deterministic=False):
        mu, std = self.actor_dist(...)               # std = fixed_std
        if deterministic:
            return mu
        return mu + std * noise
```

Critic 是两个结构相同的 Q 网络。它看 `z_rl`、`proprio` 和**被评估的 action chunk**，不看 `ref_chunk`，输出标量 Q。TD 目标用 target actor 在下一状态上采样的动作，取双 Q 的较小值，再加上 chunk 内按 `gamma` 折扣的奖励。chunk 末端还有一次 `gamma ** chunk_len` 的 bootstrap；`done=1` 时 bootstrap 为 0。

```226:248:rlt_online_rl/src/rlt_online_rl/networks.py
def build_td_target(...):
    next_action = target_actor.sample_action(..., deterministic=False)
    next_q1, next_q2 = target_critic.q_values(..., next_action)
    bootstrap = (1.0 - done) * (gamma ** rewards.shape[-1]) * minimum(next_q1, next_q2)
    return discounted_chunk_rewards(rewards, gamma) + bootstrap
```

真正用于训练的 actor 损失在 `trainer.py`，比 `networks.compute_actor_loss` 更完整：

```212:252:rlt_online_rl/src/rlt_online_rl/trainer.py
human_mask = (source_chunk == HUMAN) | (source_chunk == MIXED)
bc_target = where(human_mask, action_chunk, ref_chunk)
bc_penalty = mean(square(action_chunk - bc_target))
# 把归一化动作还原成绝对关节后，比较前 6 个关节的逐步差分
delta_penalty = mean(square(pred_step_delta - target_step_delta))
actor_loss = bc_weight * bc_penalty - q_weight * actor_q + delta_weight * delta_penalty
```

含义：

- `HUMAN` / `MIXED` 的时间步，行为克隆目标是机器人上**真正执行过的动作**。
- `BASE` / `RL` 的时间步，行为克隆目标是 VLA 的 `ref_chunk`。部署时 actor 看到的永远是 VLA 参考，所以人类接管数据教的是“怎样把 VLA 参考改成人类修正”，而不是把 `ref_chunk` 换成人类动作。
- 训练时对 `ref_chunk` 做整段 dropout（Ethernet 概率 0.5），推理时不做。
- critic 损失是两个 Q 对 TD 目标的 MSE。每个 step 都更新 critic；actor 每 `actor_update_period`（默认 2）步更新一次，并软更新 target 网络（`target_tau=0.005`）。

#### 算法是什么：off-policy 的 chunk 级 actor-critic

Stage 2 是 **off-policy** 的 actor-critic，动作单位是一整段 chunk，而不是单步。它和 SAC、TD3 同一族：双 Q、target 网络做 Polyak 平均、actor 用重参数化从高斯里采样再对 Q 求梯度。和标准 SAC 的差别是标准差固定、损失里没有熵项，另外加了行为克隆和关节增量惩罚。代码里没有 PPO 的 clip、没有 GAE、也没有对行为策略的 importance weight。

判定为 off-policy 的依据是：产生样本的行为策略，和正在被更新的 actor，不是同一个分布，而且更新时不把旧样本丢掉。

行为策略是混合的。warmup 阶段执行冻结 VLA，source 为 `BASE`。online 阶段执行当时的 actor，source 为 `RL`，而 replay 里还留着更早版本 actor 的样本。人类接管的步是 `HUMAN` 或 `MIXED`。Ethernet 默认 `sample_strategy: uniform`，`ReplayBuffer.sample` 在整段环形缓冲上均匀抽下标，warmup 的 VLA 数据和旧 actor 的数据会与当前策略的数据混在同一个 batch 里。

```427:441:rlt_online_rl/src/rlt_online_rl/replay.py
def sample(self, batch_size: int) -> ArrayDict:
    ...
    if self._sample_strategy == "uniform":
        indices = self._sample_uniform_indices(batch_size)
    return {key: value[indices].copy() for key, value in self._storage.items()}
```

Critic 的贝尔曼备份也是 off-policy 的写法。当前 Q 回归的是缓冲里存下来的 `action_chunk`，也就是当时行为策略真正执行的动作。bootstrap 用的下一步动作则从 **target actor** 重新采样，不沿用采集时的动作，也不乘行为策略的概率比。

```141:158:rlt_online_rl/src/rlt_online_rl/trainer.py
return compute_critic_loss(
    ...
    batch["action_chunk"],          # 缓冲中的历史执行动作
    batch["rewards"],
    batch["done"],
    batch["next_z_rl"],
    ...
    batch["next_ref_chunk"],
    rl_config.gamma,
    critic_rng,
)
```

```239:248:rlt_online_rl/src/rlt_online_rl/networks.py
next_action = target_actor.sample_action(          # 当前 target actor，不是采集时的策略
    target_actor_params, rng, next_z_rl, next_proprio, next_ref_chunk, deterministic=False,
)
next_q1, next_q2 = target_critic.q_values(..., next_action)
bootstrap = (1.0 - done) * (gamma ** chunk_len) * minimum(next_q1, next_q2)
```

Actor 更新同样站在 replay 的状态分布上：给定缓冲里的 `(z_rl, proprio, ref_chunk)`，从**当前** actor 重参数化采样一个新 chunk，再最大化 Q。状态来自旧策略，动作来自新策略。同一条 transition 会被反复使用。Ethernet 里每新增 1 条样本给 5 次梯度，warmup 数据攒够后还先做 20000 次更新。on-policy 算法（PPO、A2C）要求梯度对应的轨迹由当前策略当场采出，用完即弃；这里的环形缓冲容量是 200000，旧数据一直留到被新数据覆盖。

Ethernet 的权重在 `rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml`：

- warmup：`bc_weight=10`，`q_weight=0.1`，先把 actor 拉向参考/示教。
- online：`bc_weight=5`，`q_weight=0.1`。
- `delta_weight=10`，约束相邻关节增量不要偏离目标轨迹。
- `gamma=0.99`，`chunk_len=10`，`action_representation=delta_chunk`。

`delta_chunk` 的含义在 `action_representation.py`：前 6 维是相对当前 proprio 的增量，第 7 维夹爪保持绝对，然后再做 q01/q99 分位数归一化。统计来自 `stats/norm_stats_delta.json`。Learner 采样后先调用 `prepare_training_batch` 做这步变换，actor 输出再反变换回可执行的绝对关节。

Stage 2 的样本不是 LeRobot 帧，而是 `RLTTransition`：当前与下一步的 `z_rl / proprio / ref_chunk`、实际执行的 `action_chunk`、逐步 `rewards`、`done`、控制来源和 episode 元数据。这些字段怎样从机器人轨迹里切出来，见第 4 问。

---

# 3. 推理过程是如何定义的？模型的声明和数据的流动过程是怎样的？

推理也分成两条链路。Machine A 上是冻结的 VLA+RL-token，一次观测同时产出参考动作和 `z_rl`。Machine B 上是 actor，只在关键阶段把参考 chunk 改写成要执行的 chunk。

### 3.1 Machine A：模型声明

`scripts/serve_rlt_policy.py` 按训练配置重建 `RLTInferenceModel`：`config.model.create` 得到 `Pi0`，再用同一套 `RLTokenConfig` 建 encoder/decoder，然后从 checkpoint 的 `params/` 恢复权重（bfloat16）。`--shared-prefix-inference` 只影响推理，不改变权重。

```66:112:scripts/serve_rlt_policy.py
class RLTInferenceModel(nnx.Module):
    def infer(self, rng, observation):
        if self.shared_prefix_inference:
            return self._infer_shared_prefix(rng, observation)
        return self._infer_legacy(rng, observation)

    def _infer_legacy(...):
        prefix_embs, _ = self.vla.extract_prefix_embeddings(..., image_only=True)
        rl_token = self.rlt_module(prefix_embs.astype(float32), method="encode", train=False)
        actions = self.vla.sample_actions(rng, observation)
        return actions, rl_token

    def _infer_shared_prefix(...):
        prefix_cache = self.vla.prepare_prefix_for_inference(observation)
        rl_token = self.rlt_module(prefix_cache.image_prefix_out, method="encode", train=False)
        actions = self.vla.sample_actions_from_prefix_cache(rng, prefix_cache)
        return actions, rl_token
```

两条路径的输出相同。legacy 把 prefix 算两遍：一遍给 RL-token，一遍在 `sample_actions` 里建 KV cache。shared 路径只跑一次 prefix，图像 token 送给 encoder，同一份 KV cache 用于 10 步动作采样。

服务端套上与训练相同的输入/输出变换：默认 prompt、`AgilexBagImageInputs`、分位数归一化、tokenize、pad 到 32 维；输出侧先裁回 7 维，再 `AbsoluteActions` 把 delta 加回当前 state，再反归一化。所以客户端拿到的 `ref_chunk` 已经是绝对关节空间。

```241:280:scripts/serve_rlt_policy.py
def _build_single_result(self, actions, rl_token, state, infer_time):
    outputs = self._output_transform({"state": raw_state, "actions": actions})
    z_rl = rl_token.reshape(-1)                         # [2048]
    proprio = raw_state[:7]
    ref_chunk = outputs["actions"][:50, :7]
    return {"z_rl": z_rl, "proprio": proprio, "ref_chunk": ref_chunk, ...}
```

WebSocket 协议（`src/openpi/serving/websocket_policy_server.py`）：连接后先发 metadata，之后每条 msgpack 消息是一个观测字典。带 `"batch"` 键时走批量前向，并 pad 到预编译的 batch 大小（1, 2, 4, …, 16），避免在线回填特征时反复 JIT。

### 3.2 从机器人观测到 Machine A 输入

`EnvDriver` 不直接跑 VLA。它通过 `MachineAFeatureClient` 把观测字典用 msgpack 发到 `ws://MACHINE_A:8000`。

机器人侧观测由 `PikaRobotROSBridge.get_observation` 组装，键名和 Stage 1 的 repack 对齐：

```914:922:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
return {
    "state": pos[:7],                       # JointState 前 7 维
    "images": {
        "global_camera": ...,               # uint8 HWC
        "pikaGripperFisheyeCamera": ...,
        "pikaGripperDepthCamera": ...,
    },
    "prompt": task,
}
```

Machine A 返回后，`normalize_feature_payload` **不用**服务端附带的 proprio，而是用本地观测的 `state[:7]` 覆盖。`ref_chunk` 被裁成在线配置的 `chunk_len × action_dim`，Ethernet 是 10×7，即使 VLA 本身 horizon 是 50。

```176:194:rlt_online_rl/src/rlt_online_rl/inference.py
normalized["z_rl"] = payload["z_rl"]          # 必须是 2048
normalized["proprio"] = observation["state"][:proprio_dim]
normalized["ref_chunk"] = payload["ref_chunk"][:chunk_len, :action_dim]
```

### 3.3 Machine B：actor 推理

`ActorService` 是本机 HTTP（默认 `127.0.0.1:9101`）。请求体是 `z_rl / proprio / ref_chunk`。它把绝对 `ref_chunk` 变成 `delta_chunk` 并分位数归一化，再跑 `ChunkActor`。训练 rollout 按 `actor_deterministic` 采样（Ethernet 为 false，即加固定标准差的噪声）；eval 强制用均值。输出再反归一化成绝对 7 维关节目标。

```318:337:rlt_online_rl/src/rlt_online_rl/inference.py
model_ref_chunk = self._action_adapter.normalize_ref_chunk(request.ref_chunk, request.proprio)
refined_chunk = self._wrapper.infer(..., deterministic=request.deterministic)
refined_chunk = self._action_adapter.denormalize_to_abs_chunk(refined_chunk, request.proprio)
return ActorResponse(refined_chunk=refined_chunk, source=TransitionSource.RL, ...)
```

权重来自 learner 原子写入的 pickle：`runs/agilex_ethernet/actor_snapshot/actor_snapshot.pkl`。actor 进程每 0.25 秒看一次文件，版本变了就热加载。没有 snapshot 时直接把 `ref_chunk` 原样返回，source 标成 `BASE`。

是否真的调用 actor，由 `PhaseAwareActorClient` 决定。warmup、尚未进入关键阶段、或本回合锁定为 base 策略时，它不发 RPC，直接回传 VLA 参考。

```433:447:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
def infer(self, request):
    if (phase == "warmup"
        or not self._runtime_context.in_critical_phase()
        or self._runtime_context.episode_critical_policy_mode() != "actor"):
        return ActorResponse(refined_chunk=request.ref_chunk, source=TransitionSource.BASE, ...)
    return self._actor_client.infer(request)
```

### 3.4 一次控制周期的数据流

`EnvDriver.run_episode` 的内层 planner 把上面几步串起来，然后交给环境按 `chunk_exec_horizon` 逐步执行。

```741:795:rlt_online_rl/src/rlt_online_rl/inference.py
def _policy_planner(plan_observation, local_step):
    current = normalize_feature_payload(
        self._feature_provider.get_features(plan_observation), ...)
    refine = maybe_refine_chunk(
        self._actor_client,
        z_rl=current_features.z_rl,
        proprio=current_features.proprio,
        ref_chunk=current_features.ref_chunk, ...)
    return PolicyPlan(action_chunk=refine.refined_chunk, ref_chunk=ref_chunk,
                      source=refine.source, start_features=current_features, ...)
next_observation, rewards, done, info = self._execute_chunk(observation, _policy_planner)
```

真机上 `_execute_chunk` 转到 `PikaChunkEnvAdapter.execute_chunk`：以 `control_frequency_hz`（Ethernet 为 20 Hz）为周期，每拍发 `action_chunk` 的一行。人类接管时这拍不走 planner，改为记录遥操作话题上的最新 7 维命令。

可以用一条链路概括一次关键阶段的推理：

```text
ROS JointState + 3 路 Image
        │  PikaRobotROSBridge.get_observation
        ▼
{state[7], images, prompt}
        │  WebSocket msgpack  →  Machine A: serve_rlt_policy
        ▼
输入变换 → Pi0.prefix → RLTokenEncoder → z_rl[2048]
         └→ flow matching 10 步 → 反变换 → ref_chunk[10,7]
        │
        ▼
PhaseAwareActorClient
   warmup / 非关键段 / base 模式 → 直接执行 ref_chunk
   online 关键段 + actor 模式   → HTTP :9101
        │  归一化(delta_chunk) → ChunkActor → 反归一化
        ▼
绝对关节目标 → /joint_states_gripper + Gripper 话题
```

Eval 模式（`launch_actor_eval.py` → `pika_sync_ros.py --eval_actor_only`）仍走 Machine A 和 actor，但 actor 用均值，并且 `ReplayClient` 换成 `NullReplayClient`，不写 replay、不启动 learner。

---

# 4. 在训练过程中，在线强化学习和环境的互动时怎样设定的？收集的数据是怎样的？怎样保存这些数据并利用其继续进行策略训练的？

在线循环不在仿真器里。环境是真机适配器 `PikaChunkEnvAdapter`，算法侧是 `EnvDriver`。两者的边界是：环境负责复位、20 Hz 执行、人类信号和逐步轨迹；`EnvDriver` 负责向 Machine A / actor 要动作、在回合结束时把轨迹切成 transition 并交给 replay。

### 4.1 进程与阶段

Machine B 由 `launch/launch_machine_b.py` 拉起三个进程，角色分别是 `replay_manager`、`learner_service`、`actor_service`。机器人进程是 `launch/launch_robot_rollout.py`，它等待 actor 和 replay 的 `/healthz` 之后执行 `pika_sync_ros.py`。

回合阶段由 `RolloutPhaseController` 维护，并且**只在回合之间切换**，不会在一个回合中途从 warmup 切到 online。

```244:261:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
def begin_episode(self):
    self.observe_progress()
    if self._warmup_data_ready and not self._is_online_ready():
        self._status = "warmup_wait_online"
        self._wait_until_online_ready()
    elif self._warmup_data_ready:
        self._status = "online"
    else:
        self._status = "warmup_collect"
    self._episode_phase = "online" if self._status == "online" else "warmup"
```

Ethernet 配置下的门槛：

- `warmup_min_size: 600`。replay 里的 transition 少于此数时，阶段是 `warmup_collect`。actor 不控机器人，执行的是 VLA `ref_chunk`。
- 数据够了之后进入 `warmup_wait_online`，直到 `learner_status.json` 里 `ready_for_online == true`，且 actor 版本不低于 `push_actor_interval / actor_update_period`。Ethernet 是 `500 / 2 = 250`。
- `warmup_post_collect_updates: 20000`，所以 learner 在数据刚攒够后先做 20000 次更新，才把 `ready_for_online` 置真。
- 之后每个新回合是 `online`。关键阶段里 `PhaseAwareActorClient` 才会把 chunk 交给 actor。

`task_mode` 有两种：

- `critical_phase`（Ethernet 当前默认）：回合一开始 `SIGNAL_CRITICAL_STARTED=True`，从第一拍就处于关键段。
- `full_task`：先执行 VLA 参考，操作员按 `c` 后才进入关键段。关键段之前的 chunk 被标成 `drop_transition=True`，不进 replay。

```1225:1239:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
if not critical_started:
    return (..., {"drop_transition": True, "step_trace": step_trace, ...})
```

奖励不是环境自动算的。默认 `_default_reward_fn` 在操作员标记成功时，把本 chunk 最后一拍的奖励设为 1，其余为 0。失败或 done 只结束回合，不给正奖励。

```962:973:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
def _default_reward_fn(...):
    rewards = np.zeros((executed_steps,), dtype=np.float32)
    if executed_steps > 0 and success:
        rewards[-1] = 1.0
    return rewards
```

人类可以在回合中接管。`TeleopTriggerNode` 切换 `policy_enabled`；接管期间 `execute_chunk` 不再消费 actor chunk，而是从 `HumanActionRecorder` 读遥操作命令话题上的最新 7 维动作，按控制周期采样进轨迹。source 标为 `HUMAN`；一个窗口里既有策略步又有人类步则为 `MIXED`。

### 4.2 收集到的原始数据

每个控制拍在 `step_trace` 里记一条：该拍开始时的观测、实际发出的 7 维动作、该拍对应的参考动作、奖励、下一观测、是否人类控制、source、actor 参数版本、done。

`EnvDriver` 把这些拍收成 `RawEpisodeTrace`：

- `observations`：原始观测字典列表（图像 + state + prompt），供回合结束后向 Machine A 补特征。
- `steps`：`RawEpisodeStep`，含动作、奖励、source、episode/step id。
- `chunks`：每个规划窗口的起止下标。若当时已经向 Machine A 要过特征，会把 `start_z_rl / start_proprio / start_ref_chunk` 缓存在 chunk 上。
- `policy_start_steps`：人类交还控制、策略重新开始的锚点。

非关键段的 chunk 带 `drop_transition`，切 replay 时会被跳过，从而把轨迹拆成互不相连的 segment。

### 4.3 回合结束时怎样变成训练样本

原始 trace 先落盘，再切窗口，再写入 replay。eval 模式跳过这整段。

```860:865:rlt_online_rl/src/rlt_online_rl/inference.py
raw_episode_path = self._persist_raw_episode(...)
transitions, stats = self._build_episode_replay(raw_episode)
if transitions:
    self._replay_client.add_transitions(transitions)
```

`step_trace_stride: 0`（Ethernet 当前值）走 chunk 边界窗口：每个未丢弃 chunk 的起点、策略重新接管的锚点，以及终止步对齐的最后一个满窗口。`step_trace_stride > 0` 则按步长做稠密滑窗，论文设置是 2。窗口长度等于 `chunk_len`（10）。不够长的尾巴被丢掉。

每个窗口变成一条 `RLTTransition`。窗口起点和终点若没有缓存的 Machine A 特征，会再请求一次（稠密模式按 16 条一批）。因此 transition 里的 `ref_chunk` / `next_ref_chunk` 是该观测上 VLA 的参考，`action_chunk` 是这 10 拍真正执行的动作。

```99:115:rlt_online_rl/src/rlt_online_rl/replay.py
class RLTTransition:
    z_rl: np.ndarray
    proprio: np.ndarray
    ref_chunk: np.ndarray
    action_chunk: np.ndarray
    rewards: np.ndarray
    done: bool
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    next_ref_chunk: np.ndarray
    source: int                  # BASE / RL / HUMAN / MIXED
    source_chunk: np.ndarray     # 每拍一个 source，BC 目标按它选
    collection_phase: str        # warmup 或 online
    success: int
    intervention_flag: bool
    episode_id: int
    step_id: int
```

#### 在线 buffer 和离线文件各存什么

在线训练同时维护三份数据，它们不是同一块存储的三个名字。

| 存储 | 位置 | 内容 | 谁拥有 |
| --- | --- | --- | --- |
| 回合内存 `RawEpisodeTrace` | 机器人进程的堆 | 本回合尚未切窗的逐步观测、动作、奖励 | `EnvDriver`，回合结束即丢 |
| 在线 buffer `ReplayBuffer` | replay 进程的内存 | 已经切好的 `RLTTransition`，最多 `capacity` 条 | `ReplayManager` |
| 离线 journal `replay_journal.pkl` | 磁盘，Ethernet 为 `runs/agilex_ethernet/replay/replay_journal.pkl` | 历史上每一次 `add` 的追加日志，不设上限 | `ReplayManager` 在写入 buffer 的同一把锁里追加 |
| 离线原始回合 `replay/episodes/episode_XXXXXX_<ts>.pkl` | 磁盘，与 journal 同目录 | 带图像的整段 `RawEpisodeTrace` | 机器人进程直接写文件，不经过 replay 的 HTTP |

buffer 是 learner 的采样源。journal 是 buffer 的持久化日志：内容是同一批 transition，但 journal 保留全部历史，buffer 只保留最近 `capacity` 条。原始回合文件比 transition 多了图像，在线训练不读它。

#### 什么时候写入 buffer，什么时候读取 buffer

**控制环的 20 Hz 节拍不写 buffer。** 每拍只往机器人进程里的 `RawEpisodeTrace` 追加。buffer 的唯一在线写入点是回合结束：`EnvDriver` 先把原始 trace 存盘，再在内存里切出 transition，然后 `POST /extend`。

```860:865:rlt_online_rl/src/rlt_online_rl/inference.py
raw_episode_path = self._persist_raw_episode(...)      # 先写 episodes/*.pkl
transitions, stats = self._build_episode_replay(...)   # 仍用内存中的 trace 切窗
self._replay_client.add_transitions(transitions)       # HTTP，此时才进入 buffer
```

`ReplayManager.add_transitions` 在同一临界区内先 `_buffer.add`，再把同一批记录追加到 journal。eval 模式使用 `NullReplayClient`，这一步不发生，buffer 保持不动。

**读取 buffer 发生在 replay 进程收到 HTTP 请求时，不打开磁盘。** 两类读：

- `GET /stats`：返回 `size`、`capacity`、`adds_total`、`max_episode_id`。learner 用 `size` 判断是否达到 `warmup_min_size`（600），用 `adds_total` 计算还欠多少次梯度。机器人侧用 `max_episode_id + 1` 分配下一回合编号，并用 `size` 判断 warmup 数据是否已经够。
- `POST /sample`：`ReplayBuffer.sample` 在已占用的槽位上抽下标，把 numpy 行拷贝出去。learner 每个 `train_once` 在还有更新预算时调用一次，Ethernet 的 batch 是 128。抽样不删除、不改写槽位，所以同一条 transition 会被多次读出。

learner 与机器人可以并行：上一个回合的 transition 已经进 buffer 之后，learner 在下一个回合采集的同时继续 `sample`。buffer 里因此混着 warmup 的 VLA 数据、旧 actor 的数据和刚写入的新回合。

#### 什么时候写入离线文件，什么时候读取离线文件

写入有两个时刻，都在回合结束，且都早于或同步于 buffer 更新：

1. 原始回合文件：机器人进程在切窗之前调用 `save_raw_episode`。先写 `episode_*.pkl.tmp`，再 `os.replace`。切窗用的仍是内存中的 `RawEpisodeTrace`，不会把刚写好的文件读回来。
2. journal：replay 进程在 `_buffer.add` 之后立刻 `_append_journal_many`。以 `"ab"` 追加 pickle，每条 transition 一次 `pickle.dump`，然后 `flush` 和 `fsync`。journal 只增不删；buffer 后来覆盖旧槽位时，磁盘上的旧记录还在。

```408:421:rlt_online_rl/src/rlt_online_rl/replay.py
def add(self, record):
    self._storage[key][self._position] = value
    self._position = (self._position + 1) % self.capacity
    self._size = min(self._size + 1, self.capacity)
```

```601:606:rlt_online_rl/src/rlt_online_rl/replay.py
def _append_journal_many(self, records):
    with open(self._journal_path, "ab") as f:
        for record in records:
            pickle.dump(record.to_journal_record(), f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
```

在线循环里，**读离线文件只有一次**：replay 进程启动时的 `_restore_from_journal`。它按写入顺序把 journal 里的每条记录再 `_buffer.add` 一遍。若 journal 条数超过 `capacity`，环形缓冲只留下最后 `capacity` 条，但 `adds_total` 被设成 journal 的总条数。这是进程重启后的恢复，不是梯度步的数据通道。进程一直活着时，`train_once` 不再 `open` journal。

```608:623:rlt_online_rl/src/rlt_online_rl/replay.py
def _restore_from_journal(self):
    with open(self._journal_path, "rb") as f:
        while True:
            raw = pickle.load(f)
            self._buffer.add(RLTTransition.from_mapping(raw))
            restored += 1
    self._adds_total = restored
```

`episodes/*.pkl` 不参与这段恢复，learner 也不打开它。离线工具才会另开文件：`scripts/tools/inspect_replay_journal.py` 和 `scripts/offline/offline_train_from_replay.py` 直接 `pickle.load` journal；`scripts/build_replay.py` 从一份外部 episode pickle 重新生成 journal。这些脚本不在 `LearnerService.run_forever` 里。

对照起来：

| 时刻 | buffer | `replay_journal.pkl` | `episodes/*.pkl` |
| --- | --- | --- | --- |
| 回合进行中，20 Hz | 不写、不读 | 不写、不读 | 不写、不读。数据在 `RawEpisodeTrace` |
| 回合刚结束 | 尚无本回合数据 | 尚无本回合数据 | **写入**整段 trace |
| 切窗完成、`POST /extend` | **写入**本回合全部 transition | **同时追加**同样的 transition | 不再读取刚写的文件 |
| learner 有更新预算 | **读** `sample`，不修改槽位 | 不读 | 不读 |
| rollout / learner 看进度 | **读** `stats` 的 `size`、`adds_total` | 不读 | 不读 |
| replay 进程启动 | 被 journal 重放填满 | **读**一遍 | 不读 |
| 离线分析或离线再训练 | 不参与 | 脚本自行读 | 在线路径不读 |

#### buffer 的大小和更新怎样定义

Ethernet 的 `replay.capacity` 是 **200000**。这是槽位数，不是字节数。第一条 transition 到达时，按该条记录的形状分配数组，每个字段一块 `(capacity, *field_shape)` 的 numpy 空数组。`z_rl`、`ref_chunk`、`action_chunk`、`next_*` 用 float16，`proprio` 和 `rewards` 用 float32。分配之后形状固定，后续 transition 的各字段形状必须一致。

运行中有四个不同的计数：

- `capacity`：槽位上限，配置后不变。
- `size`：当前可抽样的条数，`min(累计写入次数, capacity)`。learner 的 `replay_size` 和 warmup 门槛 600 看的都是它。
- `position`：下一笔写入的下标，每次加 1 后对 `capacity` 取模。
- `adds_total`：进程生命周期内的累计写入次数，覆盖旧槽位时也继续增加。learner 用它而不是用 `size` 来计算梯度预算：`warmup_post_collect_updates`（20000）做完之后，每增加 1 条 `adds_total`，再允许 `grad_updates_per_cycle`（5）次更新。重启后 `adds_total` 等于 journal 总条数，可以大于 `size`。

未满时，新数据写在 `[0, size)`，`size` 随回合结束的批量写入增长。满了之后 `size` 停在 200000，新数据覆盖 `position` 指向的最老槽位。抽样范围始终是 `[0, size)`。Ethernet 默认 `sample_strategy: uniform`，用 `integers(0, size)` 有放回地抽 128 个下标。配置成 `stratified` 时才会按最近 online 回合、warmup、人类接管的比例分池抽样；当前 Ethernet yaml 没有改这个默认值。

buffer 没有优先级、没有单条删除，也没有“用过就弹出”。一次更新就是回合结束时的一批 `add`：要么占据新槽，要么覆盖最老槽。梯度步只读不写。journal 不跟着覆盖，所以磁盘文件可以长过 200000 条；内存里能被在线 learner 抽到的，始终是最后写进环里的那 200000 条。

### 4.4 这些数据怎样继续训练策略

`LearnerService.train_once` 的门控：

1. `replay_size < warmup_min_size` 时空转。
2. 更新预算用 `grad_updates_per_cycle`（5）跟着新增 transition 走。配置了 `warmup_post_collect_updates` 时，先把这 20000 次做完，之后每新增 1 条 transition 再给 5 次梯度。预算用尽就等待新数据。
3. 采样 batch（Ethernet 为 128）。若 `action_norm_stats_path` 存在，先把绝对 chunk 变成归一化的 `delta_chunk`。
4. 调用 jit 过的 `train_step`：先更新 critic，再按周期更新 actor 和 target。
5. warmup 用 `warmup_bc_weight / warmup_q_weight`，`global_step` 超过 warmup 更新预算后改用 `online_bc_weight / online_q_weight`。
6. 每 `push_actor_interval_steps`（500）把 actor 参数写成 snapshot；actor 服务热加载后，**下一次**关键阶段推理就用新参数。同时每 1000 step 把 actor、critic、target、优化器状态写入 `checkpoints/latest.pkl` 和 `step_N.pkl`，重启可恢复。

```527:543:rlt_online_rl/src/rlt_online_rl/trainer.py
if self._action_adapter is not None:
    batch_np = self._action_adapter.prepare_training_batch(batch_np)
bc_weight, q_weight = _resolve_actor_loss_weights(self._rl_config, progress)
self._state, raw_metrics = train_step(
    self._state, batch,
    bc_weight=bc_weight, q_weight=q_weight,
    delta_weight=self._rl_config.delta_weight, ...)
```

VLA 和 RL-token 在这个循环里不更新。在线学习改变的只有 actor（以及只存在于 learner 里的 critic）。机器人下一回合通过 snapshot 文件间接受益，不需要重载 Machine A。

#### 梯度步会不会把 4.3 的文件再读一遍

在线训练的每个 batch **不读取** `replay_journal.pkl`，也不读取 `episodes/episode_*.pkl`。`LearnerService.train_once` 调用的是 `ReplayClient.sample_batch`，对应 HTTP `POST /sample`。服务端 `ReplayManager.sample_batch` 直接在已经驻留内存的 `ReplayBuffer` 上抽下标，返回 numpy 数组。

```587:588:rlt_online_rl/src/rlt_online_rl/replay.py
def sample_batch(self, batch_size: int) -> ArrayDict:
    return self._buffer.sample(batch_size)
```

```725:726:rlt_online_rl/src/rlt_online_rl/replay.py
def sample_batch(self, batch_size: int) -> ArrayDict:
    return self._post("/sample", {"batch_size": batch_size})
```

因此运行中的数据路径是：回合结束 → HTTP 写入内存缓冲，同时追加 journal → learner 再经 HTTP 从内存缓冲抽样。journal 是这份内存缓冲的持久化副本。进程一直活着时，训练循环里没有第二次 `open(journal)`。只有 replay 进程重启时，`_restore_from_journal` 才把文件载回内存，之后抽样仍走内存。

`episodes/*.pkl` 含原始图像，体积大，learner 的输入里也没有图像键。在线训练用的是 transition 上已经算好的 `z_rl`。这些原始回合文件留给检查脚本和真机回放，不参与 `train_step`。

另有一条离线脚本会显式读文件：`rlt_online_rl/scripts/offline/offline_train_from_replay.py` 用 `--replay-path` 指向 `replay_journal.pkl`，自己 `pickle.load` 后重新训练 actor/critic。那是独立的离线入口，不是 `LearnerService.run_forever` 的在线循环。

#### 这条“先落盘、再从缓冲里反复抽”的流程仍然是 off-policy

数据写进硬盘并不改变 on-policy / off-policy 的划分。划分看的是更新所用的动作分布是否就是当前 actor 的分布。

这里的在线循环仍是 2.3 里的 off-policy actor-critic，原因和磁盘无关：

- 一次 `sample_batch` 均匀覆盖缓冲中的全部历史，包括 VLA warmup（`BASE`）、旧版本 actor（`RL`）和人类接管（`HUMAN` / `MIXED`）。机器人上正在执行的 actor 版本，可以比这条样本被采集时的 `actor_param_version` 新很多：snapshot 每 500 个梯度步才发布一次，而缓冲里的旧 transition 继续被抽到。
- Critic 拟合的是缓冲中记录的执行动作；bootstrap 动作来自 target actor。两者之间没有 importance ratio。
- 同一条已经写进 journal 的 transition 会被重复使用：warmup 后先做 20000 次更新，之后每新增一条再做 5 次梯度。样本在容量 200000 的环里保留，直到被覆盖。

on-policy 需要当前策略当场产出轨迹、用该轨迹的对数概率或 PPO 风格的比率做一次更新，然后丢弃。本仓库的在线 learner 保留混合行为策略的 transition，并在这些状态上评估另一个 actor。journal 只是让这批 off-policy 数据在进程重启后还在。

---

# 5. 在线收集数据时模型推理和机器人硬件，以及数据存储是怎样进行联通的？

三者不在一个进程里，靠 ROS 话题、一条 WebSocket 和两条本机 HTTP 接在一起。Machine A 只做冻结模型推理；Machine B 上 actor、learner、replay 分开；机器人进程同时是 ROS 节点和 `EnvDriver`。

### 5.1 启动后谁连谁

`pika_sync_ros.main` 把依赖显式接好：

```1581:1645:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
feature_provider = MachineAFeatureClient(system.env_driver.machine_a_ws_url, ...)
replay_client = ReplayClient(system.env_driver.replay_service_url, ...)   # eval 时为 NullReplayClient
base_actor_client = ActorClient(system.env_driver.actor_service_url, ...)
actor_client = PhaseAwareActorClient(base_actor_client, phase_controller, runtime_context)
env = PikaChunkEnvAdapter(system, robot, ..., phase_controller, runtime_context, ...)
driver = EnvDriver(env, feature_provider, actor_client, replay_client, ...)
```

Ethernet 地址（`configs/tasks/agilex_ethernet/online_rl.yaml`）：

| 链路 | 地址 | 载荷 |
| --- | --- | --- |
| 机器人 → Machine A | `ws://MACHINE_A_IP:8000` | 观测进，`z_rl` + `ref_chunk` 出 |
| 机器人 → actor | `http://127.0.0.1:9101/infer` | `z_rl/proprio/ref_chunk` 进，绝对 `refined_chunk` 出 |
| 机器人 → replay | `http://127.0.0.1:9102` | 回合结束时的 transition 列表 |
| learner → replay | 同一个 9102 | `sample_batch` / `stats` |
| learner → actor | 文件系统 | `actor_snapshot.pkl`，actor 每 0.25 s 轮询 |
| rollout → learner 状态 | 文件 | `metrics/learner_status.json`，用来判断能否进入 online |

actor 和 replay 都在机器人侧机器上（bind `127.0.0.1`）。跨机器的只有 Machine A 的 WebSocket。`MachineAFeatureClient` 会先打 HTTP `/healthz`，连上后再收一条 metadata，失败则按 `machine_a_retry_interval_sec` 重试。单次推理超时默认 5 秒，失败重连一次。

### 5.2 硬件侧：ROS 订阅与发布

`ROSObsBuffer` 订阅四路并按时间戳对齐：关节、全局相机、鱼眼、深度相机的 RGB。`aligned_snapshot` 取四路最新时间戳的最小值作为帧时刻，丢掉更早的消息，避免用不同时刻的图像和关节去问 VLA。

```688:692:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
self.create_subscription(JointState, joint_topic, self._on_joint, qos)
self.create_subscription(ROSImage, global_topic, self._on_global, qos)
self.create_subscription(ROSImage, fisheye_topic, self._on_fisheye, qos)
self.create_subscription(ROSImage, depth_topic, self._on_depth, qos)
```

动作下去走两个发布器：

- `ROSCommandPublisher` 往命令话题发 `sensor_msgs/JointState`：6 个关节（弧度）加 1 个夹爪距离（米）。
- `GripperCtrlStreamer` 按固定频率发 `data_msgs/Gripper`。人类接管时 `set_policy_control_active(False)` 会暂停这条流，避免策略夹爪命令和遥操作打架。

```924:937:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
def send_action(self, action7):
    q_rad = action7[:6]
    gripper_m = clip(action7[6], 0.0, max_gripper_m)
    self._cmd_node.publish_action(q_rad, gripper_m, grip_effort=1.0)
    self._gripper_streamer.set_target(gripper_m, ...)
```

`HumanActionRecorder` 订阅**同一条命令话题**，只保留最新的 7 维 `position`。接管时控制环按 20 Hz 采样这个快照，而不是把遥操作的原始事件流写进 replay。这样 replay 的时间轴和策略控制周期一致。

回合开始前，适配器把机器人插值复位到 `critical_phase_reset_action` 或 `full_task_reset_action`（7 维绝对关节），再阻塞等待操作员请求下一回合。

### 5.3 控制环里推理和硬件怎样交叠

`execute_chunk` 在一个 chunk horizon（10 拍）内循环。策略可用时，planner 在 chunk 用完或人类刚交还控制时才再问一次 Machine A 和 actor；同一份 `action_chunk` 按行发出，每拍之后重新读 ROS 观测。节拍是 `1 / control_hz`，推理耗时从睡眠里扣掉。

```1107:1144:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
if policy_enabled and (current_plan is None or plan_cursor >= len(action_chunk)):
    current_plan = policy_planner(step_observation, local_step)   # Machine A + actor
if policy_enabled and current_plan is not None:
    self._robot.send_action(self._apply_action_limits(current_plan.action_chunk[plan_cursor]))
else:
    human_action = self._sample_latest_human_action(step_observation)
    executed.append(human_action)
time.sleep(max(0, period - elapsed))
step_observations.append(self._robot.get_observation(...))
```

`_apply_action_limits` 可选地把相对上一拍的关节增量夹到 `action_delta_limits` 内。actor RPC 失败且 `safe_fallback_to_ref=true` 时，`maybe_refine_chunk` 退回 VLA 参考，source 记为 `BASE`。

键盘进程（`keyboard_toggle_teleop_record_reward_isolation.py`）不碰模型和磁盘。它调用 `ManualSignalBridge` 暴露的 ROS service，写入 `RolloutRuntimeContext` 的信号：下一回合、成功、失败、进入关键段、选 actor 或 base、遥操作接管。`execute_chunk` 在每拍开头读这些信号来决定停不停、奖不奖励。

### 5.4 数据怎样从硬件落到磁盘，再回到训练

采集过程中图像和关节只留在机器人进程的 `RawEpisodeTrace` 内存里。回合因成功、失败、done 或达到 `max_chunk_steps_per_episode` 结束后：

1. `EnvDriver._persist_raw_episode` 把整条 trace pickle 到 replay journal 同目录的 `episodes/`。路径来自 replay 服务 `stats()` 返回的 `journal_path`，所以原始数据和训练日志在同一台 Machine B 上。
2. `_build_episode_replay` 用缓存特征，缺的锚点再经 WebSocket 批量问 Machine A。这一步发生在回合结束、机器人已经停之后，不占用 20 Hz 控制环。
3. `ReplayClient.add_transitions` 把 `RLTTransition` POST 给 replay manager。manager 写入内存环形缓冲，并 append 到 `replay_journal.pkl`。
4. learner 轮询 replay 的 size 和 `adds_total`，按第 4 问的预算采样、更新 critic/actor。
5. 新 actor 写成 `actor_snapshot/actor_snapshot.pkl`，历史版本在 `actor_snapshot/history/actor_vXXXXXX.pkl`。actor 服务下一次 `/infer` 若已加载新版本，`ActorResponse.actor_param_version` 会写回逐步轨迹，于是磁盘上的原始回合可以对应到产生该动作的参数。

learner 的状态文件 `learner_status.json` 再被 rollout 读回去。`ready_for_online` 和 actor 版本一起决定下一个回合是继续执行 VLA 参考，还是把关键段交给 actor。这样，硬件采集、冻结 VLA 推理、replay 存储和 actor 更新形成闭环，而 VLA checkpoint 本身在在线阶段保持不动。

# 6. 如果我想利用该库在Lerobot框架下的so-arm机器人上应用，其中推理所用PC和连接机器人的PC不同。我该对代码怎样进行调整。
（不需要修改代码，文字说明修改的内容和步骤即可）

建议把机器分成两台，算法进程留在现有模块里，只替换机器人边界和网络地址。

- **推理 PC（有 GPU）**：Stage 1 的 VLA+RL-token 服务（现在的 Machine A），以及在线的 learner 和 actor。learner 写 snapshot、actor 读 snapshot，这两步今天是本机文件，放在同一台机器上就不用改协议。
- **机器人 PC（接着 SO-ARM）**：只跑 rollout。它用 LeRobot 的 follower 读关节和相机、发目标关节，通过已经存在的 WebSocket / HTTP 去问推理 PC。键盘和主从遥操作也放在这台机器上。

`EnvDriver`、replay 的 transition 格式、actor/critic 的损失不用重写。要改的是：数据配置和动作维、ROS 适配器换成 LeRobot、以及几处默认绑在 `127.0.0.1` 和本地磁盘上的路径。

### 6.1 现在哪些链路已经能跨机器

VLA 推理已经按远程服务写好。`scripts/serve_rlt_policy.py` 监听 `0.0.0.0:8000`，机器人侧用 `env_driver.machine_a_ws_url`（Ethernet 配置里就是 `ws://MACHINE_A_IP:8000`）发观测、收回 `z_rl` 和 `ref_chunk`。这台服务直接放到推理 PC 上即可。

actor 和 replay 的客户端也是 URL，不是写死的函数调用。`pika_sync_ros.py` 已有 `--actor_service_url` 和 `--replay_service_url`。真正跨不过去的是服务端绑定和两条文件通道。

```58:59:rlt_online_rl/src/rlt_online_rl/config.py
bind_host: str = "127.0.0.1"
port: int = 9101
```

```86:87:rlt_online_rl/src/rlt_online_rl/config.py
bind_host: str = "127.0.0.1"
port: int = 9102
```

Ethernet 和 `runtime.dual_machine.yaml` 都把 actor、replay 绑在 `127.0.0.1`。机器人 PC 访问不到。要在推理 PC 的 yaml 里改成 `0.0.0.0`，机器人 PC 的 yaml 里把两个 URL 写成 `http://推理PC的IP:9101` 和 `:9102`。Machine A 的超时（连接 5 秒、接收 5 秒）按局域网延迟加长；actor 的 `actor_request_timeout_sec` 同样加长，否则一次跨机推理超时会退回 VLA 参考。

### 6.2 跨机器时会断掉的两条本地文件

**Actor 权重。** learner 把参数写成 `actor_snapshot.pkl`，actor 服务每 0.25 秒在本机 `open` 这个文件。中间没有 HTTP 推送。

```436:446:rlt_online_rl/src/rlt_online_rl/inference.py
def _poll_snapshot_loop(self):
    if not os.path.exists(self._snapshot_path):
        ...
    with open(self._snapshot_path, "rb") as f:
        payload = pickle.load(f)
```

因此 actor 和 learner 要放在同一台推理 PC 上，继续共用本地 snapshot。若把 actor 放到机器人 PC 以减少一跳延迟，就要么用 NFS 把 `runs/.../actor_snapshot/` 挂到两边的同一路径，要么给 actor 加一个从 learner 拉参数的接口。只改 URL 不够。

**learner 是否允许进入 online。** rollout 读的是本机 `metrics/learner_status.json`，不是 HTTP。

```388:396:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
def _make_learner_status_reader(path: Path):
    def _read():
        if not path.exists():
            return {}
        with path.open("r") as f:
            return json.load(f)
```

learner 在推理 PC 上写这个文件时，机器人 PC 上看不到 `ready_for_online`，阶段会停在 `warmup_wait_online`。处理方式和 snapshot 一样：共享 `runs/<task>/metrics/`，或让 learner 提供一个状态 HTTP，把 `bind_learner_status_getter` 指过去。

**原始回合文件的路径是字符串，不是网络写入。** `EnvDriver._persist_raw_episode` 向 replay 要 `journal_path`，然后在**调用方进程的磁盘**上写 `episodes/episode_*.pkl`。replay 若在推理 PC，返回的是推理 PC 上的路径，机器人 PC 会按这个字符串在自己的磁盘上建目录。在线训练不读这份文件，transition 仍经 HTTP 进 buffer，训练能进行；带图像的原始回合会留在机器人 PC。若希望它和 journal 放在一起，把 replay 目录用 NFS 挂成两边相同的路径，或把 `save_raw_episode` 改成 POST 给 replay 进程再落盘。

### 6.3 SO-ARM 和当前 Agilex 接口差在哪里

当前机器人边界是 ROS，不是 LeRobot 的运行时。LeRobot 在这个仓库里只出现在 **Stage 1 的数据集**（`LeRobotDataset`）。在线控制读的是 `JointState` 和三路 `Image`，写出的是 7 维：6 个关节弧度加夹爪米制开合，并另发 `data_msgs/Gripper`。

```914:934:rlt_online_rl/train_deploy_alignment/pika_sync_ros.py
return {
    "state": pos[:7],
    "images": {
        "global_camera": ...,
        "pikaGripperFisheyeCamera": ...,
        "pikaGripperDepthCamera": ...,
    },
    "prompt": task,
}
# send_action: action7[:6] 为关节弧度，action7[6] 为夹爪距离
```

SO-100 / SO-101 在 LeRobot 里是 6 个飞特舵机：`shoulder_pan`、`shoulder_lift`、`elbow_flex`、`wrist_flex`、`wrist_roll`、`gripper`。运行时用 follower 的 `get_observation()` / `send_action()`，观测是电机位置和配置里的相机，动作为目标位置。主从遥操作用 leader arm 的当前位置，而不是订阅一条 ROS 命令话题。

因此要新写一个与 `PikaChunkEnvAdapter` 同接口的适配器，实现 `reset`、`execute_chunk`、`current_phase_name`。`EnvDriver` 只要求环境有这些方法。适配器内部：

- `get_observation` 把 LeRobot 的电机位置拼成 `state`，相机拼成 `images`，并带上 `prompt`。键名要和下面的数据变换一致。
- `send_action` 把一行绝对关节目标拆回 `{motor}.pos` 再 `send_action`。
- 人类接管时，按控制周期读 leader 的 6 维位置，填进 `step_trace`，source 标成 `HUMAN`。这对应现在的 `HumanActionRecorder`。
- 复位用一段插值把 follower 送到 yaml 里的起始姿态。`critical_phase_reset_action` 从 7 个数改成 6 个数。
- 键盘今天通过 ROS service 调 `RolloutRuntimeContext`。SO-ARM 这条链路可以去掉 ROS：键盘和 rollout 放在同一进程，直接调 `request_next_episode`、`mark_manual_success` 这些方法。`ManualSignalBridge` 和 `keyboard_*.py` 都依赖 `rclpy`。

π0 固定要三路图像槽位 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。SO-ARM 常见是一路外部相机加一路腕部相机。缺的那一路用全黑图，并把 `image_mask` 设为 False，让 prefix 忽略它。`AgilexBagImageInputs` 现在缺任何一路都会直接报错，SO-ARM 的输入变换要改成“有则映射、无则补掩码”。

动作维有两处把“前 6 维当手臂、第 7 维当夹爪”写死了。SO-ARM 一共 6 维，前 5 维是手臂，第 6 维是夹爪。继续沿用会把夹爪也做成相对量。

```66:70:rlt_online_rl/src/rlt_online_rl/action_representation.py
def jax_delta_to_abs_chunk(chunk_delta, state0):
    chunk_abs = chunk_delta.at[..., :6].add(state0[..., :6])
    return chunk_abs
```

```245:246:rlt_online_rl/src/rlt_online_rl/trainer.py
pred_step_delta = pred_abs_chunk[:, 1:, :6] - pred_abs_chunk[:, :-1, :6]
target_step_delta = target_abs_chunk[:, 1:, :6] - target_abs_chunk[:, :-1, :6]
```

`DeltaActions` 的 mask、`action_representation.py` 里 numpy/jax 的关节切片、`delta_penalty` 的 `:6`，以及 `pika_sync_ros.py` 里 `--action_delta_limits` 的 `nargs=7`，都要改成“前 5 维相对、夹爪绝对”。`serve_rlt_policy.py` 里的 `ACTION_DIM = 7` 只是从网络输出里裁一刀；数据变换若已经把动作裁成 6 维，客户端再按 `rl.action_dim` 裁一次，这处常量可以保持为上界。在线 yaml 必须写成 `action_dim: 6`、`proprio_dim: 6`。

`scripts/serve_rlt_policy.py` 返回的 proprio 会被 `normalize_feature_payload` 用机器人本地的 `state` 覆盖，所以 6 维状态以机器人 PC 上的 LeRobot 读数为准。

### 6.4 建议的修改步骤

**1. 在机器人 PC 上用 LeRobot 采集示教。** 用 SO-ARM follower + leader 录成 LeRobot 数据集，确认一条样本里有 `observation.state`（6 维）、`action`（6 维）和实际用到的 `observation.images.*`。语言任务写成 dataset 的 task 字符串。这一步只用 LeRobot，还用不到本仓库的在线运行时。

**2. 在推理 PC 上加一套数据变换和训练配置。** 仿照 `LerobotAgilexBagImageDataConfig` 和 `AgilexBagImageInputs`：把 SO-ARM 的相机键映射到三路模型槽位，缺的槽位补黑图和 `image_mask=False`；`output_action_dim=6`；`use_delta_joint_actions=True` 时 mask 用 `make_bool_mask(5, -1)`。在 `src/openpi/training/config.py` 注册一份 `rlt_pi05_soarm`，权重仍从 `pi05_base` 加载，`rlt_num_tokens=1`、`rlt_embed_dim=2048`、`rlt_input_dim=2048`。数据集 id 用环境变量，和现在的 `AGILEX_LEROBOT_REPO` 一样。然后跑 `scripts/compute_norm_stats.py`，再跑 `scripts/train_rlt.py`。本仓库的 LeRobot 导入是 `lerobot.common.datasets.lerobot_dataset`，采集环境和训练环境的 LeRobot 版本要对上这个导入路径。

**3. 在推理 PC 上启动冻结策略。** `serve_rlt_policy.py --config rlt_pi05_soarm --checkpoint-dir <ckpt> --port 8000`。确认单次推理返回的 `ref_chunk` 最后一维是 6。

**4. 改在线配置。** 复制 `configs/tasks/agilex_ethernet/online_rl.yaml`。`action_dim` / `proprio_dim` 改为 6，`action_norm_stats_path` 指向第 2 步的分位数统计，复位姿态改成 6 维。`delta_chunk` 可以继续用，但第 6.3 节的切片要先改成前 5 维。`chunk_len` 和 `control_frequency_hz` 按 SO-ARM 舵机能跟上的频率定，不必沿用 Ethernet 的 20 Hz 或 10 步。

**5. 把 actor、learner、replay 留在推理 PC，并放开绑定地址。** `launch_machine_b.py` 仍按 replay → learner → actor 的顺序起。`bind_host: 0.0.0.0`。snapshot、checkpoint、journal 都写在推理 PC 的 `runs/<task>/` 下。actor 与 learner 同机，热加载逻辑保持原样。

**6. 在机器人 PC 上换掉 ROS 适配器。** 新脚本做现在 `pika_sync_ros.main` 里除 ROS 以外的事：构造 `MachineAFeatureClient(ws://推理PC:8000)`、`ActorClient(http://推理PC:9101)`、`ReplayClient(http://推理PC:9102)`、`PhaseAwareActorClient`、`EnvDriver`。环境对象内部持有 LeRobot 的 SO-ARM follower。`execute_chunk` 的节拍、人类接管、成功时最后一拍奖励为 1，按 `PikaChunkEnvAdapter.execute_chunk` 的语义照搬，这样 replay 窗口和 BC 目标不用改。

**7. 打通 learner 状态和原始回合目录。** 把推理 PC 的 `runs/<task>/metrics/learner_status.json` 共享给机器人 PC，或加一个状态查询。原始回合要么 NFS 到 journal 同目录，要么接受它只留在机器人 PC。transition 本身走已有的 `POST /extend`，buffer 和 journal 仍在推理 PC 上更新。

**8. 按原来的顺序跑通。** 推理 PC 上先有 Machine A 和 Machine B。机器人 PC 上复位到起始姿态，再开始回合。warmup 仍执行 VLA 的 `ref_chunk`，replay 超过 `warmup_min_size` 且 learner 写完 warmup 更新之后，下一回合的关键段才走 actor。eval 继续用 `NullReplayClient` 和确定性 actor 均值，只是 URL 指向推理 PC。

Stage 1 的 flow matching、RL-token 重建损失、在线的 off-policy actor-critic 和 replay 环形缓冲都不用改。SO-ARM 要换的是示教数据的键和 6 维动作约定，以及机器人 PC 上那层硬件适配；跨机器要改的是 `bind_host`、三个 URL，以及 snapshot 与 `learner_status.json` 这两份今天只在本机打开的文件。
