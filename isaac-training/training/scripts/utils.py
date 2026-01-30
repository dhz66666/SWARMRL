# import torch
# import torch.nn as nn
# import wandb
# import numpy as np
# from typing import Iterable, Union
# from tensordict.tensordict import TensorDict
# from omni_drones.utils.torchrl import RenderCallback
# from torchrl.envs.utils import ExplorationType, set_exploration_type

# class ValueNorm(nn.Module):
#     def __init__(
#         self,
#         input_shape: Union[int, Iterable],
#         beta=0.995,
#         epsilon=1e-5,
#     ) -> None:
#         super().__init__()

#         self.input_shape = (
#             torch.Size(input_shape)
#             if isinstance(input_shape, Iterable)
#             else torch.Size((input_shape,))
#         )
#         self.epsilon = epsilon
#         self.beta = beta

#         self.running_mean: torch.Tensor
#         self.running_mean_sq: torch.Tensor
#         self.debiasing_term: torch.Tensor
#         self.register_buffer("running_mean", torch.zeros(input_shape))
#         self.register_buffer("running_mean_sq", torch.zeros(input_shape))
#         self.register_buffer("debiasing_term", torch.tensor(0.0))

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.running_mean.zero_()
#         self.running_mean_sq.zero_()
#         self.debiasing_term.zero_()

#     def running_mean_var(self):
#         debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
#         debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
#             min=self.epsilon
#         )
#         debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
#         return debiased_mean, debiased_var

#     @torch.no_grad()
#     def update(self, input_vector: torch.Tensor):
#         assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
#         dim = tuple(range(input_vector.dim() - len(self.input_shape)))
#         batch_mean = input_vector.mean(dim=dim)
#         batch_sq_mean = (input_vector**2).mean(dim=dim)

#         weight = self.beta

#         self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
#         self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
#         self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

#     def normalize(self, input_vector: torch.Tensor):
#         assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
#         mean, var = self.running_mean_var()
#         out = (input_vector - mean) / torch.sqrt(var)
#         return out

#     def denormalize(self, input_vector: torch.Tensor):
#         assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
#         mean, var = self.running_mean_var()
#         out = input_vector * torch.sqrt(var) + mean
#         return out

# def make_mlp(num_units):
#     layers = []
#     for n in num_units:
#         layers.append(nn.LazyLinear(n))
#         layers.append(nn.LeakyReLU())
#         layers.append(nn.LayerNorm(n))
#     return nn.Sequential(*layers)

# class IndependentNormal(torch.distributions.Independent):
#     arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
#     def __init__(self, loc, scale, validate_args=None):
#         scale = torch.clamp_min(scale, 1e-6)
#         base_dist = torch.distributions.Normal(loc, scale)
#         super().__init__(base_dist, 1, validate_args=validate_args)

# class IndependentBeta(torch.distributions.Independent):
#     arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

#     def __init__(self, alpha, beta, validate_args=None):
#         beta_dist = torch.distributions.Beta(alpha, beta)
#         super().__init__(beta_dist, 1, validate_args=validate_args)

# class Actor(nn.Module):
#     def __init__(self, action_dim: int) -> None:
#         super().__init__()
#         self.actor_mean = nn.LazyLinear(action_dim)
#         self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
#     def forward(self, features: torch.Tensor):
#         loc = self.actor_mean(features)
#         scale = torch.exp(self.actor_std).expand_as(loc)
#         return loc, scale

# class BetaActor(nn.Module):
#     def __init__(self, action_dim: int) -> None:
#         super().__init__()
#         self.alpha_layer = nn.LazyLinear(action_dim)
#         self.beta_layer = nn.LazyLinear(action_dim)
#         self.alpha_softplus = nn.Softplus()
#         self.beta_softplus = nn.Softplus()
    
#     def forward(self, features: torch.Tensor):
#         alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
#         beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
#         # print("alpha: ", alpha)
#         # print("beta: ", beta)
#         return alpha, beta

# class GAE(nn.Module):
#     def __init__(self, gamma, lmbda):
#         super().__init__()
#         self.register_buffer("gamma", torch.tensor(gamma))
#         self.register_buffer("lmbda", torch.tensor(lmbda))
#         self.gamma: torch.Tensor
#         self.lmbda: torch.Tensor
    
#     def forward(
#         self, 
#         reward: torch.Tensor, 
#         terminated: torch.Tensor, 
#         value: torch.Tensor, 
#         next_value: torch.Tensor
#     ):
#         num_steps = terminated.shape[1]
#         advantages = torch.zeros_like(reward)
#         not_done = 1 - terminated.float()
#         gae = 0
#         for step in reversed(range(num_steps)):
#             delta = (
#                 reward[:, step] 
#                 + self.gamma * next_value[:, step] * not_done[:, step] 
#                 - value[:, step]
#             )
#             advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
#         returns = advantages + value
#         return advantages, returns

# def make_batch(tensordict: TensorDict, num_minibatches: int):
#     tensordict = tensordict.reshape(-1) 
#     perm = torch.randperm(
#         (tensordict.shape[0] // num_minibatches) * num_minibatches,
#         device=tensordict.device,
#     ).reshape(num_minibatches, -1)
#     for indices in perm:
#         yield tensordict[indices]

# # @torch.no_grad()
# # def evaluate(env, policy, cfg, seed: int=0, exploration_type: ExplorationType=ExplorationType.MEAN):
# #     # 1. 确保环境和策略被显式推向指定设备
# #     env.to(cfg.device) 
# #     policy.to(cfg.device)
    
# #     env.enable_render(True)
# #     env.eval()
# #     env.set_seed(seed)

# #     # 2. 获取初始帧并强制指定设备
# #     # 即使环境里改了，这里手动 .to() 也是双重保险
# #     td = env.reset().to(cfg.device) 

# #     render_callback = RenderCallback(interval=2)
    
# #     # 3. 执行 10 步快速测试循环
# #     max_steps = 10 
# #     traj_list = []
    
# #     with set_exploration_type(exploration_type):
# #         for _ in range(max_steps):
# #             # 确保输入 Policy 前 TensorDict 带有正确的 device 属性
# #             td = td.to(cfg.device) 
            
# #             # 执行动作
# #             td = env.step(policy(td))
            
# #             # 执行渲染回调
# #             render_callback(env)
            
# #             # 克隆并存储，确保每一帧的设备属性都被保留
# #             traj_list.append(td.clone())
            
# #         # 4. 手动堆叠
# #         trajs = torch.stack(traj_list, dim=1) 
# #         # --- 诊断代码开始 ---
# #     print("-" * 30)
# #     print(f"DEBUG: trajs shape: {trajs.shape}") # 整体数据形状
# #     stats_td = trajs.get(("next", "stats"))
# #     print(f"DEBUG: stats_keys: {stats_td.keys()}") # 统计项有哪些
# #     test_key = list(stats_td.keys())[0]
# #     print(f"DEBUG: sample stats item ({test_key}) shape: {stats_td[test_key].shape}")
# #     print(f"DEBUG: done shape: {trajs.get(('next', 'done')).shape}")
# #     print("-" * 30)
# #     # --- 诊断代码结束 ---
# #     # 5. 后续处理逻辑
# #     env.enable_render(not cfg.headless)
# #     env.train()
# #     env.reset() # 评估后重置环境状态
# #     # --- 以下为原有的数据统计逻辑，无需修改 ---
# #     done = trajs.get(("next", "done")) 
# #     # 找到每个环境第一次 done 的索引
# #     first_done = torch.argmax(done.long(), dim=1).cpu()

# # # 1. 找到每个环境第一次 done 的索引 (形状: [num_envs])
# # # 1. 获取 Done 信号并处理维度
# #     # done 原始形状通常是 [128, 2, 10, 1] (env, agent, time, 1)
# #     done = trajs.get(("next", "done")).cpu()
    
# #     # 把它压平前两个维度，变成 [256, 10]
# #     # 然后找到每一架飞机第一次 done 的步数索引
# #     first_done = torch.argmax(done.squeeze(-1).float(), dim=-1).flatten() 

# #     def take_first_episode(tensor: torch.Tensor):
# #         # t 原始形状可能是 [128, 2, 10, ...]
# #         t = tensor.cpu()
# #         shape = t.shape
        
# #         # 💥 关键点：把 [128, 2] 合并成 [256]，让它和 first_done 的 256 对齐
# #         # 变形后为 [256, 10, ...]
# #         t = t.reshape(-1, shape[2], *shape[3:])
        
# #         # 计算需要补齐的维度（比如 stats 是标量还是向量）
# #         needed_dims = t.ndim - first_done.ndim 
        
# #         # 这里的 first_done 形状是 [256]，view 之后是 [256, 1, 1...]
# #         indices = first_done.view(first_done.shape[0], *((1,) * needed_dims))
        
# #         # 从时间轴 (dim=1) 提取数据
# #         res = torch.take_along_dim(t, indices, dim=1)
# #         return res.reshape(-1)

# #     # 2. 应用到所有的 stats 项上
# #     stats_dict = trajs.get(("next", "stats"))
# #     traj_stats = {
# #         k: take_first_episode(v)
# #         for k, v in stats_dict.items()
# #     }

# #     # 3. 计算最终均值发送给 wandb
# #     info = {
# #         "eval/stats." + k: torch.mean(v.float()).item() 
# #         for k, v in traj_stats.items()
# #     }
# #     return info
# @torch.no_grad()
# def evaluate(
#     env,
#     policy,
#     cfg,
#     seed: int=0, 
#     exploration_type: ExplorationType=ExplorationType.MEAN
# ):
#     # 1. 强制物理和策略对齐到 GPU
#     env.to(cfg.device) 
#     policy.to(cfg.device)
#     env.enable_render(True)
#     env.eval()
#     env.set_seed(seed)

#     # 2. 补丁：手动重置并确保初始 TensorDict 携带 device 属性
#     # 这解决了 "got None" 的初始帧问题
#     td = env.reset().to(cfg.device) 

#     render_callback = RenderCallback(interval=2)
    
#     # 3. 获取步数：直接从你在 YAML 里定义的变量读取
#     max_steps = env.max_episode_length 
#     traj_list = []
    
#     with set_exploration_type(exploration_type):
#         # 💥 关键改动：使用手动循环代替 rollout，确保每一帧都强制对齐设备
#         for _ in range(max_steps):
#             # 确保输入数据一定在 GPU 
#             td = td.to(cfg.device) 
            
#             # 环境步进
#             td = env.step(policy(td))
            
#             # 执行渲染（之前已改过 Callback 确保 frame 为 numpy）
#             render_callback(env)
            
#             # 存入列表 (clone 保证数据独立性)
#             traj_list.append(td.clone())
            
#         # 4. 手动执行堆叠：模拟 rollout 的输出结构 [env, time, agent, ...]
#         trajs = torch.stack(traj_list, dim=1) 

#     env.enable_render(not cfg.headless)
#     env.reset()
    
#     # --- 下面是已经验证过的多智能体兼容统计逻辑 ---
#     done = trajs.get(("next", "done")).cpu()
#     # 合并前两个维度 [128, 2] -> [256] 以匹配你的飞机总数
#     first_done = torch.argmax(done.squeeze(-1).float(), dim=-1).flatten() 

#     def take_first_episode(tensor: torch.Tensor):
#         t = tensor.cpu()
#         shape = t.shape
#         # 同样合并前两个维度，变成 [256, max_steps, ...]
#         t = t.reshape(-1, shape[2], *shape[3:])
#         needed_dims = t.ndim - first_done.ndim
#         indices = first_done.view(first_done.shape[0], *((1,) * needed_dims))
#         return torch.take_along_dim(t, indices, dim=1).reshape(-1)

#     traj_stats = {
#         k: take_first_episode(v)
#         for k, v in trajs.get(("next", "stats")).items()
#     }

#     info = {
#         "eval/stats." + k: torch.mean(v.float()).item() 
#         for k, v in traj_stats.items()
#     }

#     # 视频记录
#     info["recording"] = wandb.Video(
#         render_callback.get_video_array(axes="t c h w"), 
#         fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
#         format="mp4"
#     )
    
#     env.train()
#     return info

# # def vec_to_new_frame(vec, goal_direction):
# #     if (len(vec.size()) == 1):
# #         vec = vec.unsqueeze(0)
# #     # print("vec: ", vec.shape)

# #     # goal direction x
# #     goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
# #     z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
# #     # goal direction y
# #     goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
# #     goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
# #     # goal direction z
# #     goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
# #     goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

# #     n = vec.size(0)
# #     if len(vec.size()) == 3:
# #         vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
# #         vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
# #         vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
# #     else:
# #         vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
# #         vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
# #         vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

# #     vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

# #     return vec_new

# def vec_to_new_frame(vec, goal_direction):
#     """
#     万能版坐标变换：支持任意维度 (N, 3) 或 (N, A, 3) 或 (T, N, A, 3)
#     只要最后一维是 3 (x,y,z) 即可。
#     """
    
#     # 1. 计算 X 轴基向量 (归一化)
#     # 假设输入是 (N, A, 3)，norm 后是 (N, A, 1)，分母会自动广播
#     goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
#     # 2. 计算 Y 轴基向量 (Z cross X)
#     # 构造一个形状一样的 Z 轴向量 (0, 0, 1)
#     z_direction = torch.zeros_like(goal_direction_x)
#     z_direction[..., 2] = 1.0 
    
#     # torch.cross 支持广播，维度保持 (N, A, 3)
#     goal_direction_y = torch.cross(z_direction, goal_direction_x, dim=-1)
#     goal_direction_y = goal_direction_y / goal_direction_y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
#     # 3. 计算 Z 轴基向量 (X cross Y)
#     goal_direction_z = torch.cross(goal_direction_x, goal_direction_y, dim=-1)
#     goal_direction_z = goal_direction_z / goal_direction_z.norm(dim=-1, keepdim=True).clamp(min=1e-6)

#     # 4. 投影 (Projection) -> 使用点积 (Dot Product)
#     # 原理: 向量 A 在 B 方向的投影 = A · B
#     # 这里的 * 是逐元素相乘，sum(dim=-1) 是求和，合起来就是点积
#     # unsqueeze(-1) 是为了最后 cat 的时候方便
#     vec_x_new = (vec * goal_direction_x).sum(dim=-1, keepdim=True)
#     vec_y_new = (vec * goal_direction_y).sum(dim=-1, keepdim=True)
#     vec_z_new = (vec * goal_direction_z).sum(dim=-1, keepdim=True)

#     # 5. 拼接回 (N, A, 3)
#     vec_new = torch.cat([vec_x_new, vec_y_new, vec_z_new], dim=-1)

#     return vec_new
# def vec_to_world(vec, goal_direction):
#     world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
#     # directional vector of world coordinate expressed in the local frame
#     world_frame_new = vec_to_new_frame(world_dir, goal_direction)

#     # convert the velocity in the local target coordinate to the world coodirnate
#     world_frame_vel = vec_to_new_frame(vec, world_frame_new)
#     return world_frame_vel


# def construct_input(start, end):
#     input = []
#     for n in range(start, end):
#         input.append(f"{n}")
#     return "(" + "|".join(input) + ")"

import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) 
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int = 0,
    exploration_type: ExplorationType = ExplorationType.MEAN
):
    # 1. 初始化
    env.enable_render(True)
    env.eval()
    env.set_seed(seed)

    # ==========================================
    # 🛡️ 补丁: Blind Warmup (防物理崩溃)
    # ==========================================
    # 很多时候 Isaac Sim 需要这一步来刷新 Buffer
    try:
        env.reset()
        if hasattr(env, "num_agents"):
            # 构造微小动作强行跑一步
            warmup_action = torch.zeros(env.num_envs, env.num_agents, 4, device=cfg.device).fill_(0.05)
            td_warmup = env.reset().to(cfg.device)
            td_warmup.set(("agents", "action"), warmup_action)
            env.step(td_warmup)
    except Exception as e:
        print(f"[WARN] Warmup skipped: {e}")

    # 2. Rollout
    render_callback = RenderCallback(interval=2)
    
    with set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=env.max_episode_length,
            policy=policy,
            callback=render_callback,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )

    # 3. 还原环境设置
    env.enable_render(not cfg.headless)
    env.reset() # 清理状态

    # ==========================================
    # 🛡️ 补丁: 修复 维度不匹配 & 未完成任务Bug
    # ==========================================
    
    # 获取 done 信号 [Batch, Time, 1]
    key_done = ("next", "terminated") if ("next", "terminated") in trajs.keys(True) else ("next", "done")
    done = trajs.get(key_done).cpu()
    
    # 找到第一次 done 的索引 [Batch]
    # 注意：如果全0 (没done)，argmax 会返回 0，这是错误的。
    first_done_idx = torch.argmax(done.long().squeeze(-1), dim=1)
    
    # 修正没 done 的环境，设为最后一步
    has_done = (done.long().squeeze(-1).sum(dim=1) > 0)
    first_done_idx[~has_done] = trajs.shape[1] - 1

    # ✅✅✅ 修正维度的核心函数
    def take_first_episode(tensor: torch.Tensor):
        # tensor shape: [32, 250, 1] (3维)
        # first_done_idx shape: [32] (1维)
        
        # 我们需要 indices 也是 3维: [32, 1, 1]
        # (tensor.ndim - 1) = 2, 所以补两个 1
        indices = first_done_idx.reshape(first_done_idx.shape + (1,) * (tensor.ndim - 1))
        
        # 现在 [32, 250, 1] 和 [32, 1, 1] 维度数相同了，take_along_dim 才能工作
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    # 提取统计信息
    key_stats = ("next", "stats") if ("next", "stats") in trajs.keys(True) else ("stats",)
    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[key_stats].cpu().items()
    }

    info = {
        "eval/stats." + k: torch.mean(v.float()).item() 
        for k, v in traj_stats.items()
    }

    # 记录视频
    if hasattr(render_callback, "get_video_array"):
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"), 
            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
            format="mp4"
        )

    env.train()
    return info
def vec_to_new_frame(vec, goal_direction):
    """
    万能版坐标变换：支持任意维度
    """
    # 1. 计算 X 轴基向量
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    # 2. 计算 Y 轴基向量 (Z cross X)
    z_direction = torch.zeros_like(goal_direction_x)
    z_direction[..., 2] = 1.0 
    
    goal_direction_y = torch.cross(z_direction, goal_direction_x, dim=-1)
    goal_direction_y = goal_direction_y / goal_direction_y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    # 3. 计算 Z 轴基向量 (X cross Y)
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y, dim=-1)
    goal_direction_z = goal_direction_z / goal_direction_z.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # 4. 投影
    vec_x_new = (vec * goal_direction_x).sum(dim=-1, keepdim=True)
    vec_y_new = (vec * goal_direction_y).sum(dim=-1, keepdim=True)
    vec_z_new = (vec * goal_direction_z).sum(dim=-1, keepdim=True)

    # 5. 拼接
    vec_new = torch.cat([vec_x_new, vec_y_new, vec_z_new], dim=-1)

    return vec_new

def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel

def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"