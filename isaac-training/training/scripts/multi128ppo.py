import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world



class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        cnn_core = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            nn.Flatten(),           # (N,C,W,H) -> (N, C*W*H)
            nn.LazyLinear(128),
            nn.LayerNorm(128),
        ).to(self.device)

        class LidarAdapter(nn.Module):
            def __init__(self, core: nn.Module):
                super().__init__()
                self.core = core

            def forward(self, x: torch.Tensor):
                """
                支持任意 batch 维，只要求最后两维是 (W,H)，可选多一个 channel=1：
                (..., W, H) 或 (..., 1, W, H)
                输出：
                (..., 128)
                """
                if x.dim() < 3:
                    raise RuntimeError(f"lidar unexpected shape: {tuple(x.shape)}")

                # 允许 (...,1,W,H) -> (...,W,H)
                if x.dim() >= 4 and x.shape[-3] == 1:
                    x = x.squeeze(-3)

                *batch, W, H = x.shape  # batch 可能是 (E,A) / (E,T,A) / (B,A) / (B,) ...
                N = 1
                for b in batch:
                    N *= b

                # 送进 CNN：必须是 (N,1,W,H)
                y = self.core(x.reshape(N, 1, W, H))   # (N,128)
                return y.reshape(*batch, -1)           # (...,128)

        feature_extractor_network = LidarAdapter(cnn_core).to(self.device)

        self.feature_extractor = TensorDictSequential(
            TensorDictModule(
                feature_extractor_network,
                in_keys=[("agents", "observation", "lidar")],
                out_keys=[("agents", "_cnn_feature")],
            ),
            CatTensors(
                [("agents", "_cnn_feature"), ("agents", "observation", "state")],
                out_key=("agents", "_feature"),
                del_keys=False,
            ),
            TensorDictModule(
                make_mlp([256, 256]),
                in_keys=[("agents", "_feature")],
                out_keys=[("agents", "_feature")],
            ),
        ).to(self.device)



        # # =============== dynamic obstacle ===============
        # dynamic_obstacle_network = nn.Sequential(
        #     nn.Flatten(start_dim=-2),   # (E,A,K,10)->(E,A,K*10)
        #     nn.LazyLinear(128), nn.ELU(),   # ✅ 不要写死输入维度
        #     nn.Linear(128, 64), nn.ELU(),
        # ).to(self.device)

        # ✅ 拼接后的输入维度 = 128(cnn)+8(state)+64(dyn)=200
        # 你之前写 make_mlp([256,256]) 会对不上
        # self.feature_extractor = TensorDictSequential(
        #     TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], [("agents","_cnn_feature")]),
        #     CatTensors([("agents","_cnn_feature"), ("agents", "observation", "state")], ("agents","_feature"), del_keys=False), 
        #     TensorDictModule(make_mlp([256, 256]), [ ("agents","_feature")], [ ("agents","_feature")]),
        # ).to(self.device)

        self.n_agents=2
        self.action_dim = action_spec.shape[-1]
        self.actor = ProbabilisticActor(
            TensorDictModule(
                BetaActor(self.action_dim),
                in_keys=[("agents", "_feature")],
                out_keys=["alpha", "beta"],   # ✅ 两个 key
            ),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)


        self.critic = TensorDictModule(
            nn.LazyLinear(1),
            in_keys=[("agents", "_feature")],   # ✅ 一个 tuple key
            out_keys=["state_value"],           # ✅ 放在根上：tensordict["state_value"]
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)


        self.forward(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    # def __call__(self, tensordict):
    #     self.feature_extractor(tensordict)
    #     self.actor(tensordict)
    #     self.critic(tensordict)

    #     # Cooridnate change: transform local to world
    #     actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
    #     actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
    #     tensordict["agents", "action"] = actions_world
    #     return tensordict
    def forward(self, tensordict):
            # 1. 依次运行三个模块
            # 内部的 Rearrange ("e a ... -> (e a) ...") 会自动处理 EA 维度的合与拆
            self.feature_extractor(tensordict)
            self.actor(tensordict)
            self.critic(tensordict)
        # # --- 打印调试开始 ---
        #     print("-" * 30)
        #     # 查看整个 tensordict 的逻辑 batch_size (通常是 [E])
        #     print(f"Root Batch Size: {tensordict.batch_size}")
            
        #     # 查看经过网络处理后，具体特征的物理形状
        #     # 这里的 _feature 应该输出 torch.Size([E, A, 256])
        #     print(f"Feature Shape (_feature): {tensordict[('agents', '_feature')].shape}")
            
        #     # 查看状态数据的物理形状 (E, A, 8)
        #     print(f"State Shape: {tensordict[('agents', 'observation', 'state')].shape}")
            
        #     # 查看 Actor 算出的动作形状 (E, A, 4)
        #     print(f"Action Normalized Shape: {tensordict[('agents', 'action_normalized')].shape}")
        #     print("-" * 30)
            # --- 打印调试结束 ---
            # 2. 映射动作范围：从 Beta 分布的 [0, 1] 映射到 [-limit, limit]
            # 此时 action_normalized 的形状是 (E, A, 4)
            action_norm = tensordict["agents", "action_normalized"]
            actions = (2 * action_norm - 1.0) * self.cfg.actor.action_limit

            # 3. 坐标变换：将无人机机体系下的动作转到世界系
            # 注意：tensordict["agents", "observation", "direction"] 形状也是 (E, A, 3)
            # 这一步能拼对，正是因为两个张量的 EA 维度是完全镜像对齐的
            direction = tensordict["agents", "observation", "direction"]
            actions_world = vec_to_world(actions, direction)

            # 4. 写入最终动作
            # 必须写回 ("agents", "action")，这样环境才能通过 indices 准确拿走每架飞机的指令
            tensordict["agents", "action"] = actions_world
            
            return tensordict
    def train(self, tensordict):
        print(f"\n" + "="*30)
        print(f"[Train] 收到 Collector 的大包裹 Batch Size (E, T): {tensordict.batch_size}")
        
        # 这里的 reward 带有 T 维度
        reward_shape = tensordict["next", "agents", "reward"].shape
        print(f"[Train] 奖励形状 (E, T, A, 1): {reward_shape}")
        
        # 运行特征提取后
        self.feature_extractor(tensordict)
        print(f"[Train] 拼接后的特征形状 (E, T, A, 200): {tensordict[('agents', '_feature')].shape}")
        print("="*30)
            # 此时 tensordict 的 batch_size 包含 [num_envs, num_frames]
        next_tensordict = tensordict["next"]
        
        with torch.no_grad():
            # 1. 计算下一时刻特征。注意：不需要手动 vmap，
            # 你的 feature_extractor 内部已经处理了 (E, A) 维度。
            self.feature_extractor(next_tensordict) 
            next_values = self.critic(next_tensordict)["state_value"] # 形状: (E, T, A, 1)

        # 2. 获取奖励和终止信号
        rewards = tensordict["next", "agents", "reward"]  # (E,T,A,1)
        dones   = tensordict["next", "terminated"]        # (E,T,1)
        values  = tensordict["state_value"]               # (E,T,A,1)
        next_values = next_values                         # (E,T,A,1)

        if rewards.dim() == 3:
            rewards = rewards.unsqueeze(-1) 
        E, T, A, _ = rewards.shape

        print("\n===== DEBUG reward =====")
        print("reward.shape =", rewards.shape)
        print("reward.dim   =", rewards.dim())
        print("tensordict.batch_size =", tensordict.batch_size)
        print("next.batch_size       =", tensordict["next"].batch_size)

        # 也顺手看看 terminated / done 的维度（很关键）
        term = tensordict["next", "terminated"]
        done = tensordict["next", "done"] if ("done" in tensordict["next"].keys(True, True)) else None
        print("terminated.shape =", term.shape, "dim=", term.dim())
        if done is not None:
            print("done.shape       =", done.shape, "dim=", done.dim())

        # 再看看 agent 维到底在哪（reward 是不是 per-agent）
        print("obs.state.shape =", tensordict["next", "agents", "observation", "state"].shape)

        print("===== DEBUG END =====\n")


        # dones 扩展到 agent 维: (E,T,1)->(E,T,A,1)
        dones = dones.unsqueeze(-2).expand(-1, -1, A, -1)

        # (E,T,A,1) -> (E*A,T,1)
        def EA_to_batch(x):
            return x.permute(0, 2, 1, 3).reshape(E*A, T, 1)

        rewards_b = EA_to_batch(rewards)
        dones_b   = EA_to_batch(dones)
        values_b  = EA_to_batch(values)
        next_b    = EA_to_batch(next_values)

        # 现在喂给单智能体 GAE 就对了
        adv_b, ret_b = self.gae(rewards_b, dones_b, values_b, next_b)  # (E*A,T,1)

        # 再还原回 (E,T,A,1)
        def batch_to_EA(x):
            return x.reshape(E, A, T, 1).permute(0, 2, 1, 3)

        adv = batch_to_EA(adv_b)
        ret = batch_to_EA(ret_b)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # 8. 训练循环
        infos = []
        
        for epoch in range(self.cfg.training_epoch_num):
            # make_batch 会把 [E, T, A] 全部 reshape(-1) 拍扁
            # 这样 _update 看到的就是纯粹的批次数据
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        # 9. 统计平均指标
        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    
    def _update(self, tensordict): 
        # 此时 tensordict.batch_size 是 [Minibatch_Size] (例如 [4096])
        
        # 1. 特征提取
        # 确保 feature_extractor 内部不再依赖固定的 (E, A) 结构
        self.feature_extractor(tensordict)

        # 2. 计算当前策略的分布
        action_dist = self.actor.get_dist(tensordict) 
        
        # 3. 计算 Log Prob
        # 这里的 action_normalized 形状是 (Batch, 4)
        # log_probs 输出形状通常是 (Batch,)
        log_probs = action_dist.log_prob(tensordict["agents", "action_normalized"])

        # 4. Entropy Loss
        action_entropy = action_dist.entropy()
        # mean 会跨越所有智能体和所有环境，这在平铺后的 Batch 里是正确的
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # 5. Actor Loss (PPO)
        # 优势函数形状是 (Batch, 1)
        advantage = tensordict["adv"] 
        new_lp = log_probs                      # 你算出来的
        old_lp = tensordict["sample_log_prob"]  # 你取出来的

        # print("action        ", tensordict[("agents","action")].shape if ("agents","action") in tensordict.keys(True) else tensordict["action"].shape)
        # print("new_log_probs ", new_lp.shape)
        # print("old_log_prob  ", old_lp.shape)
        # print("old_log_prob keys top:", [k for k in tensordict.keys(True) if "log_prob" in str(k)])

        # # 重要：确保 log_probs 增加一维变成 (Batch, 1) 从而与 advantage 匹配
        # sample_log_prob 是采样时存下的老策略概率

        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
                        
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        
        # 计算 Policy Loss
        actor_loss = -torch.mean(torch.min(surr1, surr2))

        # 6. Critic Loss (Value Clipping)
        b_value = tensordict["state_value"] # 采样时的旧估值 (Batch, 1)
        ret = tensordict["ret"]             # GAE 算出的目标值 (Batch, 1)
        
        # 当前 Critic 的新预测
        value = self.critic(tensordict)["state_value"] 
        
        # PPO 特有的 Critic 剪切，防止价值网络更新过猛
        value_clipped = b_value + (value - b_value).clamp(
            -self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio
        )
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # 7. 总损失与优化
        # 注意：如果 feature_extractor 也有参数，必须一起反向传播
        loss = entropy_loss + actor_loss + critic_loss

        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        # 梯度裁剪：防止训练崩溃
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()

        # 统计：解释方差 (Explained Variance)
        # 越接近 1 说明 Critic 预测越准
        explained_var = 1 - F.mse_loss(value, ret) / (ret.var() + 1e-7)

        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])