# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Any, Dict, Optional, Sequence, Union, Tuple
import numpy as np
import torch
from tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms import (
    TransformedEnv,
    Transform,
    Compose,
    FlattenObservation,
    CatTensors
)
from torchrl.data import (
    TensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    CompositeSpec,
)
from .env import AgentSpec
from dataclasses import replace


def _transform_agent_spec(self: Transform, agent_spec: AgentSpec) -> AgentSpec:
    return agent_spec
Transform.transform_agent_spec = _transform_agent_spec


def _transform_agent_spec(self: Compose, agent_spec: AgentSpec) -> AgentSpec:
    for transform in self.transforms:
        agent_spec = transform.transform_agent_spec(agent_spec)
    return agent_spec
Compose.transform_agent_spec = _transform_agent_spec


def _agent_spec(self: TransformedEnv) -> AgentSpec:
    agent_spec = self.transform.transform_agent_spec(self.base_env.agent_spec)
    return {name: replace(spec, _env=self) for name, spec in agent_spec.items()}
TransformedEnv.agent_spec = property(_agent_spec)


class FromDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = None,
    ):
        if nbins is None:
            nbins = 2
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            self.minimum = action_spec.space.minimum.unsqueeze(-2)
            self.maximum = action_spec.space.maximum.unsqueeze(-2)
            self.mapping = torch.cartesian_prod(
                *[torch.linspace(0, 1, dim_nbins) for dim_nbins in nbins]
            ).to(action_spec.device)  # [prod(nbins), len(nbins)]
            n = self.mapping.shape[0]
            spec = DiscreteTensorSpec(
                n, shape=[*action_spec.shape[:-1], 1], device=action_spec.device
            )
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        mapping = self.mapping * (self.maximum - self.minimum) + self.minimum
        action = action.unsqueeze(-1)
        action = torch.take_along_dim(mapping, action, dim=-2).squeeze(-2)
        return action


class FromMultiDiscreteAction(Transform):
    def __init__(
        self,
        action_key: Tuple[str] = ("agents", "action"),
        nbins: Union[int, Sequence[int]] = 2,
    ):
        if action_key is None:
            action_key = "action"
        super().__init__([], in_keys_inv=[action_key])
        if not isinstance(action_key, tuple):
            action_key = (action_key,)
        self.nbins = nbins
        self.action_key = action_key

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if isinstance(action_spec, BoundedTensorSpec):
            if isinstance(self.nbins, int):
                nbins = [self.nbins] * action_spec.shape[-1]
            elif len(self.nbins) == action_spec.shape[-1]:
                nbins = self.nbins
            else:
                raise ValueError(
                    "nbins must be int or list of length equal to the last dimension of action space."
                )
            spec = MultiDiscreteTensorSpec(
                nbins, shape=action_spec.shape, device=action_spec.device
            )
            self.nvec = spec.nvec.to(action_spec.device)
            self.minimum = action_spec.space.minimum
            self.maximum = action_spec.space.maximum
        else:
            NotImplementedError("Only BoundedTensorSpec is supported.")
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        action = action / (self.nvec - 1) * (self.maximum - self.minimum) + self.minimum
        return action

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super()._inv_call(tensordict)


class DepthImageNorm(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        min_range: float,
        max_range: float,
        inverse: bool=False
    ):
        super().__init__(in_keys=in_keys)
        self.max_range = max_range
        self.min_range = min_range
        self.inverse = inverse

    def _apply_transform(self, obs: torch.Tensor) -> None:
        obs = torch.nan_to_num(obs, posinf=self.max_range, neginf=self.min_range)
        obs = obs.clip(self.min_range, self.max_range)
        if self.inverse:
            obs = (obs - self.min_range) / (self.max_range - self.min_range)
        else:
            obs = (self.max_range - obs) / (self.max_range - self.min_range)
        return obs


def ravel_composite(
    spec: CompositeSpec, key: str, start_dim: int=-2, end_dim: int=-1
):
    r"""
    
    Examples:
    >>> obs_spec = CompositeSpec({
    ...     "obs_self": UnboundedContinuousTensorSpec((1, 19)),
    ...     "obs_others": UnboundedContinuousTensorSpec((3, 13)),
    ... })
    >>> spec = CompositeSpec({
            "agents": {
                "observation": obs_spec
            }
    ... })
    >>> t = ravel_composite(spec, ("agents", "observation"))

    """
    composite_spec = spec[key]
    if not isinstance(key, tuple):
        key = (key,)
    if isinstance(composite_spec, CompositeSpec):
        in_keys = [k for k in spec.keys(True, True) if k[:len(key)] == key]

        # print("my print: ", in_keys)
        # ('agents', 'intrinsics', 'mass'), ('agents', 'intrinsics', 'inertia'), ('agents', 'intrinsics', 'com'), ('agents', 'intrinsics', 'KF'), ('agents', 'intrinsics', 'KM'), ('agents', 'intrinsics', 'tau_up'), ('agents', 'intrinsics', 'tau_down'), ('agents', 'intrinsics', 'drag_coef')]
        return Compose(
            FlattenObservation(start_dim, end_dim, in_keys),
            CatTensors(in_keys, out_key=key, del_keys=False)
        )
    else:
        raise TypeError


class VelController(Transform):
    def __init__(
        self,
        controller,
        yaw_control: bool = True,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state"), action_key])
        self.controller = controller
        self.yaw_control = yaw_control
        self.action_key = action_key
    

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        if (self.yaw_control):
            spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        else:
            spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(3,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    # def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
    #     # print("tensordict size: ", tensordict.shape)
    #     # print("tensor dict: ", tensordict)
    #     drone_state = tensordict[("info", "drone_state")][..., :13]
    #     # print("drone state shape: ", drone_state.shape)

    #     action = tensordict[self.action_key]
    #     print('转换推力的action:', action)
    #     if (self.yaw_control):
    #         target_vel, target_yaw = action.split([3, 1], -1)
    #         target_vel = target_vel.unsqueeze(1)
    #         target_yaw = target_yaw.unsqueeze(1)
    #         target_yaw = target_yaw * torch.pi
    #     else:
    #         target_vel = action.unsqueeze(1)
    #         # print("target vel: ", target_vel)
    #         # target_yaw = torch.zeros(action.shape[:-1] + (1,), device=action.device)
    #         target_yaw = None

    #     # print("drone vel shape: ", target_vel.shape)
    #     # print("target vel: ", target_vel)
    #     cmds = self.controller(
    #         drone_state, 
    #         target_vel=target_vel, 
    #         target_yaw=target_yaw
    #     )
    #     if torch.rand(1) < 0.05: # 降低打印频率
    #                 print(f"[3️⃣ Controller Output] Motor Cmds: {cmds[0,0].detach().cpu().numpy()}")
    #                 print("=====================================================")
    #     torch.nan_to_num_(cmds, 0.)
    #     tensordict.set(self.action_key, cmds)
    #     print('转换后的电机指令:', cmds)
    #     # if int(self.progress_buf[0].item()) % 200 == 0:
    #         # print(f"DEBUG: 转换后的 Action 维度 (应为4): {cmds.shape}")
    #     return tensordict
#     def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
#         # 1. 获取无人机真实状态
#         # state 结构通常是: [Pos(3), Quat(4), LinVel(3), AngVel(3)]
#         drone_state = tensordict[("info", "drone_state")][..., :13]
        
#         # 提取真实的世界系位置和速度，用于对比
#         current_pos = drone_state[..., :3]
#         current_vel = drone_state[..., 7:10]

#         # 2. 获取 Action (这里应该是 PPO 转好的世界系速度)
#         action = tensordict[self.action_key]
        
#         # 3. 解析结构
#         if (self.yaw_control):
#             target_vel, target_yaw = action.split([3, 1], -1)
#             target_vel = target_vel.unsqueeze(1)
#             target_yaw = target_yaw.unsqueeze(1) * torch.pi
#         else:
#             target_vel = action.unsqueeze(1)
#             target_yaw = None

#         # ==================== 🧪 强制测试区 (Debug Zone) ====================
#         # 【说明】在这里强制修改 target_vel，验证物理反应。
#         # 请每次只【取消注释】其中一个测试！
        
#         # --- 🟢 测试 1: Z轴 (推力测试) ---
#         target_vel.zero_()
#         target_vel[..., 0] = 0.0
#         target_vel[..., 1] = 0.0
#         target_vel[..., 2] = 1.0  # 向上飞 1m/s
#         print(f"target_vel: {target_vel}")
#         # --- 🔵 测试 2: X轴 (俯仰测试) ---
#         # target_vel.zero_()
#         # target_vel[..., 0] = 1.0  # 向北(前)飞 1m/s
#         # target_vel[..., 1] = 0.0
#         # target_vel[..., 2] = 0.0  # 保持高度0 (甚至可能掉下来，因为没给推力维持)
        
#         # --- 🟣 测试 3: Y轴 (滚转测试) ---
#         # target_vel.zero_()
#         # target_vel[..., 0] = 0.0
#         # target_vel[..., 1] = 1.0  # 向西(左)飞 1m/s
#         #  0.0
        
#         # ====================================================================

#         # 4. 核心解算
#         cmds = self.controller(
#             drone_state, 
#             target_vel=target_vel, 
#             target_yaw=target_yaw
#         )
        
#         # 5. 🔍 闭环监控打印
#         # 为了不刷屏，我们只在每 50 步打印一次，且只看第 0 个环境
# # 5. 🔍 闭环监控打印
#         if torch.rand(1) < 0.5:
#             # 【关键修改】多加一个 [0]，锁定到第 0 号 Agent
#             # 假设 target_vel 形状是 [Batch, 1, Agent, 3]
#             # [0] -> Batch 0
#             # [0] -> TimeStep 0 (unsqueeze的那一维)
#             # [0] -> Agent 0  <--- 之前少了这个
            
#             # 使用 try-except 保护，防止维度变化再次报错
#             try:
#                 # 这种写法兼容性最强：先转 numpy，再 flatten 展平，取前3个
#                 t_v = target_vel[0,0,0].detach().cpu().numpy().flatten()
#                 c_v = current_vel[0,0].detach().cpu().numpy().flatten() 
#                 c_p = current_pos[0,0].detach().cpu().numpy().flatten()
#                 mot = cmds[0,0].detach().cpu().numpy().flatten()
                
#                 print(f"\n======== 📡 物理链路 Debug ========")
#                 # 现在的 t_v[0] 绝对是标量了
#                 print(f"🎯 [期望] Target Vel (World): [{t_v[0]:.2f}, {t_v[1]:.2f}, {t_v[2]:.2f}]")
#                 print(f"🚙 [现状] Actual Vel (World): [{c_v[0]:.2f}, {c_v[1]:.2f}, {c_v[2]:.2f}]")
#                 print(f"📍 [位置] Actual Pos (World): [{c_p[0]:.2f}, {c_p[1]:.2f}, {c_p[2]:.2f}]")
#                 print(f"⚡ [执行] Motor Cmds        : {mot}")
                
#                 if abs(t_v[2]) > 0.1:
#                     diff = c_v[2] - t_v[2]
#                     status = "✅ 正在接近" if abs(diff) < 0.5 else "⚠️ 差距过大"
#                     print(f"   >>> Z轴响应检查: {status} (Err={diff:.2f})")
                
#                 if abs(t_v[0]) > 0.1:
#                     # 简单的逻辑检查：如果向前飞，后电机(通常index较大)应该大
#                     # 注意：这取决于你的电机排序，假设 2,3 是后电机
#                     is_pitch_down = mot[2] > mot[0] 
#                     print(f"   >>> X轴逻辑检查: {'✅ 后推力大(低头)' if is_pitch_down else '❌ 前推力大?'}")
                
#                 print("====================================")
#             except Exception as e:
#                 print(f"⚠️ Debug打印出错 (不影响训练): {e}")
#                 print(f"Shape Debug: TV={target_vel.shape}, CV={current_vel.shape}")
#         torch.nan_to_num_(cmds, 0.)
#         tensordict.set(self.action_key, cmds)
#         return tensordict
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # 1. 获取状态
        drone_state = tensordict[("info", "drone_state")][..., :13]
        current_vel = drone_state[..., 7:10] # [Batch, Agent, 3]

        # 2. 获取与处理动作
        action = tensordict[self.action_key]
        
        if self.yaw_control:
            target_vel, target_yaw = action.split([3, 1], -1)
            target_yaw = target_yaw * torch.pi 
        else:
            target_vel = action # [Batch, Agent, 3]
            target_yaw = None

        # 3. 控制器解算
        cmds = self.controller(
            drone_state, 
            target_vel=target_vel, 
            target_yaw=target_yaw
        )

        # # ==================== 📋 全量数据监控 (Full Monitor) ====================
        # # 只有 5% 的概率触发，但一旦触发，打印所有 Agent 的详情
        # if torch.rand(1) < 0.5: 
        #     try:
        #         # 转 Numpy 以便格式化，且不影响计算图
        #         tv_np = target_vel.detach().cpu().numpy()
        #         cv_np = current_vel.detach().cpu().numpy()
                
        #         # 获取维度: Batch数, Agent数
        #         B, A, _ = tv_np.shape
                
        #         # print(f"\n======== 📊 全局速度跟踪表 (Envs: {B}, Agents: {A}) ========")
        #         # # 打印表头
        #         # print(f"{'Env':<4} | {'Ag':<2} | {'Target (Vx, Vy, Vz)':<22} | {'Actual (Vx, Vy, Vz)':<22} | {'Error Diff':<22} | {'Status'}")
        #         # print("-" * 105)

        #         # 双重循环遍历所有数据
        #         for b in range(B):
        #             for a in range(A):
        #                 t = tv_np[b, a] # 目标
        #                 c = cv_np[b, a] # 现状
        #                 err = t - c     # 误差
                        
        #                 # 格式化字符串
        #                 t_str = f"[{t[0]:5.2f}, {t[1]:5.2f}, {t[2]:5.2f}]"
        #                 c_str = f"[{c[0]:5.2f}, {c[1]:5.2f}, {c[2]:5.2f}]"
        #                 e_str = f"[{err[0]:5.2f}, {err[1]:5.2f}, {err[2]:5.2f}]"
                        
        #                 # 状态诊断
        #                 status = "✅"
        #                 # 如果高度误差超过 0.5
        #                 if abs(err[2]) > 0.5:
        #                     status = "⚠️ 动力不足" if err[2] > 0 else "📉 掉高/过冲"
        #                 # 如果水平误差很大
        #                 elif np.linalg.norm(err[:2]) > 1.0:
        #                     status = "⚠️ 偏航"

        #         #         print(f"{b:<4} | {a:<2} | {t_str} | {c_str} | {e_str} | {status}")
                
        #         # print("=" * 105)

        #     except Exception as e:
        #         print(f"Debug Error: {e}")
        # # ======================================================================

        # 4. 写回
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict
class RateController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_rate, target_thrust = action.split([3, 1], -1)
        target_thrust = ((target_thrust + 1) / 2).clip(0.) * self.max_thrust
        cmds = self.controller(
            drone_state, 
            target_rate=target_rate * torch.pi, 
            target_thrust=target_thrust
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class AttitudeController(Transform):
    def __init__(
        self,
        controller,
        action_key: str = ("agents", "action"),
    ):
        super().__init__([], in_keys_inv=[("info", "drone_state")])
        self.controller = controller
        self.action_key = action_key
        self.max_thrust = self.controller.max_thrusts.sum(-1)
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(action_spec.shape[:-1]+(4,), device=action_spec.device)
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        drone_state = tensordict[("info", "drone_state")][..., :13]
        action = tensordict[self.action_key]
        target_thrust, target_yaw_rate, target_roll, target_pitch = action.split(1, dim=-1)
        cmds = self.controller(
            drone_state,
            target_thrust=((target_thrust+1)/2).clip(0.) * self.max_thrust,
            target_yaw_rate=target_yaw_rate * torch.pi,
            target_roll=target_roll * torch.pi,
            target_pitch=target_pitch * torch.pi
        )
        torch.nan_to_num_(cmds, 0.)
        tensordict.set(self.action_key, cmds)
        return tensordict


class History(Transform):
    def __init__(
        self,
        in_keys: Sequence[str],
        out_keys: Sequence[str]=None,
        steps: int = 32,
    ):
        if out_keys is None:
            out_keys = [
                f"{key}_h" if isinstance(key, str) else key[:-1] + (f"{key[-1]}_h",)
                for key in in_keys
            ]
        if any(key in in_keys for key in out_keys):
            raise ValueError
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.steps = steps
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            is_tuple = isinstance(in_key, tuple)
            if in_key in observation_spec.keys(include_nested=is_tuple):
                spec = observation_spec[in_key]
                spec = spec.unsqueeze(-1).expand(*spec.shape, self.steps)
                observation_spec[out_key] = spec
        return observation_spec

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key)
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            item = tensordict.get(in_key)
            item_history = tensordict.get(out_key).clone()
            item_history[..., :-1] = item_history[..., 1:]
            item_history[..., -1] = item
            tensordict.set(("next", out_key), item_history)
        return tensordict

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        _reset = tensordict.get("_reset", None)
        if _reset is None:
            _reset = torch.ones(tensordict.batch_size, dtype=bool, device=tensordict.device)
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if out_key not in tensordict.keys(True, True):
                item = tensordict.get(in_key)
                item_history = (
                    item.unsqueeze(-1)
                    .expand(*item.shape, self.steps)
                    .clone()
                    .zero_()
                )
                tensordict.set(out_key, item_history)
            else:
                item_history = tensordict.get(out_key)
                item_history[_reset] = 0.
        return tensordict

