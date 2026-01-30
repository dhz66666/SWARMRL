import torch  # 导入 PyTorch 深度学习库
import einops  # 导入 einops 库，用于灵活的张量维度操作（如重排、归约）
import numpy as np  # 导入 NumPy 库，用于数值计算
from tensordict.tensordict import TensorDict, TensorDictBase  # 导入 TensorDict，用于高效管理异构张量数据
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec  # 导入 TorchRL 的数据规范类，定义观测/动作空间
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec  # 导入 OmniDrones 的基础环境类
import omni.isaac.orbit.sim as sim_utils  # 导入 Isaac Orbit 的仿真工具函数
from omni_drones.robots.drone import MultirotorBase  # 导入多旋翼无人机基类
from omni.isaac.orbit.assets import AssetBaseCfg  # 导入资产配置基类
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg  # 导入地形生成和导入相关的配置类
from omni_drones.utils.torch import euler_to_quaternion, quat_axis  # 导入四元数和欧拉角转换工具
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns  # 导入光线投射器（LiDAR仿真）及其配置和扫描模式
from omni.isaac.core.utils.viewports import set_camera_view  # 导入设置相机视角的工具（调试用）
from utils import vec_to_new_frame, vec_to_world, construct_input  # 导入自定义的向量坐标变换工具函数
import omni.isaac.core.utils.prims as prim_utils  # 导入 Isaac Core 的基本图元操作工具
import omni.isaac.orbit.sim as sim_utils  # (重复导入) 导入仿真工具
import omni.isaac.orbit.utils.math as math_utils  # 导入数学工具库
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg  # 导入刚体对象及其配置类（用于动态障碍物）
import time  # 导入时间库

class NavigationEnv(IsaacEnv):  # 定义导航环境类，继承自 IsaacEnv

    # 仿真步骤说明:
    # 1. _pre_sim_step: 应用动作 -> 执行 Isaac Sim 物理步进
    # 2. _post_sim_step: 更新 LiDAR 数据
    # 3. 增加 progress_buf (步数计数器)
    # 4. _compute_state_and_obs: 获取观测值和状态，更新统计信息
    # 5. _compute_reward_and_done: 计算奖励并判断是否结束

    def __init__(self, cfg):  # 初始化函数，传入配置对象 cfg
        print("[Navigation Environment]: Initializing Env...")  # 打印初始化日志
        self.num_agents = 2
        # --- LiDAR 参数设置 ---
        self.lidar_range = cfg.sensor.lidar_range  # 从配置中获取 LiDAR 的最大探测范围
        # 设置垂直视场角 (VFOV)，限制在 -89 到 89 度之间，防止数值问题
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams  # 获取垂直方向的线束数量
        self.lidar_hres = cfg.sensor.lidar_hres  # 获取水平分辨率（每多少度一根线）
        self.lidar_hbeams = int(360/self.lidar_hres)  # 计算水平方向的总线束数量 (360度 / 分辨率)
# 调用父类，完成环境初始化，这是关键的一步。一旦你调用它，它就开始干活，并且会在中间**“插空”**调用你写的函数：完成环境等的初始化
        super().__init__(cfg, cfg.headless)  # 调用父类初始化函数，设置是否为无头模式 (headless)
        self._dbg_last_z = None

        # 只保留一个“最终动作”来源，避免被覆盖
        self._dbg_last_action_final = None   # shape [E,A,4]
        self._dbg_last_act_step = -1         # int，用于对齐
        self._dbg_seq = 0                    # 自增ID（最稳）

        # --- 无人机初始化 ---
#动作：建立 PyTorch 张量与 PhysX 显存数据的映射关系 (View)。 找到spawn创建的物理地址的数据，传给torch
        self.drone.initialize()  # 初始化无人机对象
        self.init_vels = torch.zeros_like(self.drone.get_velocities())  # 创建一个全零张量，用于存储初始速度
# 雷达只是一个物理属性，要完成无人机初始化后才能找到无人机
        # --- LiDAR 传感器初始化 ---
        ray_caster_cfg = RayCasterCfg(  # 配置 RayCaster (光线投射器)
# 挂载 (Attach)：找到后（比如找到了 4096 个无人机的 base_link），它会在逻辑上把 LiDAR 的原点 “钉” 在这些 base_link 上。
# 这就是“跟随更新”的秘密： 只要父节点（无人机）动了，根据计算机图形学的规则，所有子节点（LiDAR）会自动继承父节点的变换矩阵（位置和旋转）。
            prim_path="/World/envs/env_.*/Hummingbird.*/base_link", # 指定传感器绑定的物体路径 (绑定到无人机的 base_link)
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),  # 设置传感器相对于绑定物体的偏移量 (0,0,0)
            attach_yaw_only=True,  # 设置为 True 表示 LiDAR 只跟随偏航角旋转，不随俯仰/翻滚倾斜 (保持水平)
            # attach_yaw_only=False, # (注释掉的选项)
            pattern_cfg=patterns.BpearlPatternCfg(  # 配置扫描模式，这里使用 Bpearl 模式
                horizontal_res=self.lidar_hres, # 设置水平分辨率
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams)  # 设置垂直方向的光线角度分布
            ),
            debug_vis=False,  # 关闭调试可视化
            mesh_prim_paths=["/World/ground"],  # 指定 LiDAR 探测的目标网格路径 (这里只探测地面/地形)
            # mesh_prim_paths=["/World"], # (注释掉的选项)
        )
        self.lidar = RayCaster(ray_caster_cfg)  # 实例化 RayCaster 对象
        self.lidar._initialize_impl()  # 调用内部初始化实现
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)  # 存储 LiDAR 的分辨率 (水平线束, 垂直线束)
        # ==================== 🕵️ 审计 B：视觉几何 ====================
        print(f"\n👁️ [LiDAR Geometry Audit]")
        print(f"  > V-Beams: {self.lidar_vbeams}")
        print(f"  > V-FOV:   {self.lidar_vfov}")
        
        # 计算每一束线的理论角度
        angles = torch.linspace(*self.lidar_vfov, self.lidar_vbeams)
        print(f"  > Ray Angles (deg): {angles.cpu().numpy()}")
        
        # 如果角度里包含 -45 或 -90，说明它肯定在看地板

        # ==========================================================
#         # --- 目标点和状态变量初始化 ---
#         with torch.device(self.device):  # 在指定的设备 (GPU/CPU) 上创建张量
#             # self.start_pos = torch.zeros(self.num_envs, 1, 3) # (注释掉的变量)
#             self.target_pos = torch.zeros(self.num_envs, 1, 3)  # 初始化h 目标位置张量 (环境数, 1, 3)
            
#             # 坐标变换变量: 添加目标方向变量
#             self.target_dir = torch.zeros(self.num_envs, 1, 3)  # 初始化目标方向张量
#             self.height_range = torch.zeros(self.num_envs, 1, 2)  # 初始化高度范围张量 (用于高度限制)
#             self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)  # 初始化上一帧无人机速度 (用于计算平滑度奖励)
#             # (以下是注释掉的硬编码目标位置代码)
#             # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
#             # self.target_pos[:, 0, 1] = 24.
#             # self.target_pos[:, 0, 2] = 2.     
# # 假设你在 __init__ 开头定义了 self.num_agents = 2
# 改为多智能体
        with torch.device(self.device):
            # self.terminated = torch.zeros(self.num_envs, self.num_agents, dtype=torch.bool)
            # self.truncated = torch.zeros(self.num_envs, self.num_agents, dtype=torch.bool)
            # self.terminated = torch.zeros(self.num_envs, self.num_agents, dtype=torch.bool)
            # self.truncated = torch.zeros(self.num_envs, self.num_agents, dtype=torch.bool)
            # (N, 2, 3) -> 每个房间 2 个目标点
            self.target_pos = torch.zeros(self.num_envs, self.num_agents, 3) 
            
            # (N, 2, 3) -> 每个无人机都有自己的朝向
            self.target_dir = torch.zeros(self.num_envs, self.num_agents, 3) 
            
            # (N, 2, 2) -> 每个无人机可能有不同的高度限制
            self.height_range = torch.zeros(self.num_envs, self.num_agents, 2) 
            
            # (N, 2, 3) -> 记录每一架无人机的上一帧速度
            self.prev_drone_vel_w = torch.zeros(self.num_envs, self.num_agents, 3)

    def _design_scene(self):  # 设计仿真场景的函数
        # --- 初始化无人机 ---
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # 从注册表中获取指定的无人机模型类
        cfg = drone_model.cfg_cls(force_sensor=False)  # 获取无人机配置，关闭力传感器
        self.drone = drone_model(cfg=cfg)  # 实例化无人机对象
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0] # (注释掉的生成代码)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]  # 在场景中生成无人机，初始高度 2.0 米
        translations = [
            (0.0, -1.0, 2.0), # Agent 0
            (0.0, 1.0, 2.0)
            # (0.0, 3.0, 2.0), # Agent 1
            # (0.0, -3.0, 2.0)  # Agent 2
        ] 
# 确保 translations 数量匹配 self.num_agents  实体化（克隆）
# 因此，后续当你调用 self.drone.initialize() 时，它就会根据这里 spawn 的数量，
# 自动把速度、位置等张量的维度初始化为 (num_envs, num_agents, 3)。环境初始化
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]
        drone_prim = self.drone.spawn(translations=translations[:self.num_agents])
# 🟢 开关控制：True = 显示目标球，False = 隐藏目标球
        visualize_target = False  
        
        colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
        self.target_objs = [] # 存储 RigidObject 句柄

        # 只有当开关打开时，才执行下面的生成逻辑
        if visualize_target:
            for i in range(self.num_agents):
                # 1. 定义配置
                target_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Target_{i}", 
                    spawn=sim_utils.SphereCfg(
                        radius=0.2,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[i % 3]),
                        
                        # rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                        # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        
                        # 简化写法：既然只是为了看，其实很多属性可以精简，保持你原来的也没问题
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -10.0)),
                )

                # 2. 手动生成
                target_cfg.spawn.func(
                    f"/World/envs/env_0/Target_{i}", 
                    target_cfg.spawn, 
                    translation=(0.0, 0.0, -10.0)
                )

                # 3. 清空 spawn
                target_cfg.spawn = None
                
                # 4. 创建对象并加入列表
                target_obj = RigidObject(cfg=target_cfg)
                self.target_objs.append(target_obj)
        # --- 环境光照设置 ---
        light = AssetBaseCfg(  # 配置远距离平行光
            prim_path="/World/light",  # 光源路径
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),  # 设置颜色和强度
        )
        sky_light = AssetBaseCfg(  # 配置天空光 (环境光)
            prim_path="/World/skyLight",  # 天空光路径
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),  # 设置颜色和强度
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)  # 在仿真中生成平行光
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)  # 在仿真中生成天空光
        
        # --- 地面设置 ---/世界坐标系被钉在了原点，高度 (Z)：Z=0 是地平面（虽然为了防止渲染闪烁 z-fighting，稍微抬高了 0.01m）。
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))  # 配置地面平面，颜色深灰，尺寸 300x300
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))  # 生成地面，稍微抬高 0.01 防止z-fighting

        self.map_range = [20.0, 20.0, 4.5]  # 定义地图范围 [x, y, z]

        # --- 地形生成 ---
        terrain_cfg = TerrainImporterCfg(  # 地形导入器配置
            num_envs=self.num_envs,  # 环境数量
            env_spacing=0.0,  # 环境间距 (因为使用单个大地形，所以设为0)
            prim_path="/World/ground",  # 地形路径
            terrain_type="generator",  # 地形类型为生成器
            terrain_generator=TerrainGeneratorCfg(  # 地形生成器配置
                seed=0,  # 随机种子
                size=(self.map_range[0]*2, self.map_range[1]*2),  # 地形总尺寸
                border_width=5.0,  # 边界宽度
                num_rows=1,  # 行数
                num_cols=1,  # 列数
                horizontal_scale=0.1,  # 水平缩放比例
                vertical_scale=0.1,  # 垂直缩放比例
                slope_threshold=0.75,  # 坡度阈值
                use_cache=False,  # 不使用缓存
                color_scheme="height",  # 根据高度着色
                sub_terrains={  # 子地形配置
                    "obstacles": HfDiscreteObstaclesTerrainCfg(  # 使用离散障碍物地形
                        horizontal_scale=0.1,  # 水平缩放
                        vertical_scale=0.1,  # 垂直缩放
                        border_width=0.0,  # 边界宽
                        num_obstacles=self.cfg.env.num_obstacles,  # 障碍物数量
                        obstacle_height_mode="range",  # 障碍物高度模式为范围
                        obstacle_width_range=(0.4, 1.1),  # 障碍物宽度范围
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],  # 障碍物可选高度列表
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],  # 对应高度的概率分布
                        platform_width=0.0,  # 平台宽度
                    ),
                },
            ),
            visual_material = None,  # 视觉材质
            max_init_terrain_level=None,  # 最大初始地形等级
            collision_group=-1,  # 碰撞组
            debug_vis=True,  # 开启调试可视化
        )
        terrain_importer = TerrainImporter(terrain_cfg)  # 实例化地形导入器

        if (self.cfg.env_dyn.num_obstacles == 0):  # 如果配置中动态障碍物数量为 0
            return  # 直接返回，不生成动态障碍物
        
        # --- 动态障碍物 (Dynamic Obstacles) ---
        # 注意：使用长方体 (Cuboid) 代表 3D 浮动障碍物，使用长圆柱体 (Cylinder) 代表 2D 障碍物 (必须绕行)
        # 障碍物宽度分为 N_w=4 个区间: [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # 障碍物高度分为 N_h=2 个区间: [[0, 0.5], [0.5, inf]]，用于区分 3D 和 2D 障碍物
        N_w = 4 # 宽度区间数量
        N_h = 2 # 高度区间数量 (目前只支持二元分类)
        max_obs_width = 1.0  # 最大障碍物宽度
        self.max_obs_3d_height = 1.0  # 最大 3D 障碍物高度
        self.max_obs_2d_height = 5.0  # 最大 2D 障碍物高度 (足够高以至于无法飞跃)
        self.dyn_obs_width_res = max_obs_width/float(N_w)  # 宽度分辨率
        dyn_obs_category_num = N_w * N_h  # 动态障碍物总类别数
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)  # 每类障碍物的数量
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # 修正总数量以避免整除误差

        # 动态障碍物信息初始化
        self.dyn_obs_list = []  # 存储障碍物对象列表
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 障碍物状态张量 (位置+姿态+速度等)
        self.dyn_obs_state[:, 3] = 1. # 初始化四元数的实部为 1 (无旋转)
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)  # 障碍物移动目标点
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)  # 障碍物初始位置
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)  # 障碍物速度
        self.dyn_obs_step_count = 0 # 动态障碍物运动步数计数器
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # 障碍物尺寸

        # 辅助函数：检查位置有效性，确保障碍物分布均匀
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:  # 遍历已存在的位置
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):  # 如果距离太近
                    return False  # 返回无效
            return True  # 返回有效
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # 计算理想的障碍物间距
        curr_obs_dist = obs_dist  # 当前使用的间距阈值
        prev_pos_list = [] # 用于存储已生成的位置
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)  # 计算长方体和圆柱体各自的类别数
        
        # 遍历所有类别生成障碍物
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # 为该类别的每个实例生成原点
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # 随机采样原点，直到满足均匀分布条件
                start_time = time.time()  # 记录开始时间
                while (True):  # 循环尝试
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])  # 随机 X
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])  # 随机 Y
                    if (category_idx < cuboid_category_num):  #如果是 3D 障碍物
                        oz = np.random.uniform(low=0.0, high=self.map_range[2])  # 随机 Z
                    else:  # 如果是 2D 障碍物
                        oz = self.max_obs_2d_height/2. # 高度固定为中心位置
                    curr_pos = np.array([ox, oy])  # 当前平面位置
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)  # 检查位置是否有效
                    curr_time = time.time()  # 获取当前时间
                    if (curr_time - start_time > 0.1):  # 如果尝试时间超过 0.1秒
                        curr_obs_dist *= 0.8  # 降低距离要求
                        start_time = time.time()  # 重置时间
                    if (valid):  # 如果位置有效
                        prev_pos_list.append(curr_pos)  # 加入已生成列表
                        break  # 退出循环
                curr_obs_dist = obs_dist  # 重置距离阈值
                origin = [ox, oy, oz]  # 确定的原点
                # 记录原点到 Tensor
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                # 创建一个 Xform 原点 Prim，用于后续生成几何体
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # 生成不同尺寸的动态障碍物
            if (category_idx < cuboid_category_num):
                # 生成 3D 动态障碍物 (长方体)
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)  # 计算宽度
                obs_height = self.max_obs_3d_height  # 固定高度
                cuboid_cfg = RigidObjectCfg(  # 长方体配置
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid", # 路径使用了正则匹配模式
                    spawn=sim_utils.CuboidCfg(  # 生成长方体
                        size=[width, width, self.max_obs_3d_height],  # 尺寸
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # 刚体属性
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # 质量属性
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # 碰撞属性 (设为 False 由 RayCaster 处理或手动计算)
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),  # 材质颜色 (绿色)
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),  # 初始状态
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)  # 实例化对象
            else:
                # 生成 2D 动态障碍物 (圆柱体)
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.  # 计算半径
                obs_width = radius * 2  # 计算直径
                obs_height = self.max_obs_2d_height  # 高度
                cylinder_cfg = RigidObjectCfg(  # 圆柱体配置
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(  # 生成圆柱体
                        radius = radius,  # 半径
                        height = self.max_obs_2d_height,  # 高度
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)  # 实例化对象
            self.dyn_obs_list.append(dynamic_obstacle)  # 加入列表
            # 记录尺寸信息
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)


    # def move_dynamic_obstacle(self):  # 动态障碍物移动逻辑
    #     # 步骤 1: 为需要更新的动态障碍物随机采样新目标
    #     # 计算当前位置到目标的距离
    #     dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
    #         else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
    #     dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # 如果距离小于 0.5，则标记需要新目标
        
    #     # 在局部范围内采样新目标
    #     num_new_goal = torch.sum(dyn_obs_new_goal_mask)  # 需要更新的数量
    #     # 随机采样局部 X, Y, Z 偏移
    #     sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
    #     sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
    #     sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
    #     sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)  # 组合成向量
    
    #     # 将局部目标应用到全局范围 (基于原点偏移)
    #     self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
    #     # 将目标限制在地图范围内
    #     self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
    #     self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
    #     self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
    #     self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # 强制 2D 障碍物的高度中心不变


    #     # 步骤 2: 大约每 2 秒随机采样一次速度
    #     if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
    #         # 随机生成速度范数 (大小)
    #         self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
    #           - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
    #         # 设置速度方向指向目标，并应用速度大小
    #         self.dyn_obs_vel = self.dyn_obs_vel_norm * \
    #             (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

    #     # 步骤 3: 计算当前时间步的位置更新 (欧拉积分)
    #     self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


    #     # 步骤 4: 更新仿真中的可视化位置
    #     for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
    #         # 将计算出的状态写入仿真
    #         dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
    #         dynamic_obstacle.write_data_to_sim()  # 应用数据
    #         dynamic_obstacle.update(self.cfg.sim.dt)  # 更新物理

    #     self.dyn_obs_step_count += 1  # 步数加一


 

    def _set_specs(self):
        # ▼▼▼ 核心修改：定义环境的 Batch Size 为 [16, 2] ▼▼▼
        # 这意味着 TorchRL 会认为我们有 32 个独立的“智能体实例”
        E = self.num_envs
        A = self.num_agents

        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        dyn_obs_num = self.cfg.algo.feature_extractor.dyn_obs_num

        action_dim = self.drone.action_spec.shape[-1]
        drone_state_dim = self.drone.state_spec.shape[0]  # 如果你后面要 central state 用得到
        self.batch_size = torch.Size([E])

        # ----------------------------------------------------------------------
        # 1. 观测空间 (Observation Spec)
        # ----------------------------------------------------------------------
        observation_agent_spec = CompositeSpec({
            "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),

            # 你实际 lidar_scan 是 [E,A,H,V]，所以单个 agent 就是 (H,V)
            "lidar": UnboundedContinuousTensorSpec((self.lidar_hbeams, self.lidar_vbeams), device=self.device),

            "direction": UnboundedContinuousTensorSpec((3,), device=self.device),

            # "dynamic_obstacle": UnboundedContinuousTensorSpec(
            #     (dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device
            # ),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
        "agents": {
            # 每个 env 有 A 个 agent，每个 agent 看到 observation_agent_spec
            "observation": observation_agent_spec.expand(A),

            # central 给 critic / debug：每个 env 一份（内部自己带 A 维）

        },
        }).expand(E).to(self.device)
   
# 修改后
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                # 这里的 torch.stack 已经处理了智能体维度 A
                "action": torch.stack([self.drone.action_spec] * self.num_agents, dim=0),
            })
        }).expand(self.num_envs).to(self.device) # <--- expand 放在这里，确保整个 spec 的 batch_size 变为 [E]
        # self.reward_spec = CompositeSpec({
        #     "agents": CompositeSpec({
        #         "reward": UnboundedContinuousTensorSpec((1,), device=self.device).expand(A),
        #     })
        # }).expand(E).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((A,), device=self.device),
            })
        }).expand(E).to(self.device)

        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(E).to(self.device)

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec((A,), device=self.device),   # 每个 agent 一条 return
            "episode_len": UnboundedContinuousTensorSpec((1,), device=self.device),
            "reach_goal": UnboundedContinuousTensorSpec((1,), device=self.device),
            "collision": UnboundedContinuousTensorSpec((1,), device=self.device),
            "truncated": UnboundedContinuousTensorSpec((1,), device=self.device),
        }).expand(E).to(self.device)

        info_spec = CompositeSpec({
            # 必须写成 (A, 13)
            "drone_state": UnboundedContinuousTensorSpec((13,), device=self.device).expand(self.num_agents, 13),
        }).expand(E).to(self.device)

        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

        # ----------------------------------------------------------------------
        # 8) ✅ AgentSpec：像 Formation 那样把 key 绑好（训练框架才知道去哪取）
        # ----------------------------------------------------------------------
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            A,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            
        )
    def reset_target(self, env_ids: torch.Tensor):  # 重置目标点函数
        if (self.training):  # 如果是训练模式
            # 定义两种掩码和位移，用于将目标点随机放置在地图两侧
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))  # 随机选择位置模式
            selected_masks = masks[mask_indices].unsqueeze(1)  # 获取选中的掩码
            selected_shifts = shifts[mask_indices].unsqueeze(1)  # 获取选中的位移


            # 生成随机位置
            target_pos = 48. * torch.rand(len(env_ids), self.num_agents, 3, device=self.device) - 24.
# 再给每个 agent 单独 heights、masks、shifts

            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)  # 随机高度 [0.5, 2.5]
            target_pos[:, 0, 2] = heights # 设置高度
            target_pos = target_pos * selected_masks + selected_shifts  # 应用掩码和位移
            
            # 应用目标位置
            self.target_pos[env_ids] = target_pos

            # (以下为注释代码)
            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        # else:  # 如果是评估模式 (位置固定)
        #     self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
        #     self.target_pos[:, 0, 1] = -24.
        #     self.target_pos[:, 0, 2] = 2.            
        else:  # 如果是评估模式 (位置固定)
            # ▼▼▼ 修改：使用 [:, :, x] 覆盖所有智能体 ▼▼▼
            # 这样 Agent 0 和 Agent 1 的目标点完全重合
            self.target_pos[:, :, 0] = torch.linspace(-0.5, 0.5, self.num_envs).unsqueeze(-1) * 32.
            self.target_pos[:, :, 1] = -24.
            self.target_pos[:, :, 2] = 2.
    
    #     self.stats[env_ids] = 0.  # 重置统计信息
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)

        # 1. 获取当前重置的环境数量和智能体数量
        N = len(env_ids)
        A = self.num_agents

        # 2. ✅✅✅ 关键修复：在这里（最前面）定义 sep
        # 这样无论走下面的 if 还是 else，sep 都是存在的
        sep = torch.zeros(N, A, 3, device=self.device)
        if A >= 2:
            sep[:, 0, 1] = -2.0  # Agent 0 向 Y 负方向偏移
            sep[:, 1, 1] =  2.0  # Agent 1 向 Y 正方向偏移
        # if A >= 4: ... (如果需要更多)

        # 3. 根据模式选择中心点生成逻辑
        if self.training:
            # === 训练模式 ===
            masks = torch.tensor(
                [[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]],
                dtype=torch.float, device=self.device
            )
            shifts = torch.tensor(
                [[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]],
                dtype=torch.float, device=self.device
            )
            mask_indices = torch.randint(0, masks.size(0), (N,), device=self.device)
            selected_masks = masks[mask_indices].unsqueeze(1)    # (N,1,3)
            selected_shifts = shifts[mask_indices].unsqueeze(1)  # (N,1,3)

            # 随机中心点
            center = 48.0 * torch.rand(N, 1, 3, device=self.device) - 24.0
            heights = 0.5 + torch.rand(N, 1, device=self.device) * (2.5 - 0.5)
            center[:, 0, 2] = heights[:, 0]
            
            center = center * selected_masks + selected_shifts   # (N,1,3)

            # 应用偏移
            pos = center.expand(N, A, 3) + sep                  # (N,A,3)

        else:
            # === 评估模式 (Evaluation) ===
            pos = torch.zeros(N, A, 3, device=self.device)
            
            # 计算基准 X 坐标 (均匀排列)
            base_x = (env_ids.float() / self.num_envs - 0.5) * 32.0  # (N,)
            
            pos[..., 0] = base_x.unsqueeze(-1)  # (N,A) 此时 Agent 0 和 1 重叠
            pos[..., 1] = 24.0
            pos[..., 2] = 2.0

            # ✅ 此时 sep 已经被定义了，不会再报错
            pos = pos + sep

        # 4. 后续通用逻辑 (目标点、旋转、写入仿真等)
        # ====== 保持不变 ======
        target = self.target_pos[env_ids]
        if target.dim() == 3 and target.shape[1] == 1 and A != 1:
            target = target.expand(N, A, 3)
        elif target.dim() == 2:
            target = target.unsqueeze(1).expand(N, A, 3)

        # target_dir / yaw
        diff = target - pos                                      
        self.target_dir[env_ids] = target - pos  

        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])     
        rpy = torch.zeros(N, A, 3, device=self.device)
        rpy[..., 2] = facing_yaw
        rot = euler_to_quaternion(rpy)                           
        
        # 检查坏点 (防止 NaN)
        bad = (~torch.isfinite(pos).all(dim=-1)) | (~torch.isfinite(rot).all(dim=-1))
        if bad.any():
            print(f"[ERROR] Invalid pose detected during reset! Resetting to safe default.")
            # 简单的 Fallback 策略：重置回带偏移的原点
            pos[bad] = sep[bad] + torch.tensor([0,0,2.0], device=self.device)
            
        # 写入仿真
        vel = torch.zeros(N, A, 6, device=self.device)
        self.drone.set_world_poses(pos.contiguous(), rot.contiguous(), env_ids)
        self.drone.set_velocities(vel.contiguous(), env_ids)

        self.prev_drone_vel_w[env_ids] = 0.0

        self.height_range[env_ids, :, 0] = torch.minimum(pos[..., 2], target[..., 2])
        self.height_range[env_ids, :, 1] = torch.maximum(pos[..., 2], target[..., 2])

        self.stats[env_ids] = 0.
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        
        # print(f'apply前{actions}')
        self.drone.apply_action(actions)
    # def _post_sim_step(self, tensordict: TensorDictBase):  # 物理步进后的操作
    #     if (self.cfg.env_dyn.num_obstacles != 0):  # 如果有动态障碍物
    #         self.move_dynamic_obstacle()  # 移动它们
    #     self.lidar.update(self.dt)  # 更新 LiDAR 数据
    
    def _post_sim_step(self, tensordict: TensorDictBase):
            # if (self.cfg.env_dyn.num_obstacles != 0):
            #     self.move_dynamic_obstacle()
            self.lidar.update(self.dt)
    def _compute_state_and_obs(self):
        E = self.num_envs
        A = self.num_agents  # 或 A = self.drone.n，但要和你的 spec/agent_spec 对齐
        W, H = self.lidar_resolution
        K = self.cfg.algo.feature_extractor.dyn_obs_num

        # ============================================================
        # 0) Drone state
        # ============================================================
        # 期望: (E,A, state_dim)
        self.root_state = self.drone.get_state(env_frame=False)
        # info: (E,A,13)
        self.info["drone_state"][:] = self.root_state[..., :13]

        # ============================================================
        # 1) LiDAR scan -> (E,A,W,H) （推荐不加“1”通道维）
        # ============================================================
        pos_w = self.lidar.data.pos_w        # (E*A,3) or (E,A,3)
        hits_w = self.lidar.data.ray_hits_w  # (E*A,R,3) or (E,A,R,3)

        if pos_w.dim() == 2:  # (E*A,3) -> (E,A,3)
            pos_w = pos_w.view(E, A, 3)

        if hits_w.dim() == 3:  # (E*A,R,3) -> (E,A,R,3)
            R = hits_w.shape[1]
            hits_w = hits_w.view(E, A, R, 3)
        else:
            R = hits_w.shape[-2]

        dist = (hits_w - pos_w.unsqueeze(-2)).norm(dim=-1).clamp_max(self.lidar_range)  # (E,A,R)
        self.lidar_scan = (self.lidar_range - dist).view(E, A, W, H)                    # (E,A,W,H)

        # Optional render (只看 env0 agent0)
        if self._should_render(0):
            self.debug_draw.clear()
            x = pos_w[0, 0]  # (3,)
            v = (hits_w[0, 0] - x).reshape(W, H, 3)
            self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])

        # ============================================================
        # 2) Drone internal state (全部 (E,A,*))
        # ============================================================
        rpos = self.target_pos - self.root_state[..., :3]        # (E,A,3)
        distance = rpos.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # (E,A,1)
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)   # (E,A,1)
        distance_z = rpos[..., 2:3]                              # (E,A,1)

        # unit goal direction in 2D
        target_dir_2d = self.target_dir.clone()                  # (E,A,3)
        target_dir_2d[..., 2] = 0                                # Z=0，仅保留水平朝向

        rpos_unit = rpos / distance                              # (E,A,3)
        rpos_unit_g = vec_to_new_frame(rpos_unit, target_dir_2d)  # (E,A,3)

        vel_w = self.root_state[..., 7:10]                       # (E,A,3)
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)           # (E,A,3)

        # 你原来 squeeze(1) 是单机写法，多机不能 squeeze
        # 这里保持 (E,A, obs_dim)
        drone_state = torch.cat([rpos_unit_g, distance_2d, distance_z, vel_g], dim=-1)  # (E,A,3+1+1+3=8)

        # ============================================================
        # 3) Dynamic obstacles -> dyn_obs_states: (E,A, K, 10)
        # ============================================================
        if self.cfg.env_dyn.num_obstacles != 0:
            obs_pos = self.dyn_obs_state[:, :3]  # (O,3)
            O = obs_pos.shape[0]

            drone_pos = self.root_state[..., :3]                               # (E,A,3)
            obs_pos_e = obs_pos.view(1, 1, O, 3).expand(E, A, O, 3)            # (E,A,O,3)
            rpos_obs = obs_pos_e - drone_pos.unsqueeze(-2)                     # (E,A,O,3)

            dist2d = torch.norm(rpos_obs[..., :2], dim=-1)                     # (E,A,O)
            _, idx = torch.topk(dist2d, K, dim=-1, largest=False)              # (E,A,K)

            idx3 = idx.unsqueeze(-1).expand(E, A, K, 3)                        # (E,A,K,3)
            closest_rpos = torch.gather(rpos_obs, dim=-2, index=idx3)          # (E,A,K,3)

            closest_dist = closest_rpos.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # (E,A,K,1)
            closest_rpos_gn = closest_rpos / closest_dist                           # (E,A,K,3)

            closest_dist2d = torch.norm(closest_rpos[..., :2], dim=-1, keepdim=True)  # (E,A,K,1)
            closest_distz  = closest_rpos[..., 2:3]                                    # (E,A,K,1)

            obs_vel = self.dyn_obs_vel  # (O,3)
            obs_vel_e = obs_vel.view(1, 1, O, 3).expand(E, A, O, 3)
            closest_vel = torch.gather(obs_vel_e, dim=-2, index=idx3)                # (E,A,K,3)

            # width/height category（你如果有原逻辑可替换这里）
            width_cat  = torch.zeros(E, A, K, 1, device=self.device)
            height_cat = torch.zeros(E, A, K, 1, device=self.device)

            dyn_obs_states = torch.cat(
                [closest_rpos_gn, closest_dist2d, closest_distz, closest_vel, width_cat, height_cat],
                dim=-1
            )  # (E,A,K,10)

            # 动态碰撞判定（示例：用“最近K个里是否有太近”）
            # 你可以用更严格：2D+Z 同时满足
            dynamic_collision = (closest_dist2d <= 0.3).any(dim=-2, keepdim=True)  # (E,A,1)

            # reward 用的距离（示例）
            closest_dyn_obs_distance_reward = closest_dist.squeeze(-1)  # (E,A,K)
        else:
            dyn_obs_states = torch.zeros(E, A, K, 10, device=self.device)
            dynamic_collision = torch.zeros(E, A, 1, dtype=torch.bool, device=self.device)
            closest_dyn_obs_distance_reward = None

        # ============================================================
        # 4) Obs dict (注意：和你的 observation_spec 对齐)
        # ============================================================
        obs = {
            "state": drone_state,          # (E,A,8)
            "lidar": self.lidar_scan,      # (E,A,W,H)  <- spec 也要是 (W,H)
            "direction": target_dir_2d,    # (E,A,3)
            # "dynamic_obstacle": dyn_obs_states,  # 如果你 spec 里启用了就打开
        }

        # ============================================================
        # 5) Reward (per-agent) -> (E,A,1)
        # ============================================================
        # static safety: (E,A,1)
        reward_safety_static = torch.log(
            (self.lidar_range - self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)
        ).mean(dim=(-1, -2), keepdim=True).squeeze(-1)  # (E,A,1)

        if self.cfg.env_dyn.num_obstacles != 0:
            # (E,A,K) -> (E,A,1)
            reward_safety_dynamic = torch.log(
                closest_dyn_obs_distance_reward.clamp(min=1e-6, max=self.lidar_range)
            ).mean(dim=-1, keepdim=True)  # (E,A,1)
        else:
            reward_safety_dynamic = 0.0

        vel_direction = rpos / distance                     # (E,A,3)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(dim=-1, keepdim=True)  # (E,A,1)

        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1, keepdim=True)  # (E,A,1)

        # height penalty: (E,A,1)
        z = self.drone.pos[..., 2:3]
        penalty_height = torch.zeros(E, A, 1, device=self.device)
        too_high = z > (self.height_range[..., 1:2] + 0.2)
        too_low  = z < (self.height_range[..., 0:1] - 0.2)
        penalty_height[too_high] = (z - self.height_range[..., 1:2] - 0.2)[too_high] ** 2
        penalty_height[too_low]  = (self.height_range[..., 0:1] - 0.2 - z)[too_low] ** 2

        # collision (per-agent): static + dynamic
        static_collision = (self.lidar_scan.max(dim=-1).values.max(dim=-1).values > (self.lidar_range - 0.3)).unsqueeze(-1)  # (E,A,1)
        collision_agent = static_collision | dynamic_collision  # (E,A,1)
        E = self.num_envs
        A = self.num_agents

        # def shp(name, x):
        #     if torch.is_tensor(x):
        #         print(name, x.shape)
        #     else:
        #         print(name, type(x), x)

        # shp("reward_vel", reward_vel)
        # shp("reward_safety_static", reward_safety_static)
        # shp("reward_safety_dynamic", reward_safety_dynamic)
        # shp("penalty_smooth", penalty_smooth)
        # shp("penalty_height", penalty_height)

        # final reward
        if self.cfg.env_dyn.num_obstacles != 0:
            self.reward = reward_vel + 1.0 + reward_safety_static + reward_safety_dynamic \
                        - 0.1 * penalty_smooth - 8.0 * penalty_height        # (E,A,1)
        else:
            self.reward = reward_vel + 1.0 + reward_safety_static \
                        - 0.1 * penalty_smooth - 8.0 * penalty_height        # (E,A,1)

        # ============================================================
        # 6) Done (env-level) -> (E,1)
        # ============================================================
        reach_goal = (distance < 0.5)                       # (E,A,1)
        below_bound = (z < 0.2)                              # (E,A,1)
        above_bound = (z > 4.0)                              # (E,A,1)
        terminated_agent = below_bound | above_bound | collision_agent  # (E,A,1)

        # ✅ env-level：任意一架撞/越界 -> env done
        self.terminated = terminated_agent.any(dim=1)        # (E,1)

        # ✅ env-level：时间到
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)  # (E,1)

        # smoothness cache
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()  # (E,A,3)

        # stats（你想 per-agent 也可以，这里先保持原来 env 级别）
        # 如果 stats["return"] spec 是 (E,A) 那就用 self.reward.squeeze(-1)
        if self.stats["return"].dim() == 2 and self.stats["return"].shape[-1] == A:
            self.stats["return"][...] += self.reward.squeeze(-1)   # (E,A)
        else:
            self.stats["return"] += self.reward.mean(dim=1)        # (E,1)

        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)  # (E,1)
        self.stats["reach_goal"] = reach_goal.any(dim=1).float()       # (E,1)
        self.stats["collision"] = collision_agent.any(dim=1).float()   # (E,1)
        self.stats["truncated"] = self.truncated.float()               # (E,1)

        # ============================================================
        # 7) Return TensorDict
        # ============================================================
        obs_td = TensorDict(obs, batch_size=[E, A])   # ✅ 只有 observation 是 [E,A]

        return TensorDict(
            {
                "agents": {                           # ✅ agents 用 dict（或 TD 但 batch_size=[E]）
                    "observation": obs_td,
                },
                "stats": self.stats.clone(),          # stats batch_size 必须是 [E]
                "info": self.info,                    # info batch_size 必须是 [E]
            },
            batch_size=[E],
        )


    def _compute_reward_and_done(self):
        E = self.num_envs
        A = self.num_agents

        reward = self.reward          # 原本是 (E,A,1)

        # ✅ 统一：环境里输出 (E,A)
        if reward.dim() == 3 and reward.shape[-1] == 1:
            reward = reward.squeeze(-1)    # -> (E,A)

        terminated = self.terminated       # (E,1)
        truncated  = self.truncated        # (E,1)
        done = terminated | truncated      # (E,1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward,      # ✅ (E,A)
                },
                "done": done,              # (E,1)
                "terminated": terminated,  # (E,1)
                "truncated": truncated,    # (E,1)
            },
            batch_size=[E],
        )
