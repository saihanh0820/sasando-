import pyvista as pv
import numpy as np

# -------------------------- 1. 基础参数定义（严格匹配论文+传统外观）--------------------------
# 竹管参数（论文1.1/6.2.1 + 传统竹材特征）
bamboo_length = 0.7  # 竹管长度70cm
bamboo_outer_d = 0.08  # 外径8cm
bamboo_inner_d = 0.065  # 内径6.5cm
bamboo_color = [0.82, 0.71, 0.55]  # 自然竹色（偏黄棕）
bamboo_texture_scale = 0.1  # 竹节纹理缩放

# Lontar叶片谐振器参数（论文4.1.1 + 干燥棕榈叶特征）
resonator_dia = 0.35  # 谐振器直径35cm
resonator_radius = resonator_dia / 2
curvature_radius = 0.1  # 抛物面曲率半径10cm
edge_thickness = 0.0012  # 边缘厚度1.2mm
center_thickness = 0.0025  # 中心厚度2.5mm
resonator_color = [0.95, 0.92, 0.88]  # 干燥叶片色（浅米色）
resonator_vein_color = [0.85, 0.75, 0.65]  # 叶片叶脉色（深米色）

# 琴弦参数（传统sasando 12弦+银色金属）
string_count = 12  # 琴弦数量
string_radius = 0.0008  # 琴弦半径0.8mm
string_length = bamboo_length  # 琴弦长度=竹管长度
string_color = [0.95, 0.95, 0.95]  # 银色琴弦（带反光）

# 拾音器参数（论文5.1 + 工业塑料质感）
pickup_dia = 0.03  # 拾音器直径3cm
pickup_length = 0.05  # 拾音器长度5cm
pickup_pos = bamboo_length * 0.8  # 安装在竹管80%长度处
pickup_color = [0.3, 0.3, 0.3]  # 哑光黑

# 控制面板参数（论文5.1 + 防滑橡胶材质）
panel_width = 0.1  # 面板宽度10cm
panel_height = 0.05  # 高度5cm
panel_thickness = 0.02  # 厚度2cm
panel_pos = bamboo_length * 0.5  # 竹管中部
panel_color = [0.2, 0.2, 0.2]  # 亮黑（带3个按钮凹槽）
button_color = [0.5, 0.5, 0.5]  # 按钮色（浅灰）

# -------------------------- 2. 部件创建（新增纹理+真实材质）--------------------------
def create_bamboo():
    """创建带竹节纹理的中空竹管"""
    # 外圆柱（带竹节纹理）
    outer_cyl = pv.Cylinder(
        center=(0, 0, bamboo_length / 2),
        direction=(0, 0, 1),
        radius=bamboo_outer_d / 2,
        height=bamboo_length,
        resolution=80
    )
    outer_cyl = outer_cyl.triangulate()
    
    # 内圆柱（中空部分）
    inner_cyl = pv.Cylinder(
        center=(0, 0, bamboo_length / 2),
        direction=(0, 0, 1),
        radius=bamboo_inner_d / 2,
        height=bamboo_length + 0.001,
        resolution=80
    )
    inner_cyl = inner_cyl.triangulate()
    
    # 布尔运算得到中空竹管
    bamboo = outer_cyl.boolean_difference(inner_cyl)
    
    # 分配竹色（每个单元对应颜色）
    cell_count = bamboo.n_cells
    color_array = np.tile(bamboo_color, (cell_count, 1))
    
    # 添加竹节纹理（基于单元中心的极角和Z坐标）
    cell_centers = bamboo.cell_centers().points
    theta = np.arctan2(cell_centers[:, 1], cell_centers[:, 0])
    z = cell_centers[:, 2]
    
    # 纹理 = 极角+Z轴周期变化（模拟竹节）
    texture = (np.sin(theta * 12) + np.sin(z / bamboo_texture_scale)) / 4 + 0.75
    texture = np.clip(texture, 0.6, 0.9)
    
    # 纹理叠加到颜色
    for i in range(3):
        color_array[:, i] *= texture
    
    bamboo.cell_data["Color"] = color_array
    return bamboo

def create_resonator():
    """创建带叶脉纹理的lontar叶片谐振器"""
    # 使用旋转曲面创建抛物面
    r = np.linspace(0, resonator_radius, 80)
    z = (r**2) / (2 * curvature_radius)
    z = z - z.max()
    
    # 创建旋转曲面
    profile_points = np.column_stack((r, np.zeros_like(r), z))
    profile = pv.PolyData(profile_points)
    resonator_surface = profile.revolve_sweep(angle=360, resolution=120)
    
    # 赋予厚度
    thickness_profile = edge_thickness + (center_thickness - edge_thickness) * (1 - r / resonator_radius)
    thick_resonator = resonator_surface.extrude((0, 0, center_thickness), capping=True)
    
    # 清理并三角化
    resonator = thick_resonator.clean().triangulate()
    
    # 分配基础颜色
    cell_count = resonator.n_cells
    color_array = np.tile(resonator_color, (cell_count, 1))
    
    # 添加叶脉纹理
    cell_centers = resonator.cell_centers().points
    r_cell = np.linalg.norm(cell_centers[:, :2], axis=1)
    theta_cell = np.arctan2(cell_centers[:, 1], cell_centers[:, 0])
    
    # 叶脉纹理计算
    vein_texture = (np.sin(r_cell / 0.015) + np.sin(theta_cell * 12)) / 4 + 0.8
    vein_texture = np.clip(vein_texture, 0.7, 0.95)
    
    # 叶脉区域颜色分配
    vein_mask = vein_texture < 0.82
    for i in range(3):
        color_array[vein_mask, i] = resonator_vein_color[i]
        color_array[~vein_mask, i] *= vein_texture[~vein_mask]
    
    resonator.cell_data["Color"] = color_array
    
    # 移动到竹管一端
    resonator.translate([0, 0, -bamboo_length/2 - center_thickness/2])
    return resonator

def create_strings():
    """创建带金属反光的银色琴弦"""
    strings = []
    string_dist_radius = bamboo_inner_d / 2 - 0.002
    
    for i in range(string_count):
        angle = 2 * np.pi * i / string_count
        start = [string_dist_radius * np.cos(angle), string_dist_radius * np.sin(angle), -bamboo_length/2]
        end = [string_dist_radius * np.cos(angle), string_dist_radius * np.sin(angle), bamboo_length/2]
        
        # 创建琴弦
        string = pv.Cylinder(
            center=((start[0]+end[0])/2, (start[1]+end[1])/2, (start[2]+end[2])/2),
            direction=(0, 0, 1),
            radius=string_radius,
            height=string_length,
            resolution=16
        )
        
        # 分配银色带反光
        cell_count = string.n_cells
        color_array = np.tile(string_color, (cell_count, 1))
        
        # 反光效果
        z_pos = string.points[:, 2]
        reflect = (np.sin(z_pos / 0.08) + 1) / 4 + 0.75
        reflect = reflect[:cell_count]
        
        for j in range(3):
            color_array[:, j] *= reflect
        
        string.cell_data["Color"] = color_array
        strings.append(string)
    
    return strings

def create_pickup():
    """创建哑光黑拾音器（带金属接口细节）"""
    pickup = pv.Cylinder(
        center=(bamboo_outer_d/2 + pickup_length/2, 0, pickup_pos - bamboo_length/2),
        direction=(1, 0, 0),
        radius=pickup_dia/2,
        height=pickup_length,
        resolution=40
    )
    pickup = pickup.triangulate()
    
    # 分配哑光黑色
    cell_count = pickup.n_cells
    color_array = np.tile(pickup_color, (cell_count, 1))
    
    # 接口细节
    x_pos = pickup.points[:, 0]
    edge_bright = np.where(
        np.abs(x_pos - (bamboo_outer_d/2 + pickup_length/2)) > pickup_length/2 - 0.003,
        1.15,
        1.0
    )
    
    for i in range(3):
        color_array[:, i] *= edge_bright[:cell_count]
    
    pickup.cell_data["Color"] = color_array
    return pickup

def create_panel():
    """创建带按钮的控制面板"""
    # 面板主体
    panel = pv.Box(
        bounds=[
            bamboo_outer_d/2, bamboo_outer_d/2 + panel_thickness,
            -panel_width/2, panel_width/2,
            panel_pos - bamboo_length/2 - panel_height/2, panel_pos - bamboo_length/2 + panel_height/2
        ]
    )
    panel = panel.triangulate()
    
    # 创建3个按钮
    buttons = []
    button_radius = 0.008
    button_height = 0.005
    button_positions = [-panel_width/4, 0, panel_width/4]
    
    for btn_y in button_positions:
        btn_center = (
            bamboo_outer_d/2 + panel_thickness/2,
            btn_y,
            panel_pos - bamboo_length/2
        )
        btn = pv.Cylinder(
            center=btn_center,
            direction=(1, 0, 0),
            radius=button_radius,
            height=button_height,
            resolution=20
        )
        btn = btn.triangulate()
        buttons.append(btn)
    
    # 合并所有网格
    combined_mesh = panel.copy()
    for btn in buttons:
        combined_mesh = combined_mesh.merge(btn)
    
    # 手动创建颜色数组
    total_cells = combined_mesh.n_cells
    combined_colors = np.zeros((total_cells, 3))
    
    # 分配面板颜色
    panel_cells = panel.n_cells
    panel_color_array = np.tile(panel_color, (panel_cells, 1))
    combined_colors[:panel_cells] = panel_color_array
    
    # 分配按钮颜色
    current_cell = panel_cells
    for btn in buttons:
        btn_cells = btn.n_cells
        button_color_array = np.tile(button_color, (btn_cells, 1))
        combined_colors[current_cell:current_cell + btn_cells] = button_color_array
        current_cell += btn_cells
    
    combined_mesh.cell_data["Color"] = combined_colors
    return combined_mesh

# -------------------------- 3. 学术级可视化（带光照+规范标注）--------------------------
def main():
    # 创建所有部件
    bamboo = create_bamboo()
    resonator = create_resonator()
    strings = create_strings()
    pickup = create_pickup()
    panel = create_panel()
    
    # 初始化可视化窗口
    p = pv.Plotter(window_size=[1600, 1000])
    
    # 添加部件（移除所有label参数）
    p.add_mesh(
        bamboo,
        scalars="Color",
        rgb=True,
        specular=0.4,
        diffuse=0.8,
        ambient=0.3
    )
    
    p.add_mesh(
        resonator,
        scalars="Color",
        rgb=True,
        specular=0.2,
        diffuse=0.9,
        ambient=0.4
    )
    
    for i, string in enumerate(strings):
        p.add_mesh(
            string,
            scalars="Color",
            rgb=True,
            specular=0.9,
            diffuse=0.4,
            ambient=0.2
        )
    
    p.add_mesh(
        pickup,
        scalars="Color",
        rgb=True,
        specular=0.3,
        diffuse=0.7,
        ambient=0.3
    )
    
    p.add_mesh(
        panel,
        scalars="Color",
        rgb=True,
        specular=0.5,
        diffuse=0.6,
        ambient=0.4
    )
    
    # 移除图例
    p.add_axes()
    
    p.add_title(
        "3D Model of Electric Sasando",
        font_size=16
    )
    
    # 光照优化
    p.add_light(pv.Light(position=(2, 2, 2), intensity=0.8))
    p.add_light(pv.Light(position=(-2, -1, 1), intensity=0.4))
    p.add_light(pv.Light(position=(0, -2, 0), intensity=0.3))
    
    p.set_background([0.95, 0.95, 0.98])
    
    # 相机设置
    p.camera_position = [
        (1.5, -1.5, 0.8),
        (0, 0, 0),
        (0, 0, 1)
    ]
    
    # 显示模型
    p.show()

if __name__ == "__main__":
    main()