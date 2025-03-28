import numpy as np
import open3d as o3d

obj1 = {"pos": np.array([97, 202, 62]), "pitch": -1.7801286448003404,
        "yaw": -0.6774866097113286}  # pitch_degree -101.99385833740234 yaw_degree -38.81712341308594
obj2 = {"pos": np.array([297, 233, 160]), "pitch": -1.0303932710343067,
        "yaw": 0.07750784141733912}  # pitch_degree -59.03718566894531 yaw_degree 4.440872192382812
obj3 = {"pos": np.array([634, 221, 132]), "pitch": 1.3177827074319186,
        "yaw": -0.2537004486081095}  # pitch_degree 75.50338745117188 yaw_degree -14.535964965820314
obj4 = {"pos": np.array([542, 102, 195]), "pitch": 0.476214215586955,
        "yaw": -0.2643914424881236}  # pitch_degree 27.28506469726562 -yaw_degree 15.148513793945314

FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=700, origin=[0, 0, 0])
sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=10)
sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=10)
sphere3 = o3d.geometry.TriangleMesh.create_sphere(radius=10)
sphere4 = o3d.geometry.TriangleMesh.create_sphere(radius=10)
sphere1.translate(obj1["pos"])
sphere2.translate(obj2["pos"])
sphere3.translate(obj3["pos"])
sphere4.translate(obj4["pos"])

vec_len = 700


def getArrow(center, pitch, yaw):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_height=0.1 * vec_len,
        cylinder_radius=0.005 * vec_len,
        cone_height=0.008 * vec_len,
        cone_radius=0.01 * vec_len,
    )
    arrow.translate(center)
    R = arrow.get_rotation_matrix_from_xyz((np.pi+yaw, -pitch, 0))
    arrow.rotate(R, center=center)
    arrow.paint_uniform_color([1, 0, 0])
    return arrow


arrow4 = getArrow(obj4["pos"], obj4["pitch"], obj4["yaw"])
arrow3 = getArrow(obj3["pos"], obj3["pitch"], obj3["yaw"])
arrow2 = getArrow(obj2["pos"], obj2["pitch"], obj2["yaw"])
arrow1 = getArrow(obj1["pos"], obj1["pitch"], obj1["yaw"])

o3d.visualization.draw_geometries([FOR, sphere1, sphere2, sphere3, sphere4, arrow4, arrow3, arrow2, arrow1])


# 有缺陷 由于深度信息的问题，导致在尺度上不一样， 这个只能作为二维的gaze估计