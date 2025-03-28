import cv2
import numpy as np
from vcam import vcam, meshGen
from common.utils import Util
from common.oss import OSS

image_oss = OSS("hz-images")


# 旋转90度
def rotation(img):
    rot90_img = np.rot90(img)
    return rot90_img


# 水平翻转
def H_flip(img):
    h_img = cv2.flip(img, 1)
    return h_img


# 垂直翻转
def V_flip(img):
    v_img = cv2.flip(img, 0)
    return v_img


def swirl(xy, x0, y0, R):
    r = np.sqrt((xy[:, 1] - x0) ** 2 + (xy[:, 0] - y0) ** 2)
    a = np.pi * r / R
    xy[:, 1] = (xy[:, 1] - x0) * np.cos(a) + (xy[:, 0] - y0) * np.sin(a) + x0
    xy[:, 0] = -(xy[:, 1] - x0) * np.sin(a) + (xy[:, 0] - y0) * np.cos(a) + y0
    return xy


# 变形
def warp_Affine(img):
    H, W = img.shape[:2]
    # Creating the virtual camera object
    c1 = vcam(H=H, W=W)
    # Creating the surface object
    plane = meshGen(H, W)
    plane.Z += 20 * np.sin(2 * np.pi * ((plane.X - plane.W / 4.0) / plane.W)) + 20 * np.sin(
        2 * np.pi * ((plane.Y - plane.H / 4.0) / plane.H))
    # Extracting the generated 3D plane
    pts3d = plane.getPlane()
    # Projecting (Capturing) the plane in the virtual camera
    pts2d = c1.project(pts3d)
    # Deriving mapping functions for mesh based warping.
    map_x, map_y = c1.getMaps(pts2d)
    # Generating the output
    output = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    output = cv2.flip(output, 1)

    return output


# 加噪
def add_noise(img):
    mean = 0
    var = 0.003
    image = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 正态分布
    noise_img = image + noise
    if noise_img.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    noise_img = np.clip(noise_img, low_clip, 1.0)
    noise_img = np.uint8(noise_img * 255)
    return noise_img


def dataAugmentMain(local_path):
    img = cv2.imread(local_path)
    suffix = Util.extract_file_suffix(local_path)
    img_path = Util.generate_temp_file_path(suffix)

    img2 = rotation(img)
    img3 = H_flip(img)
    img4 = V_flip(img)
    img5 = warp_Affine(img)
    img6 = add_noise(img)

    cv2.imwrite(img_path, img2)
    remote_rotation_url = image_oss.upload(img_path)
    cv2.imwrite(img_path, img3)
    remote_H_flip_url = image_oss.upload(img_path)
    cv2.imwrite(img_path, img4)
    remote_V_flip_url = image_oss.upload(img_path)
    cv2.imwrite(img_path, img5)
    remote_warp_affine_url = image_oss.upload(img_path)
    cv2.imwrite(img_path, img6)
    remote_add_noise_url = image_oss.upload(img_path)

    return {
        "remote_rotation_url": remote_rotation_url,
        "remote_H_flip_url": remote_H_flip_url,
        "remote_V_flip_url": remote_V_flip_url,
        "remote_warp_affine_url": remote_warp_affine_url,
        "remote_add_noise_url": remote_add_noise_url,
    }
