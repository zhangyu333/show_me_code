# coding=utf-8
# Created : 2022/12/13 8:22
# Author  : Zy
import cv2
import time
import numpy as np
from tqdm import tqdm
from common.oss import OSS
from matplotlib import path
from common.utils import Util
from typing import Tuple, Any
from common.file_utils import clearCache
from controller.detect.vehicle_count.detect import VehicleDetection

image_oss = OSS("hz-images")


class Vehicle:
    def __init__(self, vehicleid, vehicle_cx, vehicle_cy,
                 vehicle_start_time, vehicle_over_time,
                 vehicle_v, vehicle_type, box):
        self.vehicleid = vehicleid
        self.vehicle_cx = vehicle_cx
        self.vehicle_cy = vehicle_cy
        self.vehicle_start_time = vehicle_start_time
        self.vehicle_over_time = vehicle_over_time
        self.vehicle_type = vehicle_type
        self.vehicle_v = vehicle_v
        self.box = box
        self.tracks = []
        self.vehicle_count = 0

    def updateCoords(self, x, y, box):
        self.vehicle_cx = x
        self.vehicle_cy = y
        self.box = box


class VehicleCount:
    def __init__(self, video_sources, corners1=None, corners2=None):
        super(VehicleCount, self).__init__()
        self.vd = VehicleDetection()
        self.cap = cv2.VideoCapture(video_sources)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.size = (self.cap_w, self.cap_h)
        self.output_path = Util.generate_temp_file_path(suffix="mp4")
        self.videoWrite = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.mask = np.zeros((self.cap_h, self.cap_w, 3), dtype=np.uint8)
        self.corners1 = corners1
        self.corners2 = corners2
        assert self.corners1 or self.corners2, "you need to divide a area !!!!"
        self.corners = None
        if self.corners1 and not self.corners2:
            self.corners = self.corners1
        elif self.corners2 and not self.corners1:
            self.corners = self.corners2
        self.vehicle_nums_area = 0
        self.vehicle_nums_area1 = 0
        self.vehicle_nums_area2 = 0
        self.vehicles = []
        self.vid = 1

        self.color_area1 = [0, 200, 200]
        self.color_area2 = [0, 0, 200]
        self.vehicle_counts = {}

    def getCheckArea(self, mask: np.ndarray, corners: list) -> Tuple[Any, Any, Any]:
        if not corners:
            return np.nan, None, False
        mask_copy = mask.copy()
        mask_draw_copy = mask.copy()
        for i in range(len(corners)):
            if i == len(corners) - 1:
                cv2.line(mask_copy, corners[i], corners[0], [255, 255, 255], 2)
            else:
                cv2.line(mask_copy, corners[i], corners[i + 1], [255, 255, 255], 2)
        mask_copy = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask_copy, 0, 1)
        area = cv2.contourArea(contours[0])
        mask_draw_copy = cv2.drawContours(mask_draw_copy, contours, -1, [255, 255, 255], cv2.FILLED)
        mask_th = cv2.cvtColor(mask_draw_copy, cv2.COLOR_BGR2GRAY)
        indices = mask_th.astype(bool)
        return indices, area, True

    def checkVehicleInArea(self, point: tuple, area_corners: list) -> bool:
        if not area_corners:
            return False
        poly_path = path.Path(np.array(area_corners))
        return poly_path.contains_point(point)

    def process(self, frame, indices1, sign1, check_area1, indices2, sign2, check_area2, check_area):
        if not (sign1 and sign2):
            cv2.rectangle(frame, (0, 0), (600, 50), self.color_area1 if sign1 else self.color_area2, -1)
            cv2.putText(frame, f"VehicleCount:{self.vehicle_nums_area}",
                        (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        [255, 255, 255],
                        2)
            if sign1 and not sign2:
                frame[indices1] = frame[indices1] * 0.5 + np.array(self.color_area1) * 0.5
            elif sign2 and not sign1:
                frame[indices2] = frame[indices2] * 0.5 + np.array(self.color_area2) * 0.5
        else:
            cv2.rectangle(frame, (0, 0), (600, 50), self.color_area1, -1)
            cv2.rectangle(frame, (0, 50), (600, 100), self.color_area2, -1)
            cv2.putText(frame, f"VehicleCount:{self.vehicle_nums_area1}", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        [255, 255, 255],
                        2)
            cv2.putText(frame, f"VehicleCount:{self.vehicle_nums_area2}", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        [255, 255, 255],
                        2)
            frame[indices1] = frame[indices1] * 0.5 + np.array(self.color_area1) * 0.5
            frame[indices2] = frame[indices2] * 0.5 + np.array(self.color_area2) * 0.5

        vehicle_area = 0
        vehicle_area1 = 0
        vehicle_area2 = 0

        box_probs = self.vd.detect(frame)
        for box_prob in box_probs:
            x1, y1, x2, y2, label, socre = box_prob
            cx = x1 + (x2 - x1) / 2
            cy = y1 + (y2 - y1) / 2
            create = True
            sign_area1 = self.checkVehicleInArea((cx, cy), self.corners1)
            sign_area2 = self.checkVehicleInArea((cx, cy), self.corners2)
            if not (sign1 and sign2):
                vehicle_area += (x2 - x1) * (y2 - y1)
            else:
                if sign_area1:
                    vehicle_area1 += (x2 - x1) * (y2 - y1)
                if sign_area2:
                    vehicle_area2 += (x2 - x1) * (y2 - y1)
            for vehicle in self.vehicles:
                distance = np.sqrt(
                    pow(abs(vehicle.vehicle_cx - cx), 2) + pow(abs(vehicle.vehicle_cy - cy), 2))
                if distance < 50:
                    create = False
                    vehicle.updateCoords(cx, cy, [x1, y1, x2, y2])
                    vehicle.vehicle_count += 1
                    if vehicle.vehicle_count < 10:
                        vehicle.tracks.append((int(vehicle.vehicle_cx), int(vehicle.vehicle_cy)))
                    else:
                        vehicle_tracks = vehicle.tracks[1:]
                        vehicle_tracks.append((int(vehicle.vehicle_cx), int(vehicle.vehicle_cy)))
                        vehicle.tracks = vehicle_tracks
                    for track in vehicle.tracks:
                        cv2.circle(frame, track, 2, (0, 0, 255), -1)
                    v_box = vehicle.box
                    cv2.rectangle(
                        frame, (v_box[0], v_box[1]), (v_box[2], v_box[3]), (255, 0, 0)
                    )
                    cv2.putText(frame, f"vid:{vehicle.vehicleid}|{vehicle.vehicle_type}", (v_box[0], v_box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    vehicle.vehicle_over_time = time.time()
                    vehicle.vehicle_v = distance / (vehicle.vehicle_over_time - vehicle.vehicle_start_time)
                    if not sign_area1 and not sign_area2:
                        if vehicle in self.vehicles:
                            self.vehicles.remove(vehicle)
            if not sign_area1 and not sign_area2:
                continue
            if create:
                start_time = time.time()
                if not (sign1 and sign2):
                    vehicle = Vehicle(self.vid, cx, cy, start_time, 0, 0, label, [x1, y1, x2, y2])
                    self.vehicle_nums_area += 1
                    self.vid += 1
                    self.vehicles.append(vehicle)
                else:
                    if sign_area1:
                        vehicle = Vehicle(self.vid, cx, cy, start_time, 0, 0, label, [x1, y1, x2, y2])
                        self.vehicle_nums_area1 += 1
                    else:
                        vehicle = Vehicle(self.vid, cx, cy, start_time, 0, 0, label, [x1, y1, x2, y2])
                        self.vehicle_nums_area2 += 1
                    self.vid += 1
                    if label not in self.vehicle_counts.keys():
                        self.vehicle_counts[label] = 1
                    else:
                        self.vehicle_counts[label] += 1
                    self.vehicles.append(vehicle)

        if not (sign1 and sign2):
            if vehicle_area / check_area == 2 / 3:
                cv2.putText(frame, f"Status:crowd", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [255, 255, 255], 2)
            else:
                cv2.putText(frame, f"Status:normal", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [255, 255, 255], 2)
        else:
            if vehicle_area1 / check_area1 == 2 / 3:
                cv2.putText(frame, f"Status:crowd", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [255, 255, 255], 2)
            else:
                cv2.putText(frame, f"Status:normal", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [255, 255, 255], 2)
            if vehicle_area2 / check_area2 == 2 / 3:
                cv2.putText(frame, f"Status:crowd", (340, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [255, 255, 255], 2)
            else:
                cv2.putText(frame, f"Status:normal", (340, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [255, 255, 255], 2)

    def __call__(self):
        indices1, check_area1, sign1 = self.getCheckArea(self.mask, self.corners1)
        indices2, check_area2, sign2 = self.getCheckArea(self.mask, self.corners2)
        check_area = check_area1 if sign1 else check_area2
        with tqdm(range(self.total_frames)) as pbar:
            for _ in pbar:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.process(frame, indices1, sign1, check_area1, indices2, sign2, check_area2, check_area)
                self.videoWrite.write(frame)
                pbar.set_description("数据处理并保存中!!! ")

        self.cap.release()
        self.videoWrite.release()
        remote_result_url = image_oss.upload(self.output_path)
        clearCache(self.output_path)
        result = {
            "vehicle_counts": self.vehicle_counts,
            "remote_video_url": remote_result_url
        }
        return result
