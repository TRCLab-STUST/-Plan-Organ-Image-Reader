from PIL import Image
import numpy as np
import cv2
import csv
import os


class OrganImageReader:
    def __init__(self, debug=False):
        self.debug = debug
        self.organ_list = []
        self.organ_rgb_list = []
        self.organ_rgb_set = []
        self.image_origin_rgb_set = []
        self.find_organ_set = []
        self.contours = None
        self.image = None
        self.image_origin = None
        self.image_filter = None
        self.image_size = None
        self.size = None
        self.black = None

    def load_table(self, table_path, delim='\t'):
        # 開啟檔案 color.txt
        table_file = open(table_path)
        # 使用 csv 讀取器讀取檔案
        table = csv.reader(table_file, delimiter=delim)
        # 清空資料
        self.organ_list.clear()
        self.organ_rgb_list.clear()
        self.organ_rgb_set.clear()
        # 讀取每一行，並添加進相對應得空間內
        for row in table:
            self.organ_list.append(row)
            self.organ_rgb_list.append([int(row[1]), int(row[2]), int(row[3])])
            self.organ_rgb_set.append((int(row[1]), int(row[2]), int(row[3])))
        # 刪除頭尾資料
        del self.organ_list[0]
        del self.organ_list[-1]
        del self.organ_rgb_list[0]
        del self.organ_rgb_list[-1]
        del self.organ_rgb_set[0]
        del self.organ_rgb_set[-1]
        # 創建為集合(set)
        self.organ_rgb_set = set(self.organ_rgb_set)
        # 顯示讀取資料
        self.logger_send('顯示讀取到的表格(organ_table)資料')
        self.logger_send('len(organ_list)=', len(self.organ_list))
        self.logger_send('organ_list[0]=', self.organ_list[0])
        self.logger_send('organ_rgb_list[0]=', self.organ_rgb_list[0])

    def load_image(self, image_path):
        # 使用 Pillow 讀取圖片
        self.image = Image.open(image_path)
        # 讀取圖片大小(Bytes)
        self.image_size = os.path.getsize(image_path)
        # 把圖片轉為 RGB 陣列存起來
        image_rgb = np.array(self.image)
        # 因為 OpenCV 是使用 BGR 所以把圖片轉為 BGR 色調
        self.image_origin = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        # 取得圖片裡每個像素的顏色
        color = []
        for i in image_rgb:
            for j in i:
                j = list(j)
                color.append((j[0], j[1], j[2]))
        # 轉存為集合(set)
        self.image_origin_rgb_set = set(color)
        # 顯示圖片資料
        self.logger_send('顯示讀取到的圖片(image_origin)資料')
        self.logger_send('len(image_origin_rgb_set)=', len(self.image_origin_rgb_set))
        self.logger_send('image_origin_rgb_set=', self.image_origin_rgb_set)
        self.logger_send('image_size(bytes)=', self.image_size)
        self.size = (np.size(self.image, 0), np.size(self.image, 1), 3)
        self.black = np.zeros(self.size, np.uint8)
        self.black.fill(0)

    def find_organ(self):
        # 找到表格與圖片顏色的交集
        find_organ_set = set(self.organ_rgb_set).intersection(set(self.image_origin_rgb_set))
        mask_ = []
        for mask in find_organ_set:
            mask_.append([*mask, ])
        self.find_organ_set = mask_
        self.logger_send('顯示圖片裡找到的所有器官')
        self.logger_send('find_organ_rgb_set=', self.find_organ_set)
        for xyz in self.find_organ_set:
            index = self.organ_rgb_list.index(xyz)
            self.logger_send('index=', index, 'organ name=', self.organ_list[index][0])
        return self.find_organ_set

    def filter_organ(self, index):
        image = np.array(self.image)
        for y in range(np.size(image, 1)):
            for x in range(np.size(image, 0)):
                if image[x, y, 0] == self.organ_rgb_list[index][0] \
                        and image[x, y, 1] == self.organ_rgb_list[index][1] \
                        and image[x, y, 2] == self.organ_rgb_list[index][2]:
                    image[x, y] = 255
                else:
                    image[x, y] = 0
        self.image_filter = image

    def draw_contours(self, color):
        image = np.array(self.image_filter)
        image_gray = cv2.cvtColor(self.image_filter, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image_gray, 127, 255, 0)
        self.contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger_send('Number of contours=' + str(len(self.contours)))
        if len(self.contours) == 0:
            return None
        self.logger_send('contours[0]=', self.contours[0])
        cv2.drawContours(self.black, self.contours, -1, (color[2], color[1], color[0]), 2)

    def logger_send(self, msg, *args):
        if self.debug:
            for arg in args:
                msg += " " + str(arg)
            print(msg)
