import cv2
import numpy as np
import os
import time
from args import args
from chs import chs
from chars import eng

class MainRecognition:
    def __init__(self):
        self.SZ = args.Size  # 训练图片长宽
        self.MAX_WIDTH = args.MAX_WIDTH  # 原始图片最大宽度
        self.Min_Area = args.Min_Area # 车牌区域允许最大面积
        self.PROVINCE_START = args.PROVINCE_START
        self.provinces = args.provinces
        self.cardtype = args.cardtype
        self.Prefecture = args.Prefecture
        self.cfg = args.Pic_size
        
    #一、读取图片文件
    def __imreadex(self, filename):
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    def __point_limit(self, point):
        if point[0] < 0:
            point[0] = 0
        if point[1] < 0:
            point[1] = 0

    #二、利用投影法，根据设定的阈值和图片直方图，找出波峰，用于分隔字符
    def __find_waves(self, threshold, histogram):
        up_point = -1  # 上升点
        is_peak = False
        if histogram[0] > threshold:
            up_point = 0
            is_peak = True
        wave_peaks = []
        for i, x in enumerate(histogram):
            if is_peak and x < threshold:
                if i - up_point > 2:
                    is_peak = False
                    wave_peaks.append((up_point, i))
            elif not is_peak and x >= threshold:
                is_peak = True
                up_point = i
        if is_peak and up_point != -1 and i - up_point > 4:
            wave_peaks.append((up_point, i))
        return wave_peaks

    #三、根据找出的波峰，分隔图片，从而得到逐个字符图片
    def __seperate_card(self, img, waves):
        part_cards = []
        for wave in waves:
            part_cards.append(img[:, wave[0]:wave[1]])
        return part_cards

    # 缩小车牌边界
    def __accurate_place(self, card_img_hsv, limit1, limit2, color):
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        # col_num_limit = self.cfg["col_num_limit"]
        row_num_limit = self.cfg["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh, yl

    #四、图片预处理
    def __preTreatment(self, car_pic):
        if type(car_pic) == type(""):
            img = self.__imreadex(car_pic)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]
        if pic_width > self.MAX_WIDTH:
            resize_rate = self.MAX_WIDTH / pic_width
            img = cv2.resize(img, (self.MAX_WIDTH, int(pic_hight * resize_rate)),
                             interpolation=cv2.INTER_AREA)  # 图片分辨率调整
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
        img = cv2.filter2D(img, -1, kernel=kernel) # 锐化
        blur = self.cfg["blur"]
        # 中值去噪
        if blur > 0:
            img = cv2.bilateralFilter(img, 5,210,210)
        oldimg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

        kernel = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0);  # 与上一次开运算结果融合



        # 找到图像边缘
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
        img_edge = cv2.Canny(img_thresh, 100, 200)
        


        # 使用开运算和闭运算让图像边缘成为一个整体
        kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)  # 闭运算
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)  # 开运算
        


        #五、查找图像边缘整体形成的矩形区域，并找到车牌区域
        try:
            image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:#高版本返回参数为2
            contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.Min_Area]

        # 逐个排除不是车牌的矩形区域
        car_contours = []
        for cnt in contours:
            # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            # 选择宽大于高的区域
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio > 2 and wh_ratio < 5.5:
                car_contours.append(rect)

        #六、车牌区域矫正
        card_imgs = []
        for rect in car_contours:
            if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
                angle = 1
            else:
                angle = rect[2]
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除
            box = cv2.boxPoints(rect)
            heigth_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0]:
                    left_point = point
                if low_point[1] > point[1]:
                    low_point = point
                if heigth_point[1] < point[1]:
                    heigth_point = point
                if right_point[0] < point[0]:
                    right_point = point

            if left_point[1] <= right_point[1]:  # 正角度
                new_right_point = [right_point[0], heigth_point[1]]
                pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                self.__point_limit(new_right_point)
                self.__point_limit(heigth_point)
                self.__point_limit(left_point)
                card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_imgs.append(card_img)

            elif left_point[1] > right_point[1]:  # 负角度

                new_left_point = [left_point[0], heigth_point[1]]
                pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
                pts1 = np.float32([left_point, heigth_point, right_point])
                M = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
                self.__point_limit(right_point)
                self.__point_limit(heigth_point)
                self.__point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_imgs.append(card_img)


        #七、使用颜色定位进一步排除不是车牌的矩形
        colors = []
        for card_index, card_img in enumerate(card_imgs):
            green = yellow = blue = black = white = 0
            try:
                # 有转换失败的可能，原因来自于上面矫正矩形出错
                card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)

            except:
                card_img_hsv = None

            if card_img_hsv is None:
                continue
            row_num, col_num = card_img_hsv.shape[:2]
            card_img_count = row_num * col_num

            # 确定车牌颜色
            for i in range(row_num):
                for j in range(col_num):
                    H = card_img_hsv.item(i, j, 0)
                    S = card_img_hsv.item(i, j, 1)
                    V = card_img_hsv.item(i, j, 2)
                    if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                        yellow += 1
                    elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                        green += 1
                    elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                        blue += 1

                    if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                        black += 1
                    elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                        white += 1
            color = "no"
            limit1 = limit2 = 0
            if yellow * 2 >= card_img_count:
                color = "yellow"
                limit1 = 11
                limit2 = 34  # 有的图片有色偏偏绿
            elif green * 2 >= card_img_count:
                color = "green"
                limit1 = 35
                limit2 = 99
            elif blue * 2 >= card_img_count:
                color = "blue"
                limit1 = 100
                limit2 = 124  # 有的图片有色偏偏紫
            elif black + white >= card_img_count * 0.7:
                color = "bw"
            colors.append(color)
            if limit1 == 0:
                continue

            # 根据车牌颜色再定位，缩小边缘非车牌边界
            xl, xr, yh, yl = self.__accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            need_accurate = False
            if yl >= yh:
                yl = 0
                yh = row_num
                need_accurate = True
            if xl >= xr:
                xl = 0
                xr = col_num
                need_accurate = True
            card_imgs[card_index] = card_img[yl:yh, xl:xr] \
                if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh, xl:xr]
            if need_accurate:  # 可能x或y方向未缩小，需要再试一次
                card_img = card_imgs[card_index]
                card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
                xl, xr, yh, yl = self.__accurate_place(card_img_hsv, limit1, limit2, color)
                if yl == yh and xl == xr:
                    continue
                if yl >= yh:
                    yl = 0
                    yh = row_num
                if xl >= xr:
                    xl = 0
                    xr = col_num
            card_imgs[card_index] = card_img[yl:yh, xl:xr] \
                if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh, xl:xr]

        return card_imgs, colors

    #八、分割字符并识别车牌文字
    def __identification(self, card_imgs, colors,model,modelchinese):
        # 识别车牌中的字符
        result = {}
        predict_result = []
        roi = None
        card_color = None
        for i, color in enumerate(colors):
            if color in ("blue", "yellow", "green"):
                card_img = card_imgs[i]
                # old_img = card_img
                # 做一次锐化处理
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
                card_img = cv2.filter2D(card_img, -1, kernel=kernel)

                # RGB转GARY
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yellow":
                    gray_img = cv2.bitwise_not(gray_img)
                
                # 二值化
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 查找水平直方图波峰
                x_histogram = np.sum(gray_img, axis=1)
                
                # 最小值
                x_min = np.min(x_histogram)
                
                # 均值
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = self.__find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    continue

                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]

                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

                wave_peaks = self.__find_waves(y_threshold, y_histogram)
                # 车牌字符数应大于6
                if len(wave_peaks) <= 6:
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                # 去除车牌上的分隔点
                point = wave_peaks[2]
                if point[1] - point[0] < max_wave_dis / 3:
                    point_img = gray_img[:, point[0]:point[1]]
                    if np.mean(point_img) < 255 / 5:
                        wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    continue
                
                # 分割牌照字符
                part_cards = self.__seperate_card(gray_img, wave_peaks)

                # 识别
                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉
                    if np.mean(part_card) < 255 / 5:
                        continue
                    part_card_old = part_card
                    w = abs(part_card.shape[1] - self.SZ) // 2

                    # 边缘填充
                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    

                    # 图片缩放（缩小）
                    part_card = cv2.resize(part_card, (self.SZ, self.SZ), interpolation=cv2.INTER_AREA)
                    
                    if i == 0:  # 识别汉字
                        charactor = self.modelchinese.testbegin(part_card)  # 匹配样本
        
                    else:  # 识别字母
                        charactor = self.model.testbegin(part_card)  # 匹配样本
                        

                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    print("待加入：%s" % charactor)
                    if charactor == "1" and i == len(part_cards) - 1:
                        if color == 'blue' and len(part_cards) > 7:
                            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                                continue
                        elif color == 'blue' and len(part_cards) > 7:
                            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                                continue
                        elif color == 'green' and len(part_cards) > 8:
                            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                                continue
                    predict_result.append(charactor)
                    print("已加入：%s"%charactor)
                roi = card_img  # old_img
                card_color = color
                break

        return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色

    #主体函数，调用_preTreatment并导入训练好的模型，进行预测输出
    def VLPR(self, car_pic):
        result = {}
        start = time.time()
        self.model = eng()
        self.modelchinese = chs()
        card_imgs, colors = self.__preTreatment(car_pic)
        if card_imgs is []:
            return
        else:
            predict_result, roi, card_color = self.__identification(card_imgs, colors,self.model,self.modelchinese)
            print(predict_result)
            if predict_result != []:
                result['UseTime'] = round((time.time() - start), 2)
                result['InputTime'] = time.strftime("%Y-%m-%d %H:%M:%S")
                result['Type'] = self.cardtype[card_color]
                result['List'] = predict_result
                result['Number'] = ''.join(predict_result[:2]) + '·' + ''.join(predict_result[2:])
                result['Picture'] = roi
                try:
                    result['From'] = ''.join(self.Prefecture[result['List'][0]][result['List'][1]])
                except:
                    result['From'] = '未知'
                return result
            else:
                return None
            
# 测试
if __name__ == '__main__':
    c = MainRecognition()
    result = c.VLPR('./Test/京AD77972.jpg')
    print(result)
