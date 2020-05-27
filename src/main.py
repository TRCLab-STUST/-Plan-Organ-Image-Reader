import glob
import json
import multiprocessing as mp
import os
import time
import cv2
import OrganImageReader as oir
import numpy

# Show Debug Message?
debug = True

# Dir Path
ROOT_DIR = os.path.abspath("../")
RESOURCE_DIR = os.path.join(ROOT_DIR, "resource/")
JSON_DIR = os.path.join(ROOT_DIR, "json/")
IMAGES_DIR = os.path.join(RESOURCE_DIR, "images/")
TABLE_PATH = os.path.join(RESOURCE_DIR, "color.txt")
OUTPUT_DIR = os.path.join(RESOURCE_DIR, "output/")

######################## 修改區 ############################
# Images Dir
COLOR_DIR = os.path.join(IMAGES_DIR, "origin/train")
# 'Tes': Test
# 'Val': Validation
# 'Tra': Training
mark = "Tes"
# Image Registration Output Dir
REGISTER_OUTPUT_DIR = os.path.join(IMAGES_DIR, "output/")
###########################################################

JSON_PATH_ct = os.path.join(JSON_DIR, "output" + mark + "_ct.json")
JSON_PATH_mr = os.path.join(JSON_DIR, "output" + mark + "_mr.json")

# init json file
file_ct = open(JSON_PATH_ct, 'w')
file_mr = open(JSON_PATH_mr, 'w')
file_ct.write("{\n}")
file_mr.write("{\n}")
file_ct.flush()
file_mr.flush()
file_ct.close()
file_mr.close()


#########################################################
# Ref: https://www.itread01.com/content/1536213742.html #
#########################################################
class MyEncoder(json.JSONEncoder):                      #
    def default(self, obj):                             #
        if isinstance(obj, numpy.integer):              #
            return int(obj)                             #
        elif isinstance(obj, numpy.floating):           #
            return float(obj)                           #
        elif isinstance(obj, numpy.ndarray):            #
            return obj.tolist()                         #
        else:                                           #
            return super(MyEncoder, self).default(obj)  #
#########################################################


# Thread Job
def job(image):
    tmp_ct = {}
    tmp_mr = {}
    organ_reader = oir.OrganImageReader(debug)

    # 讀取資料表
    organ_reader.load_table(TABLE_PATH)

    # 讀取圖片檔案
    filename = os.path.basename(image)
    filename = filename[:-4]
    filename_ct = filename + "_ct_output.jpg"
    filename_mr = filename + "_mr_output.jpg"

    size_ct = os.path.getsize(REGISTER_OUTPUT_DIR + filename_ct)
    size_mr = os.path.getsize(REGISTER_OUTPUT_DIR + filename_mr)
    key_ct = filename_ct + str(size_ct)
    key_mr = filename_mr + str(size_mr)

    # CT
    tmp_ct[key_ct] = {}
    tmp_ct[key_ct]['fileref'] = ''
    tmp_ct[key_ct]['size'] = size_ct
    tmp_ct[key_ct]['filename'] = filename_ct
    tmp_ct[key_ct]['base64_img_data'] = ''
    tmp_ct[key_ct]['file_attributes'] = {}
    tmp_ct[key_ct]['regions'] = {}
    # MR
    tmp_mr[key_mr] = {}
    tmp_mr[key_mr]['fileref'] = ''
    tmp_mr[key_mr]['size'] = size_mr
    tmp_mr[key_mr]['filename'] = filename_mr
    tmp_mr[key_mr]['base64_img_data'] = ''
    tmp_mr[key_mr]['file_attributes'] = {}
    tmp_mr[key_mr]['regions'] = {}
    a = 0

    if debug:
        print("Image: " + filename + "\n")

    # 讀取圖片
    organ_reader.load_image(image)

    # 找出圖片的器官
    find_org = organ_reader.find_organ()

    for index in find_org:
        idx = organ_reader.organ_rgb_list.index(index)
        if debug:
            print("Start Find Index: " + str(idx))

        # 建立過濾後器官圖片
        organ_reader.filter_organ(idx)

        # 建立過濾後圖片輪廓
        organ_reader.draw_contours(organ_reader.organ_rgb_list[idx])

        for n in range(0, len(organ_reader.contours)):
            list_x = []
            list_y = []
            for point in organ_reader.contours[n]:
                for x, y in point:
                    list_x.append(x)
                    list_y.append(y)
            # CT
            tmp_ct[key_ct]['regions'][a] = {}
            tmp_ct[key_ct]['regions'][a]['shape_attributes'] = {}
            tmp_ct[key_ct]['regions'][a]['shape_attributes']['name'] = 'polygon'
            tmp_ct[key_ct]['regions'][a]['shape_attributes']['all_points_x'] = list_x
            tmp_ct[key_ct]['regions'][a]['shape_attributes']['all_points_y'] = list_y
            tmp_ct[key_ct]['regions'][a]['region_attributes'] = {}
            tmp_ct[key_ct]['regions'][a]['region_attributes']['name'] = str(idx)
            # MR
            tmp_mr[key_mr]['regions'][a] = {}
            tmp_mr[key_mr]['regions'][a]['shape_attributes'] = {}
            tmp_mr[key_mr]['regions'][a]['shape_attributes']['name'] = 'polygon'
            tmp_mr[key_mr]['regions'][a]['shape_attributes']['all_points_x'] = list_x
            tmp_mr[key_mr]['regions'][a]['shape_attributes']['all_points_y'] = list_y
            tmp_mr[key_mr]['regions'][a]['region_attributes'] = {}
            tmp_mr[key_mr]['regions'][a]['region_attributes']['name'] = str(idx)
            a += 1

    if debug:
        print("End Finder=======================================")

    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), organ_reader.black)

    return [tmp_ct, tmp_mr]


# Json Save (Encode)
def save(path, data):
    with open(path, 'w') as f:
        j_str = json.dumps(data, cls=MyEncoder)
        f.write(j_str)
        f.flush()
        f.close()


# Main
def main():
    pool = mp.Pool()

    # 讀取整個資料夾的.bmp
    images = glob.glob(COLOR_DIR + "*.bmp", recursive=True)

    multi_res = [pool.apply_async(job, (i, )) for i in images]
    data_ct = {}
    data_mr = {}
    i = 1
    for res in multi_res:
        data_ct.update(res.get()[0])
        data_mr.update(res.get()[1])
        print(str(i) + "/" + str(len(multi_res)))
        i += 1

    save(JSON_PATH_ct, data_ct)
    save(JSON_PATH_mr, data_mr)


if __name__ == '__main__':
    tStart = time.time()
    main()
    tEnd = time.time()
    print("It cost %f sec" % (tEnd - tStart))
