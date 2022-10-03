import cv2
import numpy as np
import dlib
import glob
import easyocr
from difflib import SequenceMatcher
import json
import datetime
import os

# REF POINTS
im_ref = np.zeros((370, 800, 3), np.uint8)
pts_ref = np.array([[70, 55], [725, 55], [725, 300], [70, 300], [
                   397, 55], [725, 177], [397, 300], [70, 177]])

######## HELPER FUNCTION #######
# argmax
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# text similarity
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def dlibShape2numpyArray(shape):
    vec = np.empty([shape.num_parts, 2], dtype=int)
    for b in range(shape.num_parts):
        vec[b][0] = shape.part(b).x
        vec[b][1] = shape.part(b).y
    return vec


def bb_to_rect(bb):
    top = bb[1]
    left = bb[0]
    right = bb[0]+bb[2]
    bottom = bb[1]+bb[3]
    return np.array([top, right, bottom, left])


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

# Sort 2 numbers
def sortTwoNumbers(first, second):
    if(first >= second):
        new_first = first
        new_second = second
    else:
        new_first = second
        new_second = first
    return new_first, new_second


def splitPlate(img):
    img_h, img_w = img.shape[:2]
    img_top = img[0:int(0.375*img_h), 0:img_w]
    img_middle = img[int(0.375*img_h):int(0.625*img_h), 0:img_w]
    img_bottom = img[int(0.625*img_h):img_h, 0:img_w]

    return img_top, img_middle, img_bottom

# Recognize plate number
def recognizeNumber(img, reader):
    img_h, img_w = img.shape[:2]
    img_area = img_h*img_w
    results = reader.readtext(img)

    det_rect = None
    det_area = None
    det_text = None
    det_cf = None

    if len(results) >= 1:
        result = results[0]
        det_rect = result[0]
        det_area = abs(det_rect[0][0] - det_rect[1][0]) * \
            abs(det_rect[1][1] - det_rect[2][1])
        det_text = result[1]
        det_cf = result[2]

        for i, (rect, text, cf) in enumerate(results):
            this_area = abs(rect[0][0] - rect[1][0]) * \
                abs(rect[1][1] - rect[2][1])
            if det_area < this_area:
                det_rect = rect
                det_area = this_area
                det_text = text
                det_cf = cf

        if det_area/img_area < 0.3:
            det_text = "No Text"

    return det_text


# Recognize province
def recognizeProvince(img, reader):
    p_text = recognizeNumber(img, reader)
    # print(p_text)
    if p_text:
        my_file = open("./model/provinces.txt", encoding="utf8")
        content = my_file.read()
        province_list = content.split("\n")
        my_file.close()

        prov_score = []
        for prov in province_list:
            a = similar(p_text, prov)
            prov_score.append(a)
        max_index = argmax(prov_score)

        maxProvScore = max(prov_score)
        # print

        return province_list[max_index], maxProvScore





def createfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error : create directory.' + directory)


def check_top(resultTop):
    try:

        thChange = "เแไใา"
        x = ""
        if(resultTop == ""):
            resultTop = "Error detect"
        else:
            for i in thChange:
                if(i == resultTop[0]):
                    resultTop = "1"+resultTop[1:]
            for j in resultTop:
                if(j != "า"):
                    x = x+j
                else:
                    x = x + "ว"
                    resultTop = x
    except:
        resultTop = "Error detect"

    return resultTop

# write json file by uuid
def write_jsonFileUuid(fileOutput, uuid):
    with open('./dataPlateJSON/' + uuid + '.json', 'w', encoding='utf-8') as f:
        json.dump(fileOutput, f, ensure_ascii=False, indent=4)
    print(fileOutput)
    return fileOutput


######## MAIN FUNCTION #######
# read_plate_detect function
def read_plate_detect(uuid):

    # prepare folder
    now = datetime.datetime.now()
    date_time = str(now.year) + "_" + str(now.month) + "_" + \
        str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
    path_final_img = './ImagePlate/PlateDate' + date_time
    createfolder(path_final_img)

    # count file in folder for *
    images_path = glob.glob("../detection-api/Images/" + uuid + ".jpg")
    print(images_path)
    # count = len(os.listdir(img_folder_path))
    count = 1


    ######################################
    #        main code starts here
    ######################################
    # for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (0, 255, 255)
    lineType = 2

    # array and for save data
    result_OCRPlate = []
    result_OCRplate_json = []

    # reference points
    im_ref = np.zeros((400, 510, 3), np.uint8)
    pts_ref = np.array([[0, 0], [509, 0], [0, 399], [509, 399], [254, 0], [509, 199], [254, 399], [
                       0, 199], [20, 159], [489, 159], [20, 249], [489, 249], [20, 20], [489, 20], [20, 379], [489, 379]])

    person_Id = 0
    complete = True
    finaldata = {}

    print("-------------------- Plate OCR --------------------")
    for img_path in images_path:
        head_tail = os.path.split(img_path)

        x = head_tail[1].split("_")
        y = x[1].split(".")
        z = y[0]
        print("head_tail", head_tail)
        # print("x y z => ", x, y, z)
        # print("./" + img_path)
        try:
            img = cv2.imread("./" + img_path)
            # img = cv2.resize(img, None, fx=0.7, fy=0.7,interpolation = cv2.INTER_LINER)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_d = img.copy()

            detector = dlib.fhog_object_detector("./model/motor.svm")
            predictor = dlib.shape_predictor("./model/motor.dat")
            rects = detector(gray, 1)

            if len(rects) >= 1:
                rect = rects[0]
                (x, y, w, h) = rect_to_bb(rect)
                reader = easyocr.Reader(['th', 'en'])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                shape = predictor(gray, rect)
                shapecv = dlibShape2numpyArray(shape)
                homo, status = cv2.findHomography(shapecv, pts_ref, 0, 5.0)
                im_out = cv2.warpPerspective(
                    img_d, homo, (im_ref.shape[1], im_ref.shape[0]))
                im_number_top, im_province, im_number_bottom = splitPlate(
                    im_out)

                # top
                result_number_top = recognizeNumber(im_number_top, reader)
                # if(result_number_top == ""):
                #     result_number_top = "Error detected"
                result_number_top = check_top(result_number_top)
                result_number_top = result_number_top.strip()

                # province
                result_province, maxProvScore = recognizeProvince(
                    im_province, reader)

                # number
                result_number_bottom = recognizeNumber(
                    im_number_bottom, reader)
                result_number_bottom = filter(
                    str.isdigit, result_number_bottom)
                result_number_bottom = "".join(result_number_bottom)

                # insert description plate ,array to json file
                plate_dict = {
                    "image": str(path_final_img)+'/'+str("person_"+str(z)+".jpg"),
                    "top": str(result_number_top).strip(),
                    "province": str(result_province),
                    "bottom": str(result_number_bottom),
                    "score": str(maxProvScore)
                }
                result_OCRPlate.append(plate_dict)
                result_OCRplate_json = json.dumps(
                    result_OCRPlate, ensure_ascii=False, indent=4).encode('utf8')
                finaldata = json.loads(result_OCRplate_json)

                # write image in folder data
                cv2.imwrite(path_final_img+'/person_'+str(z)+'.jpg', img)

                # print top mid bottom
                print("---------------------------------------")
                print("true Detected : ", person_Id+1, "/", count)
                print(result_number_top)
                print(result_province)
                print(result_number_bottom)
                # print(final)
                print("----------------------------------------")

                for i, (sX, sY) in enumerate(shapecv):
                    cv2.circle(img, (sX, sY), 2, (255, 0, 0), -1)
                    cv2.circle(img, (sX, sY), 3, (0, 0, 255), 2)
                    cv2.putText(img, str(i), (sX+5, sY-5), font,
                                fontScale, fontColor, lineType)

            if (int(person_Id) == int(count-1)):
                complete = False
                write_jsonFileUuid(finaldata, uuid)
                # delete_file()
                print("_____________ End Process 3 _____________")
                # print("--- %d min ---" % ((time.time() - start_time)/60))
                break
            person_Id += 1

        # else:
        except:
            print("false Detected: ", person_Id + 1, "/", count)
            # cv2.destroyAllWindows()
            if(count == 0):
                print("Folder is empty")
                break

            elif(person_Id == int(count-1)):
                if(finaldata == {}):
                    print("Data is empty")
                    complete = False
                else:
                    complete = False
                    write_jsonFileUuid(finaldata, uuid)

                print("_____________ End Process 3 _____________")

            person_Id += 1

    return finaldata


# startProgramPlateDetection function
def startProgramPlateDetection(uuid):
    finaldata = read_plate_detect(uuid)
    return finaldata
