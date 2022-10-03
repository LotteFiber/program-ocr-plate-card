import cv2
import numpy as np
import dlib
import glob
import easyocr
import json
import datetime
import os

######## HELPER FUNCTION #######
# create folder
def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error : create directory.' + directory)

# write json file by uuid
def write_json_file_uuid(fileOutput, uuid):
    with open('./dataCardJSON/' + uuid + '.json', 'w', encoding='utf-8') as f:
        json.dump(fileOutput, f, ensure_ascii=False, indent=4)
    print(fileOutput)
    return fileOutput

######## MAIN FUNCTION #######
# read_card_detect function
def read_card_detect(uuid):
    # prepare folder
    now = datetime.datetime.now()
    date_time = str(now.year) + "_" + str(now.month) + "_" + \
        str(now.day) + "_" + str(now.hour) + "_" + str(now.minute)
    path_final_img = './ImageCard/CardDate' + date_time
    create_folder(path_final_img)

    # count file in folder for *
    images_path = glob.glob("../detection-api/Images/" + uuid + ".jpg")
    print(images_path)

    # array and for save data
    result_OCRCard = []
    result_OCRcard_json = []

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
            detector = dlib.fhog_object_detector('./model/train03.svm')
            predictor = dlib.shape_predictor('./model/shapePredict3.dat')

            frame = cv2.imread(
                './' + img_path)
            # frame = cv2.imread('./card_crop/IMG_8604.png')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 0)

            for rect in rects:
                (x, y, w, h) = (rect.left(), rect.top(),
                                (rect.right() - rect.left()), rect.bottom() - rect.top())

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                shape = predictor(gray, rect)
                for i in range(len(shape.parts())):
                    cv2.circle(frame, (shape.part(i).x, shape.part(i).y),
                               3, (255, 255, 255), -1)

                pts1 = np.float32([[shape.part(0).x, shape.part(0).y], [shape.part(3).x, shape.part(
                    3).y], [shape.part(1).x, shape.part(1).y], [shape.part(2).x, shape.part(2).y]])
                pts2 = np.float32([[0, 0], [600, 0], [0, 400], [600, 400]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(gray, M, (600, 400))

            img_d = dst.copy()
            crop_name = img_d[125:155, 150:455]
            crop_student_id = img_d[155:185, 245:455]
            crop_faculty = img_d[180:215, 150:455]

            reader = easyocr.Reader(['en', 'th'])
            text_name = reader.readtext(crop_name)
            text_student_id = reader.readtext(crop_student_id)
            text_faculty = reader.readtext(crop_faculty)

            # cv2.imshow('crop warp image', img_d)
            # cv2.imshow('crop name', crop_name)
            # cv2.imshow('crop student_id', crop_student_id)
            # cv2.imshow('crop faculty', crop_faculty)

            name = text_name[0][1].strip().split(" ")
            first_name = name[0]
            last_name = name[1]
            student_id = text_student_id[0][1]
            faculty = text_faculty[0][1].strip()[3:]
            # print(name, first_name, last_name, student_id, faculty)

            # insert description student_id, first_name, last_name, faculty
            card_dict = {
                "image": str(path_final_img) + '/' + str("card_" + str(z) + ".jpg"),
                "first_name": first_name,
                "last_name": last_name,
                "student_id": student_id,
                "faculty": faculty
            }
            result_OCRCard.append(card_dict)
            result_OCRcard_json = json.dumps(
                result_OCRCard, ensure_ascii=False, indent=4).encode('utf8')
            finaldata = json.loads(result_OCRcard_json)
            write_json_file_uuid(finaldata, uuid)

            # write image in folder data
            cv2.imwrite(path_final_img + '/' + "card_" + str(z) + ".jpg", dst)

            return finaldata
        except:
            print("false Detected: ", "/")
            cv2.destroyAllWindows()
            return False


# startProgramCardDetection function
def startProgramCardDetection(uuid):
    finaldata = read_card_detect(uuid)
    return finaldata
