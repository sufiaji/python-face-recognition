from faceorchestrator import FaceOrchestrator
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import count_frames
import random
import imutils
import pickle
import time
import cv2
import os

def run_video():
    PATH_VIDEO = 'D:/Binus/data/videos/VID_20190915_123023.mp4'
    PATH_OUT_VIDEO = 'D:/Binus/data/videos/VID_20190915_123023_2.avi'    
    fo = FaceOrchestrator()
    fs = FileVideoStream(PATH_VIDEO).start()
    time.sleep(1.0)
    frame = fs.read()
    h, w, __ = frame.shape
    out = cv2.VideoWriter(PATH_OUT_VIDEO,
            cv2.VideoWriter_fourcc('M','J','P','G'), 
            24, (w,h))    
    i = 0
    while fs.more():
        i = i+1
        print('Processing frame ' + str(i))
        try:
            frame = fs.read()
            fo.set_image_to_be_processed(frame)
            # detect faces
            fo.fd_method = 'AI'
            fo.fd_ai_min_score = 0.6
            fo.detect_faces()
            fo.align_faces(target_face_percent=0.28)
            # reconize faces
            # fo.fr_max_delta = 0.6
            # fo.fr_method = 'DELTA'
            fo.fr_min_probability = 0.5
            fo.fr_method='ML'
            frame_with_label, names = fo.recognize_faces()
            out.write(frame_with_label)
        except:
            pass
    fs.stop()
    out.release()
    print('Finish.')

def run_camera_delta():
    # URL = 'rtsp://192.168.0.116:554/1/h264major'
    # URL = 0
    # URL='rtsp://192.168.43.58:8080/h264_ulaw.sdp'
    # URL = 'rtsp://192.168.43.245:554/1/h264major'
    URL='rtsp://192.168.43.150:8080/h264_ulaw.sdp'
    fo = FaceOrchestrator()
    fs = VideoStream(URL).start()
    time.sleep(1.0)    
    d = 0
    aji = 0
    unknown = 0
    false_positive = 0
    while True:
        try:
            frame = fs.read()
            fo.set_image_to_be_processed(frame)
            # # detect faces
            fo.fd_method = 'AI'
            fo.fd_ai_min_score = 0.6
            fo.detect_faces()
            # align detected faces
            fo.align_faces(target_face_percent=0.25)
            # recognize faces
            fo.fr_method = 'DELTA'
            fo.fr_max_delta = 0.6
            frame_with_label, name = fo.recognize_faces()
            frame_with_label = FaceOrchestrator.resize(frame_with_label, 0.4)
            # FaceOrchestrator.draw_rectangle(frame, fo.boxes, 2)
            cv2.imshow('FR', frame_with_label)
            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break
        except Exception as e:
            print(str(e))
    fs.stop()
    cv2.destroyAllWindows()
    print('Finish.')

def run_camera_ml():
    # URL = 'rtsp://192.168.0.116:554/1/h264major'
    # URL = 0
    URL='rtsp://192.168.100.6:8080/h264_ulaw.sdp'
    # URL = 'rtsp://192.168.43.245:554/1/h264major'
    fo = FaceOrchestrator()
    fs = VideoStream(URL).start()
    time.sleep(1.0)    
    d = 0
    aji = 0
    unknown = 0
    false_positive = 0
    while True:
        try:
            frame = fs.read()
            fo.set_image_to_be_processed(frame)
            # # detect faces
            fo.fd_method = 'AI'
            fo.fd_ai_min_score = 0.6
            fo.detect_faces()
            # align detected faces
            fo.align_faces(target_face_percent=0.25)
            # recognize faces
            fo.fr_method = 'ML'
            fo.fr_min_probability=0.5
            frame_with_label, name = fo.recognize_faces()
            # resize
            frame_with_label = FaceOrchestrator.resize(frame_with_label, 0.4)
            # FaceOrchestrator.draw_rectangle(frame, fo.boxes, 2)
            cv2.imshow('FR', frame_with_label)
            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break
        except Exception as e:
            print(str(e))
    fs.stop()
    cv2.destroyAllWindows()
    print('Finish.')

def process_all_images():
    # extract face image from RAW dir, write image to disk DETECTED dir and ALIGNED dir
    # ONLY 1 face per image
    fo = FaceOrchestrator()
    fo.mass_align_images()

def rename_file():
    FaceOrchestrator.rename_files('AJI', '__', 
        'D:/Binus/data/aji2/')

def mass_encodings():
    # encode all face images from ALIGNED dir
    fo = FaceOrchestrator()
    fo.encode_images_and_save()

def train_svm():
    fo = FaceOrchestrator()
    fo.train_svm_model()

def print_encodings():
    fo = FaceOrchestrator()
    fo.load_encodings()
    print(fo.data['names'])
    i = 0
    # for encoding in fo.data['encodings']:
    #     i = i+1
    #     print('Encoding ' + str(i))
    #     print(encoding)

def encode_from_movie():
    fo = FaceOrchestrator()
    fo.insert_database_encode_from_video(name='CORNEL',
        videopath='D:/@Projects/Project Binus/Dataset/Cornel/videos2/(output001)VID_20190906_093155.mp4',
        num_sample=3)

def database_from_movie():
    fo = FaceOrchestrator()
    fo.create_database_from_video(
        videopath='D:/@Projects/Project Binus/Dataset/Cornel/videos2/VID_20190906_093444.mp4',
        num_sample=10)

def delete_from_encodings():
    fo = FaceOrchestrator()
    fo.remove_encoding(name='cornel')

def recognize_img():
    img = cv2.imread('C:/Users/Universe/Dropbox/Codes/Binus/data/CORNEL2.jpg', 1)
    fo = FaceOrchestrator()    
    fo.set_image_to_be_processed(img)

    # detect faces
    fo.fd_method = 'AI'
    fo.fd_ai_min_score = 0.6
    fo.detect_faces()

    # align detected faces
    fo.align_faces(target_face_percent=0.25)

    # recognize faces
    fo.fr_method = 'DELTA'
    fo.fr_max_distance = 0.9
    result_img, name = fo.recognize_faces()

    # display
    result_img = FaceOrchestrator.resize(result_img, 0.4)
    cv2.imshow('FR', result_img)
    cv2.waitKey()

if __name__ == '__main__':
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run_camera_delta()
    # run_camera_ml()
    # process_all_images()
    # rename_file()
    # mass_encodings()
    # train_svm()
    # run_video()
    # print_encodings()
    # encode_from_movie()
    # delete_from_encodings()
    # database_from_movie()
    # recognize_img()