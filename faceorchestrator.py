from sklearn.preprocessing import LabelEncoder
from imutils.face_utils import FaceAligner
from sklearn.svm import SVC
from imutils import paths

import tensorflow as tf
from tensorflow import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import face_recognition
import numpy as np
import imutils
import random
import pickle
import heapq
import dlib
import cv2



"""
ONE BIG CLASS OF FACE RECOGNITION  
Created By: merkaba/pradhonoaji
Created on: 30-Aug-2019
"""

class FaceOrchestrator(object):

    # default path for encodings file pickle
    file_encodings = './models/encodings.pickle'
    # path for SVM model and label
    file_svm_model = './models/svm_model.pickle'
    file_svm_label = './models/svm_label.pickle'
    # path for face detector model
    file_face_detector = './models/frozen_inference_graph_face.pb'
    # path for face landmark used in face aligner
    file_face_landmark = './models/shape_predictor_68_face_landmarks.dat'
    file_haar = './models/haarcascade_frontalface_alt2.xml'
    # final face width (=height ~square rectangle) for recognition in pixel
    # ==> face ratio
    target_face_width = 256
    target_face_height = 286
    # percentage of zooming, calculated based on the ratio
    # of distance between left eye to the border of image and iage width,
    # Zoom In < target_percent_face < Zoom out 
    # target_face_percent = 0.3
    # filename separator between name and random string generated from UUID
    name_separator = '__'
    # image array (BGR)
    image = []
    # boxes coordinate of faces (left, right, top, bottom)
    boxes = []
    # detected faces
    faces_detected = []
    # normalized faces
    faces_aligned = []
    # database
    data = {}
    # SVM model
    svm_model = None
    svm_label = None
    # face detection method: AI or HAAR or HOG
    fd_method = 'AI'
    fd_ai_min_score = 0.6    
    # face recognition method: ML or DELTA
    fr_method = 'DELTA'
    fr_max_delta = 0.6 # delta max
    fr_min_probability = 0.5 # SVM probability

    def __init__(self):
        # initiate tensorflow object for face detection
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.file_face_detector, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True
        # initiate face aligner
        predictor = dlib.shape_predictor(self.file_face_landmark)
        # percentage_face = (self.target_face_percent, self.target_face_percent)
        # construct face aligner
        self.faceAligner = FaceAligner(predictor, 
            desiredFaceWidth=self.target_face_width, 
            desiredFaceHeight=self.target_face_height,
            desiredLeftEye=(0.3, 0.3))        
        # load encodings         
        self.load_encodings()
        # load svm models
        self.load_svm_model()
        # cascade haar
        self.face_cascade = cv2.CascadeClassifier(self.file_haar)

    def run(self):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        # image_np = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_np = self.rgb_image
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        # start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # elapsed_time = time.time() - start_time
        # print('[Info] inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)
    
    def set_image_to_be_processed(self, image):
        """
        image = array of image (frame)
        """
        self.image = image
        self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    
    def detect_faces(self):
        """ 
        DETECT FACES USING MOBILENET SSD MODEL (TENSORFLOW) OR HAAR CASCADE        
        @output-boxes: a list of rectangle face (left, right, top, bottom)
        """
        newboxes = []
        faces_detected = []
        if self.fd_method == 'AI':
            # detect Face using Deep Learning: architecture MobileNet, method SSD
            (boxes, scores, __, __) = self.run() 
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            max_boxes_to_draw = 20
            for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                # check teh score
                if scores[i] > self.fd_ai_min_score:
                    box = tuple(boxes[i].tolist())
                    ymin, xmin, ymax, xmax = box
                    im_height, im_width, __ = self.image.shape
                    # convert tensorflow normalized coordinate to absolute coordinate
                    (left, right, top, bottom) = (int(xmin*im_width), int(xmax*im_width), int(ymin*im_height), int(ymax*im_height))
                    face = self.image[top:bottom, left:right]
                    newboxes.append((left, right, top, bottom))
                    faces_detected.append(face)                
            self.boxes = newboxes
            self.faces_detected = faces_detected
        elif self.fd_method == 'HOG':
            # detect Face using HOG
            # self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(self.rgb_image, model='hog')
            for box in boxes:
                top, right, bottom, left = box
                face = self.image[top:bottom, left:right]
                newboxes.append((left, right, top, bottom))
                faces_detected.append(face)
            self.boxes = newboxes
            self.faces_detected = faces_detected
        elif self.fd_method == 'HAAR':
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=9,
                        minSize=(30,30),
                        flags = cv2.CASCADE_SCALE_IMAGE
                    )
            for (x, y, w, h) in faces:
                newboxes.append((x, y+w, y, y+h))
                face = self.image[y:y+h, x:x+w]
                faces_detected.append(face)
            self.boxes = newboxes
            self.faces_detected = faces_detected

    
    def align_faces(self, target_face_percent=0.3):
        """
        ALIGN FACES BASED ON FACE LANDMARK USING DLIB
        Note: align face before generate face encoding
        @param-image: image which contain face (or faces)
        @param-boxes: list of rectangle of face (left, right, top, bottom)
        @output-faces_aligned: list of aligned face image
        """
        self.__set_target_face_percentage(target_face_percent)
        faces_aligned = []
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # for left, right, top, bottom in boxes:
        for box in self.boxes:
            left, right, top, bottom = box
            rect = dlib.rectangle(left=left, right=right, top=top, bottom=bottom)
            facealigned = self.faceAligner.align(self.image, gray, rect)
            faces_aligned.append(facealigned)
        self.faces_aligned = faces_aligned
    
    def recognize_faces(self):
        if self.fr_method == 'DELTA':
            return self.recognize_faces_and_draw_knn_2(self.fr_max_delta)
        else:
            return self.recognize_faces_and_draw_svm(self.fr_min_probability)
    
    def recognize_faces_and_draw_knn(self, minDistance=0.5):
        """ 
        RECOGNIZE ALL FACES DETECTED IN A FRAME
        minDistance: less value more strict
        Method Boolean
        """
        index = 0
        new_image = self.image.copy()    
        # loop all faces
        name = None
        for left, right, top, bottom in self.boxes:
            cv2.rectangle(new_image,(left,top),(right,bottom),(0,255,0),2)
            if self.faces_aligned == [] or self.faces_aligned is None:
                face = self.faces_detected[index]
            else:
                face = self.faces_aligned[index]
            # generate encoding for the face
            encoding = self.generate_encoding(face)
            # match a face with all faces in database
            # for encoding in encodings:
            matches = face_recognition.compare_faces(self.data["encodings"], encoding, minDistance)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(new_image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
            index = index + 1
        
        return new_image, name
    
    def recognize_faces_and_draw_knn_2(self, minDistance=0.5):
        """ 
        RECOGNIZE ALL FACES DETECTED IN A FRAME, RETURN AN IMAGE AND LIST OF NAME 
        minDistance: less value more strict
        Method Sort Minimum
        """
        index = 0
        new_image = self.image.copy()    
        # loop all faces
        top_names = []
        name = None
        for left, right, top, bottom in self.boxes:
            # draw rectangle around the face in frame
            cv2.rectangle(new_image,(left,top),(right,bottom),(0,255,0),2)
            if self.faces_aligned == [] or self.faces_aligned is None:
                face = self.faces_detected[index]
            else:
                face = self.faces_aligned[index]
            # generate encoding for the face
            encoding = self.generate_encoding(face)
            # get distance value 
            dict_index = -1
            # match_index = [] 
            match_value = []
            match_name = []
            for db_encoding in self.data['encodings']:
                dict_index = dict_index + 1
                # compare face
                dist = np.linalg.norm(db_encoding - encoding)
                if dist < minDistance:
                    # match_index.append(dict_index)
                    match_value.append(dist)
                    match_name.append(self.data['names'][dict_index])
            if match_value != []:
                # sort by 3-smallest distance value (ascending)
                sorted_index = np.argsort(match_value)
                name = match_name[sorted_index[0]]
                top_names.append(name)
                # write on the displayed window, for each value for this face
                yy = 0
                for k in sorted_index:
                    yy = yy + 1
                    y = top + (yy * 25)
                    xname = "{}: {:.2f}%".format(match_name[k], (1-match_value[k]) * 100) 
                    cv2.putText(new_image, xname, (right+20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if yy == 3: 
                        break
            else:
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(new_image, 'Unknown', (right+20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            index = index + 1
        
        return new_image, top_names
    
    def recognize_faces_and_draw_svm(self, minProba=0.5):
        """ 
        RECOGNIZE ALL FACES DETECTED IN A FRAME
        minDistance: less value more strict
        """
        index = 0
        alpha = 0.6
        new_image = self.image.copy()
        text1 = None
        text2 = None
        text3 = None
        top_names = []
        # loop all faces
        for left, right, top, bottom in self.boxes:
            cv2.rectangle(new_image,(left,top),(right,bottom),(0,255,0),2)
            if self.faces_aligned == [] or self.faces_aligned is None:
                face = self.faces_detected[index]
            else:
                face = self.faces_aligned[index]
            # generate encoding for current face
            encoding = self.generate_encoding(face)
            encoding = encoding.reshape(1, -1)
            # match a face with all faces in database
            # perform classification to recognize the face
            preds = self.svm_model.predict_proba(encoding)[0]
            # get the biggest probability
            j = np.argmax(preds)
            proba = preds[j]
            # overlay = new_image.copy()
            left_rect = right + 10
            top_rect = top
            # if the biggest probability is bigger than allowed, then sort 3 biggest probabilities
            if proba > minProba:                
                # get 3 biggest probability SVM
                big3_index = heapq.nlargest(3, range(len(preds)), preds.take)
                name = self.svm_label.classes_[big3_index[0]]
                prob = preds[big3_index[0]]
                text1 = "{}: {:.2f}%".format(name, prob * 100)   
                top_names.append(name) 
                name = self.svm_label.classes_[big3_index[1]]
                prob = preds[big3_index[1]]            
                text2 = "{}: {:.2f}%".format(name, prob * 100)  
                top_names.append(name) 
                name = self.svm_label.classes_[big3_index[2]]
                prob = preds[big3_index[2]]
                text3 = "{}: {:.2f}%".format(name, prob * 100)  
                top_names.append(name) 
                right_rect = left_rect + 150
                bottom_rect = top + 75
                # cv2.rectangle(overlay, (left_rect, top_rect), (right_rect, bottom_rect), (0, 0, 0), -1)         
                # new_image = cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0)
                # *** 2MPX
                cv2.putText(new_image, text1, (left_rect+10, top_rect+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(new_image, text2, (left_rect+10, top_rect+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(new_image, text3, (left_rect+10, top_rect+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # *** 8MPX
                # cv2.putText(new_image, text1, (left_rect+10, top_rect+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # cv2.putText(new_image, text2, (left_rect+10, top_rect+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # cv2.putText(new_image, text3, (left_rect+10, top_rect+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # *** 12MPX
                # cv2.putText(new_image, text1, (left_rect+10, top_rect+20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
                # cv2.putText(new_image, text2, (left_rect+10, top_rect+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
                # cv2.putText(new_image, text3, (left_rect+10, top_rect+180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
                # print(text1)
                # print(text2)
                # print(text3)
            else:
                # name = 'Unknown'
                text1 = 'Unknown'
                # print(text1)
                right_rect = left_rect + 100
                bottom_rect = top + 35
                # cv2.rectangle(overlay, (left_rect, top_rect), (right_rect, bottom_rect), (0, 0, 0), -1)         
                # new_image = cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0)
                # *** 2MPX
                cv2.putText(new_image, text1, (left_rect+10, top_rect+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(new_image, text1, (left_rect+10, top_rect+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                top_names.append('Unknown')
            
            # text_title = "threshold {:.2f}%".format(minProba * 100)
            # cv2.putText(new_image, text_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
            index = index + 1
        
        return new_image, top_names
    
    def generate_encoding(self, faceImg):
        """ 
        GENERATE LIST OF 128D VECTOR FROM A FACE IMAGE ARRAY
        Note: to increase accuracy, shape and landmark must be consistent !
        Please align face image before call this function !
        @param-faceImg: a BGR face image (aligned, normalized)
        @output-encodings: 128D face vector
        """ 
        boxes = []
        h, w, __ = faceImg.shape
        # top, right, bottom, left
        box = (0, w-1, h-1, 0)
        boxes.append(box)
        rgb = cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB) # switch to RGB for DLIB
        encodings = face_recognition.face_encodings(rgb, boxes)
        return encodings[0]
    
    def set_target_face_width(self, target_face_width):
        """ width of final aligned face image to be recognized, in pixel """
        self.target_face_width = target_face_width
        self.faceAligner.desiredFaceWidth = target_face_width
    
    def set_target_face_height(self, target_face_height):
        """ width of final aligned face image to be recognized, in pixel """
        self.target_face_height = target_face_height
        self.faceAligner.desiredFaceHeight = target_face_height
    
    def __set_target_face_percentage(self, target_face_percent):
        """ Zoom In < target_percent_face < Zoom out  """
        # self.target_face_percent = target_face_percent
        # percentage_face = (self.target_face_percent, self.target_face_percent)
        self.faceAligner.desiredLeftEye = (target_face_percent, target_face_percent)
    
    def insert_encoding(self, encoding, name):
        print('[Info] Loading ' + self.file_encodings + '...') 
        # loading from pickel file encoding     
        if self.load_encodings() == True:
            # extract names and encodings
            knownEncodings = self.data['encodings']
            knownNames = self.data['names']
            # appned the new encoding
            knownEncodings.append(encoding)
            knownNames.append(name)
            # dump back to file
            self.data['encodings'] = knownEncodings
            self.data['names'] = knownNames
            print('[Info] Serializing encodings...')
            data = {'encodings': knownEncodings, 'names': knownNames}
            f = open(self.file_encodings, 'wb')
            f.write(pickle.dumps(data))
            f.close()
            print('[Info] Encoding saved on disk.')
        else:
            print('[Error] Failed to save encoding to disk.')

    def encode_images_and_save(self, faceImgPath='./database/normalized'):
        """ 
        @path: imagePaths of normalized face images need to generate and save the encodings
        Note that all face images should have been normalized before calling this method
        """
        knownEncodings = []
        knownNames = []
        imagePaths = list(paths.list_images(faceImgPath))
        total = 0
        for (i, imagePath) in enumerate(imagePaths):
            print('[Info] processing image {}/{}'.format(i + 1, len(imagePaths)))
            head_tail = os.path.split(imagePath)
            filename = head_tail[1]
            name, d = filename.split(self.name_separator)
            image = cv2.imread(imagePath) # read image            
            encoding = self.generate_encoding(image)
            if encoding is not None or encoding != []:
                knownEncodings.append(encoding)
                knownNames.append(name)
                total += 1
        print('[Info] serializing {} encodings...'.format(total))
        data = {'encodings': knownEncodings, 'names': knownNames}
        f = open(self.file_encodings, 'wb')
        f.write(pickle.dumps(data))
        f.close()
        print('[Info] done.')
    
    def load_encodings(self):
        """
        Load encoding from pickle file to class member data
        """
        print('[Info] Loading ' + self.file_encodings + '...')
        try:     
            self.data = pickle.loads(open(self.file_encodings, 'rb').read())
        except Exception as ex:
            print('[Error] load encoding: ' + str(ex))
            return False
        return True

    def save_encodings(self, path='./database/'):
        """
        Save encoding to pickle file, from class member data
        """
        total = len(self.data['encodings'])
        print('[Info] serializing {} encodings...'.format(total))
        f = open(self.file_encodings, 'wb')
        f.write(pickle.dumps(self.data))
        f.close()
        print('[Info] done.')

    def load_svm_model(self):
        print('[Info] Loading ' + self.file_svm_model + '...')
        try:
            self.svm_model = pickle.loads(open(self.file_svm_model, 'rb').read())
            self.svm_label = pickle.loads(open(self.file_svm_label, 'rb').read())
        except Exception as ex:
            # print('[Error] No such file ' + self.file_svm_model + ' or ' + self.file_svm_label + '...')
            print('[Error] ' + str(ex))
    
    def save_faces_detected(self, filename_suffix, outPath='./database/detected/'):
        """ 
        Write detected faces (stored in this class) into disk
        filename_suffix: suffix of filename generated, example:
            filename_suffix = 'AJI', then filename will be AJI__1, AJI__2, etc.
        """
        if self.faces_detected == [] or self.faces_detected is None:
            print('[Error] No detected faces stored in this class')
            return

        import uuid
        for (i, face) in enumerate(self.faces_detected):
            print('[Info] writing face image {}/{}'.format(i + 1, len(self.faces_detected)))
            filename = outPath + filename_suffix + self.name_separator + str(uuid.uuid4()) + '.jpg'
            cv2.imwrite(filename, face)
        
    def save_faces_normalized(self, filename_suffix, outPath='./database/normalized/'):
        """ 
        Write normalized (aligned) faces (stored in this class) into disk
        filename_suffix: suffix of filename generated, example:
            filename_suffix = 'AJI', then filename will be AJI__<some guid random1>, AJI__<some guid random1>, etc.
        """
        if self.faces_aligned == [] or self.faces_aligned is None:
            print('[Error] No aligned faces stored in this class')
            return

        import uuid
        for (i, face) in enumerate(self.faces_aligned):
            print('[Info] writing face image {}/{}'.format(i + 1, len(self.faces_detected)))
            filename = outPath + filename_suffix + self.name_separator + str(uuid.uuid4()) + '.jpg'
            cv2.imwrite(filename, face)
    
    #@@ UTILITIES @@#
    def mass_align_images(self, srcPath='./database/raw/', detectedPath='./database/detected/', alignedPath='./database/normalized/'):
        """
        CAUTION: 1 file only contains 1 face need to be detected !!!
                The algorithm will only pick the biggest face if multiple faces appear in  the picture
        Detect faces for all images in srcPath
        Write all detected faces into detectedPath
        Write all normalized faces into alignedPath        
        """
        # self.set_target_face_percentage(0.33)
        imagePaths = list(paths.list_images(srcPath))
        for (i, imagePath) in enumerate(imagePaths):
            head_tail = os.path.split(imagePath)
            # split path and file
            tail = head_tail[1]
            # split filename and extension
            splits = tail.split('.')
            # get filename only without extension
            suffix = splits[0]
            #
            print('[Info] processing image {}/{} ({})'.format(i + 1, len(imagePaths), tail))
            img = cv2.imread(imagePath)
            self.set_image_to_be_processed(img)
            self.fd_method = 'AI'
            self.fd_ai_min_score = 0.6
            self.detect_faces()
            self.align_faces(target_face_percent=0.28) # tight cropped, FaceNet papper
            h0 = 0
            w0 = 0
            face_a = []
            face_d = []
            # only pick biggest face detected
            for (j, detectedImg) in enumerate(self.faces_detected):
                h,w,__ = detectedImg.shape
                if(h*w > h0*w0):
                    h0 = h
                    w0 = w
                    face_d = detectedImg
                    face_a = self.faces_aligned[j]
            filename_d = detectedPath +  suffix + '_detect.jpg'
            filename_a = alignedPath +  suffix + '_align.jpg'
            if face_d != []:
                cv2.imwrite(filename_d, face_d) 
                cv2.imwrite(filename_a, face_a)           
    
    def create_database_from_video(self, videopath, num_sample=3):
        """
        Create aligned & detected face, write into database folder
        """
        from imutils.video import count_frames
        imsizewidth = 300
        random_number = 20
        max_faces = num_sample

        nframe = count_frames(videopath)
        random_frame = []
        for x in range(random_number):
            random_frame.append(random.randint(1,nframe))
        
        dface = 0
        cap = cv2.VideoCapture(videopath)
        for frame_no in random_frame:
            _, frame = cap.read()
            # _, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            print('Get face(s) on frame: ', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            __, frame = cap.read()

            self.set_image_to_be_processed(frame)
            self.detect_faces()
            self.align_faces(target_face_percent=0.28)
            if self.faces_aligned != []:
                self.save_faces_detected(filename_suffix='FROM_VIDEO')
                self.save_faces_normalized(filename_suffix='FROM_VIDEO')
                dface = dface + 1
            
            if dface >= max_faces: 
                break

        cap.release()
        print('Done')
    
    def insert_database_encode_from_video(self, name, videopath, num_sample=3):
        """
        Create detected, aligned and encoding face, 
        write into database folder,
        inserted into encodings.pickle
        """
        from imutils.video import count_frames
        imsizewidth = 300
        random_number = 20
        max_faces = num_sample

        nframe = count_frames(videopath)
        random_frame = []
        for x in range(random_number):
            random_frame.append(random.randint(1,nframe))
        
        dface = 0
        cap = cv2.VideoCapture(videopath)
        for frame_no in random_frame:
            _, frame = cap.read()
            # _, frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            print('Get face on frame: ', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            __, frame = cap.read()

            self.set_image_to_be_processed(frame)
            self.detect_faces()
            self.align_faces(target_face_percent=0.25)
            if self.faces_aligned != []:
                e = self.generate_encoding(self.faces_aligned[0])
                self.insert_encoding(e, name)
                self.save_faces_detected(filename_suffix=name)
                self.save_faces_normalized(filename_suffix=name)
                dface = dface + 1
            
            if dface >= max_faces: 
                break

        cap.release()
        print('Done')
    
    def remove_encoding(self, pos=-1, name=''):
        """
        Remove specific encoding from encodings pickle file
        """
        if(self.load_encodings()):
            encodings = self.data['encodings']
            names = self.data['names']
            if pos > -1:
                del encodings[pos]
                del names[pos]
            else:
                # remove all occurences of name in names
                names = [e for e in names if e not in (name)]

            data = {'encodings' : encodings, 'names' : names}
            self.data = data
            self.save_encodings()

    @staticmethod
    def rename_files(filename_suffix, name_separator, srcPath):
        import uuid
        for filename in os.listdir(srcPath):
            dst = srcPath + filename_suffix + name_separator + str(uuid.uuid4()) + '.jpg'
            src = srcPath + filename
            os.rename(src, dst)
    
    @staticmethod
    def train_svm_model(file_encodings='./models/encodings.pickle', outpath='./models'):
        """
        @file_encodings: pickle file contains encodings where SVM need to be trained to
        @outpath: directory for SVM model to be saved (name is svm_model.pickle)
        """
        # load the face embeddings
        print('[Info] loading face encodings...')
        data = pickle.loads(open(file_encodings, 'rb').read())
        # encode the labels
        print('[Info] encoding labels...')
        le = LabelEncoder()
        labels = le.fit_transform(data['names'])
        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        print('[Info] training SVM model...')
        recognizer = SVC(C=1.0, kernel='linear', probability=True)
        recognizer.fit(data['encodings'], labels)
        # write the actual face recognition model to disk
        print('[Info] serializing SVM model and label...')
        model_filename = outpath + '/svm_model.pickle'
        f = open(model_filename, 'wb')
        f.write(pickle.dumps(recognizer))
        f.close()
        # write the label encoder to disk
        label_filename = outpath + '/svm_label.pickle'
        f = open(label_filename, 'wb')
        f.write(pickle.dumps(le))
        f.close()
        print('[Info] done.')
   
    @staticmethod
    def resize(img, scale=0.3):
        __, w, __ = img.shape
        return imutils.resize(img, width=int(scale*w))        
    
    @staticmethod
    def draw_rectangle(img, rectangles, linewidth=2):
        """
        @img = input image to draw the rectangles
        @rectangles = bounding box with coord (left, right, top, bottom)
        """
        for left, right, top, bottom in rectangles:
            cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),linewidth)
        