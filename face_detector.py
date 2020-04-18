import cv2
import dlib
import time

CNN_FACE_DETECTOR_WEIGHTS_FILE = './models/face_models/mmod_human_face_detector.dat'
HAAR_FACE_DETECTOR_MODEL_FILE = './models/face_models/haarcascade_frontalface_alt.xml'


class FaceDetector:
    def __init__(self, cnn_weights_file=CNN_FACE_DETECTOR_WEIGHTS_FILE, haar_model_file=HAAR_FACE_DETECTOR_MODEL_FILE,
                 verbose=False):
        # initialize hog + svm based face detector
        self.hog_face_detector = dlib.get_frontal_face_detector()

        # initialize cnn based face detector with the weights
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_weights_file)

        # initialize Haar-based face detector with frontal face model
        self.haar_face_detector = cv2.CascadeClassifier(haar_model_file)

        self.verbose = verbose
        self.hog_faces = None  # type: object
        self.cnn_faces = None  # type: object
        self.haar_faces = None  # type: object
        self.hog_time = 0
        self.cnn_time = 0
        self.haar_time = 0

    def process_frame(self, image, use_cnn=False, use_hog=True, use_haar=False):
        if use_hog:
            dlib_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            start = time.time()
            # apply face detection (hog)
            self.hog_faces = self.hog_face_detector(dlib_image, 1)
            end = time.time()
            self.hog_time = end - start

        if use_cnn:
            dlib_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            start = time.time()
            # apply face detection (cnn)
            self.cnn_faces = self.cnn_face_detector(dlib_image, 1)
            self.cnn_faces = [face.rect for face in self.cnn_faces]  # convert to DLIB Rect
            end = time.time()
            self.cnn_time = end - start

        if use_haar:
            start = time.time()
            # apply face detection (haar)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            haar_faces = self.haar_face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64)
            )
            end = time.time()
            self.haar_faces = self.convert_opencv_haar_faces_to_dlib_rect(haar_faces)
            self.haar_time = end - start

        if self.verbose:
            print("Execution Time (in seconds) :")
            if use_hog:
                print("HOG : ", format(self.hog_time, '.2f'))
            if use_cnn:
                print("CNN : ", format(self.cnn_time, '.2f'))
            if use_haar:
                print("Haar : ", format(self.haar_time, '.2f'))

    @staticmethod
    def convert_opencv_haar_faces_to_dlib_rect(haar_faces):
        dlib_haar_faces = []
        if len(haar_faces):
            for hf in haar_faces:
                dlib_haar_faces.append(dlib.rectangle(hf[0], hf[1], hf[0] + hf[2], hf[1] + hf[3]))
        return dlib_haar_faces

    @staticmethod
    def convert_dlib_rectangles_to_opencv_faces(dlib_faces):
        result = []
        if dlib_faces:
            for df in dlib_faces:
                x = df.left()
                y = df.top()
                w = df.right() - x
                h = df.bottom() - y
                opencv_face = [x, y, w, h]
                result.append(opencv_face)
        return result


def get_face_regions(image, opencv_faces):
    face_regions = []
    if opencv_faces:
        for face in opencv_faces:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            face_regions.append(image[y:y+h, x:x+w])
    return face_regions
