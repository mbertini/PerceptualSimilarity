import cv2
import argparse
from face_detector import FaceDetector, get_face_regions
from MSSIM import compute_MSSIM
import numpy as np
import fnmatch
import os
import models
from util import util

DEBUG = True
VERBOSE_DEBUG = False

BRISQUE_MODEL_FILE = './models/brisque_models/brisque_model_live.yml'
BRISQUE_RANGE_FILE = './models/brisque_models/brisque_range_live.yml'


def draw_faces(image, cnn_faces, hog_faces, haar_faces):
    if hog_faces:
        # loop over HOG detected faces
        for face in hog_faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            print("orig: x: " + str(x) + " - y: " + str(y) + " - w: " + str(w) + " - h: " + str(h))
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            new_x, new_y, new_w, new_h = compute_64x64_multiple_faces(x, y, w, h)
            if new_x != x or new_y != y or new_w != w or new_h != h:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 128), 2)

    if cnn_faces:
        # loop over CNN detected faces
        for face in cnn_faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            new_x, new_y, new_w, new_h = compute_64x64_multiple_faces(x, y, w, h)
            if new_x != x or new_y != y or new_w != w or new_h != h:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 128), 2)

    if haar_faces:
        # loop over Haar detected faces
        for face in haar_faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            new_x, new_y, new_w, new_h = compute_64x64_multiple_faces(x, y, w, h)
            if new_x != x or new_y != y or new_w != w or new_h != h:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 128), 2)

    # write at the top left corner of the image
    # for color identification
    img_height, img_width = image.shape[:2]
    cv2.putText(image, "HOG", (img_width - 50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.putText(image, "CNN", (img_width - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    cv2.putText(image, "Haar", (img_width - 50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2)

    return image


def resize_64x64_multiple_faces_list(faces):
    result = []
    for face in faces:
        x, y, w, h = compute_64x64_multiple_faces(face[0], face[1], face[2], face[3])
        resized_face = [x, y, w, h]
        result.append(resized_face)
    return result


def compute_64x64_multiple_faces(x, y, w, h):
    extra_w = extra_h = left_extra = right_extra = upper_extra = lower_extra = 0
    if w % 64:
        extra_w = 64 - (w % 64)
        if extra_w % 2:
            left_extra = extra_w // 2 + 1
            right_extra = extra_w // 2
        else:
            left_extra = right_extra = extra_w // 2
    if h % 64:
        extra_h = 64 - (h % 64)
        if extra_h % 2:
            upper_extra = extra_h // 2 + 1
            lower_extra = extra_h // 2
        else:
            upper_extra = lower_extra = extra_h // 2
    x -= upper_extra
    y -= left_extra
    w += extra_w
    h += extra_h
    if VERBOSE_DEBUG:
        if left_extra == right_extra == upper_extra == lower_extra == 0:
            print("64x64 multiples pixels face")
        else:
            print("new: x: {0} - y: {1} - w: {2} - h: {3}".format(str(x), str(y), str(w), str(h)))
    return x, y, w, h


def get_64x64_face_regions(face_region):
    width = face_region.shape[0]
    height = face_region.shape[1]
    blocks = np.array([face_region[i:i + 64, j:j + 64] for j in range(0, width, 64) for i in range(0, height, 64)])
    return blocks


def main():
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d0', '--dir0', required=True, type=str, default='./imgs/ex_dir0',
                    help='Reference videos directory')
    ap.add_argument('-m0', '--mask0', required=False, type=str, default='*.mp4',
                    help='file mask for files in d0 (e.g. .mp4)')
    ap.add_argument('-d1', '--dir1', required=True, type=str, default='./imgs/ex_dir1', help='Other videos directory')
    ap.add_argument('-m1', '--mask1', required=False, type=str, default='*.mkv',
                    help='file mask for files in d1 (e.g. .mp4)')
    ap.add_argument('-o', '--out', required=True, type=str, default='./results/',
                    help='Distance files (.csv) directory')
    ap.add_argument('--use_gpu', action='store_true', help='Flag to use GPU to compute distance')
    ap.add_argument('-w', '--weights', default='./models/face_models/mmod_human_face_detector.dat',
                    help='path to face model file')
    ap.add_argument('-H', '--hog_detector', required=False, action='store_true', help='use HOG detector')
    ap.add_argument('-A', '--haar_detector', required=False, action='store_true', help='use Haar detector')
    ap.add_argument('-C', '--cnn_detector', required=False, action='store_true', help='use CNN-based detector')
    ap.add_argument('-v', '--verbose', required=False, help='verbose output', action='store_true')
    args = ap.parse_args()

    print("Input dir0:", args.dir0)
    print("Input dir0 file mask:", args.mask0)
    print("Input dir1:", args.dir1)
    print("Input dir1 file mask:", args.mask1)
    print("Output dir: ", args.out)
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if args.use_gpu:
        print("Compute LPIPS similarity using GPU (fast)")
    else:
        print("Compute LPIPS similarity using CPU (slow)")
    if args.hog_detector:
        print("Find faces using DLib HOG detector")
    if args.haar_detector:
        print("Find faces using OpenCV Haar detector")
    if args.cnn_detector:
        print("Find faces using DLib CNN detector")
    if not args.hog_detector and not args.haar_detector and not args.cnn_detector:
        args.haar_detector = True
        print("No face detector selected. Using default OpenCV Haar detector")

    dir0_files = []
    for file in os.listdir(args.dir0):
        if fnmatch.fnmatch(file, args.mask0):
            dir0_files.append(file)
    print("Dir 0 (ref videos) contains " + str(len(dir0_files)) + " " + args.mask0 + " files.")
    dir1_files = []
    for file in os.listdir(args.dir1):
        if fnmatch.fnmatch(file, args.mask1):
            dir1_files.append(file)
    print("Dir 1 (mod videos) contains " + str(len(dir1_files)) + " " + args.mask1 + " files.")

    ## Initializing face detector
    fd = FaceDetector()
    ## Initializing LPIPS perceptual quality model
    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=args.use_gpu)
    brisque_model = cv2.quality_QualityBRISQUE.create(BRISQUE_MODEL_FILE, BRISQUE_RANGE_FILE)

    # process all files
    num_all_files = len(dir0_files)
    processed_file = 1
    for reference_video_name, modified_video_name in zip(dir0_files, dir1_files):
        print("Processing ref. video " + str(processed_file) + "/" + str(num_all_files))
        print("Ref. video: " + reference_video_name + " - Mod. video: " + modified_video_name)
        out_file_name_LPIPS = os.path.join(args.out, os.path.splitext(reference_video_name)[0] + "-" +
                                           os.path.splitext(modified_video_name)[0] + "-LPIPS.csv")
        out_file_name_MSSIM = os.path.join(args.out, os.path.splitext(reference_video_name)[0] + "-" +
                                           os.path.splitext(modified_video_name)[0] + "-MSSIM.csv")
        out_file_name_BRISQUE = os.path.join(args.out, os.path.splitext(reference_video_name)[0] + "-" +
                                             os.path.splitext(modified_video_name)[0] + "-BRISQUE.csv")
        # open files and write headers
        out_file_LPIPS = open(out_file_name_LPIPS, 'w')
        out_file_MSSIM = open(out_file_name_MSSIM, 'w')
        out_file_BRISQUE = open(out_file_name_BRISQUE, 'w')
        out_file_BRISQUE.writelines('frame_num, BRISQUE score ref,  BRISQUE score mod\n')
        out_file_MSSIM.writelines('frame_num, SSIM R, SSIM G, SSIM B\n')
        out_file_LPIPS.writelines('frame_num, LPIPS distance\n')
        # open reference and modified videos
        reference_video = cv2.VideoCapture(os.path.join(args.dir0, reference_video_name))
        modified_video = cv2.VideoCapture(os.path.join(args.dir1, modified_video_name))
        frame_count = 0
        while reference_video.isOpened() and modified_video.isOpened():
            if args.verbose and (frame_count % 100 == 0):
                print("Frame: {}".format(frame_count))
            ref_ret_val, ref_image = reference_video.read()
            mod_ret_val, mod_image = modified_video.read()
            if not ref_ret_val:
                break
            if not mod_ret_val:
                break
            fd.process_frame(ref_image, use_cnn=args.cnn_detector, use_hog=args.hog_detector,
                             use_haar=args.haar_detector)
            detected_faces = fd.convert_dlib_rectangles_to_opencv_faces(fd.cnn_faces) + \
                             fd.convert_dlib_rectangles_to_opencv_faces(fd.hog_faces) + \
                             fd.convert_dlib_rectangles_to_opencv_faces(fd.haar_faces)
            resized_faces = resize_64x64_multiple_faces_list(detected_faces)

            ref_face_regions = get_face_regions(ref_image, resized_faces)
            mod_face_regions = get_face_regions(mod_image, resized_faces)
            for ref_face_region, mod_face_region in zip(ref_face_regions, mod_face_regions):
                MSSIM_dist = compute_MSSIM(ref_face_region, mod_face_region)
                BRISQUE_score_ref = brisque_model.compute(ref_face_region)
                BRISQUE_score_mod = brisque_model.compute(mod_face_region)
                print("{}, {:.6f}, {:.6f}".format(frame_count, BRISQUE_score_ref[0], BRISQUE_score_mod[0]),
                      file=out_file_BRISQUE)
                print('{}, {:.6f}, {:.6f}, {:.6f}'.format(frame_count, round(MSSIM_dist[2] * 100, 2),
                                                          round(MSSIM_dist[1] * 100, 2),
                                                          round(MSSIM_dist[0] * 100, 2)), file=out_file_MSSIM)
                ref_face_blocks = get_64x64_face_regions(ref_face_region)
                mod_face_blocks = get_64x64_face_regions(mod_face_region)
                LPIPS_dist = 0
                block_count = 0
                for ref_face_block, mod_face_block in zip(ref_face_blocks, mod_face_blocks):
                    img0 = util.im2tensor(cv2.cvtColor(ref_face_block, cv2.COLOR_BGR2RGB))  # RGB image from [-1,1]
                    img1 = util.im2tensor(cv2.cvtColor(mod_face_block, cv2.COLOR_BGR2RGB))
                    if args.use_gpu:
                        img0 = img0.cuda()
                        img1 = img1.cuda()
                    # Compute distance
                    LPIPS_dist += model.forward(img0, img1)
                    block_count += 1
                out_file_LPIPS.writelines('%d, %.6f\n' % (frame_count, LPIPS_dist / block_count))

            if DEBUG:
                processed_image = draw_faces(ref_image.copy(), cnn_faces=fd.cnn_faces, hog_faces=fd.hog_faces,
                                             haar_faces=fd.haar_faces)
                cv2.imshow('Processed video', processed_image)
                out_img_filename = "/tmp/" + reference_video_name + "-" + str(frame_count) + ".jpg"
                cv2.imwrite(out_img_filename, processed_image)
                deb_face_count = 0
                ref_face_regions = get_face_regions(ref_image, resized_faces)
                for ref_detected_face in ref_face_regions:
                    out_img_filename = "/tmp/" + reference_video_name + "-frame_" + str(frame_count) + "_-_face_" + str(
                        deb_face_count) + ".jpg"
                    cv2.imwrite(out_img_filename, ref_detected_face)
                    ref_face_blocks = get_64x64_face_regions(ref_detected_face)
                    deb_block_count = 0
                    for ref_face_block in ref_face_blocks:
                        out_img_filename = "/tmp/" + reference_video_name + "-frame_" + str(
                            frame_count) + "_-_face_" + str(
                            deb_face_count) + "_-_block_" + str(deb_block_count) + ".jpg"
                        cv2.imwrite(out_img_filename, ref_face_block)
                        deb_block_count += 1
                    deb_face_count += 1
            if cv2.waitKey(1) == 27:
                break  # esc to quit

            frame_count += 1

        if DEBUG:
            cv2.destroyAllWindows()

        out_file_LPIPS.close()
        out_file_MSSIM.close()
        out_file_BRISQUE.close()


if __name__ == '__main__':
    main()
