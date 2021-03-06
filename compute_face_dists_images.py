import cv2
import argparse
from face_detector import FaceDetector, get_face_regions
from MSSIM import compute_MSSIM
import numpy as np
import fnmatch
import os
import models
from util import util
import random

DEBUG = False
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
    width = face_region.shape[0]  # TODO invert width and height. Keep it for the current experiments...
    height = face_region.shape[1]
    blocks = np.array([face_region[i:i + 64, j:j + 64] for j in range(0, width, 64) for i in range(0, height, 64)])
    return blocks


def get_64x64_full_image_regions(full_image):
    crop_size = 64
    width = full_image.shape[1]
    height = full_image.shape[0]
    all_blocks = np.array(
        [full_image[i:i + crop_size, j:j + crop_size] for j in range(0, width - (width % crop_size), crop_size) for i in
         range(0, height - (height % crop_size), crop_size)])
    max_blocks = 15
    if max_blocks > len(all_blocks):
        return all_blocks
    else:
        subsampled_indexes = random.sample(range(0, len(all_blocks)), max_blocks)
        subsampled_blocks = [all_blocks[i] for i in subsampled_indexes]
        return subsampled_blocks


def main():
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d0', '--dir0', required=True, type=str, default='./imgs/ex_dir0',
                    help='Reference images directory')
    ap.add_argument('-m0', '--mask0', required=False, type=str, default='*.png',
                    help='file mask for files in d0 (e.g. .png)')
    ap.add_argument('-d1', '--dir1', required=True, type=str, default='./imgs/ex_dir1', help='Other images directory')
    ap.add_argument('-m1', '--mask1', required=False, type=str, default='*.png',
                    help='file mask for files in d1 (e.g. .png)')
    ap.add_argument('-o', '--out', required=False, type=str, default='./results/',
                    help='Distance files (.csv) directory')
    ap.add_argument('--use_gpu', action='store_true', help='Flag to use GPU to compute distance')
    ap.add_argument('-w', '--weights', default='./models/face_models/mmod_human_face_detector.dat',
                    help='path to face model file')
    ap.add_argument('-H', '--hog_detector', required=False, action='store_true', help='use HOG detector')
    ap.add_argument('-A', '--haar_detector', required=False, action='store_true', help='use Haar detector')
    ap.add_argument('-C', '--cnn_detector', required=False, action='store_true', help='use CNN-based detector')
    ap.add_argument('-N', '--no_face_detector', required=False, action='store_true',
                    help='Do NOT perform face detection. Compute quality over whole image')
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
    if args.no_face_detector:
        print("Do NOT perform face detection. Compute frame quality over the whole image.")
        file_det_name = "-no-face"
    else:
        if args.hog_detector:
            print("Find faces using DLib HOG detector")
        if args.haar_detector:
            print("Find faces using OpenCV Haar detector")
        if args.cnn_detector:
            print("Find faces using DLib CNN detector")
        if not args.hog_detector and not args.haar_detector and not args.cnn_detector:
            args.haar_detector = True
            print("No face detector selected. Using default OpenCV Haar detector")
        file_det_name = ""

    # it is expected that directories contain same number of files that can be sorted so that one is associated with the
    # file in the corresponding position in the other directory
    dir0_files = []
    for file in os.listdir(args.dir0):
        if fnmatch.fnmatch(file, args.mask0):
            dir0_files.append(file)
    dir0_files.sort()
    print("Dir 0 (ref images) contains " + str(len(dir0_files)) + " " + args.mask0 + " files.")
    dir1_files = []
    for file in os.listdir(args.dir1):
        if fnmatch.fnmatch(file, args.mask1):
            dir1_files.append(file)
    dir1_files.sort()
    print("Dir 1 (mod images) contains " + str(len(dir1_files)) + " " + args.mask1 + " files.")

    # Initializing face detector
    fd = FaceDetector()
    # Initializing LPIPS perceptual quality model
    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=args.use_gpu)
    brisque_model = cv2.quality_QualityBRISQUE.create(BRISQUE_MODEL_FILE, BRISQUE_RANGE_FILE)

    # process all files
    num_all_files = len(dir0_files)
    # open files and write headers
    out_file_base_name = "all_files-" + os.path.basename(args.dir0) + "-" + os.path.basename(
                        args.dir1) + file_det_name
    print("Base file name:" + out_file_base_name)
    out_file_LPIPS_all = open(
        os.path.join(args.out, out_file_base_name + "-LPIPS.csv"), 'w')
    out_file_MSSIM_RGB_all = open(
        os.path.join(args.out, out_file_base_name + "-MSSIM-RGB.csv"), 'w')
    out_file_MSSIM_Y_all = open(
        os.path.join(args.out, out_file_base_name + "-MSSIM-Y.csv"), 'w')
    out_file_BRISQUE_all = open(
        os.path.join(args.out, out_file_base_name + "-BRISQUE.csv"), 'w')
    out_file_BRISQUE_all.writelines('files, BRISQUE score ref,  BRISQUE score mod\n')
    out_file_MSSIM_RGB_all.writelines('files, SSIM R, SSIM G, SSIM B\n')
    out_file_MSSIM_Y_all.writelines('files, SSIM Y\n')
    out_file_LPIPS_all.writelines('files, LPIPS distance\n')
    processed_file = 1
    for reference_image_name, modified_image_name in zip(dir0_files, dir1_files):
        if processed_file % 50 == 0:
            print("Processing ref. image " + str(processed_file) + "/" + str(num_all_files))

        # open reference and modified images
        ref_image = cv2.imread(os.path.join(args.dir0, reference_image_name))
        mod_image = cv2.imread(os.path.join(args.dir1, modified_image_name))
        if ref_image is None or mod_image is None:
            # print("Skipping images: {} and {}".format(reference_image_name, modified_image_name))
            continue
        filename_all = os.path.splitext(reference_image_name)[0] + "-" + os.path.splitext(modified_image_name)[0]

        if args.no_face_detector:  # compute quality over whole image. No face detection
            ref_face_regions = [ref_image]
            mod_face_regions = [mod_image]
        else:  # perform face detection. use face regions to compute quality
            fd.process_frame(ref_image, use_cnn=args.cnn_detector, use_hog=args.hog_detector,
                             use_haar=args.haar_detector)
            detected_faces = fd.convert_dlib_rectangles_to_opencv_faces(fd.cnn_faces) + \
                             fd.convert_dlib_rectangles_to_opencv_faces(fd.hog_faces) + \
                             fd.convert_dlib_rectangles_to_opencv_faces(fd.haar_faces)
            resized_faces = resize_64x64_multiple_faces_list(detected_faces)
            ref_face_regions = get_face_regions(ref_image, resized_faces)
            mod_face_regions = get_face_regions(mod_image, resized_faces)

        for ref_face_region, mod_face_region in zip(ref_face_regions, mod_face_regions):
            MSSIM_Y_dist = compute_MSSIM(cv2.cvtColor(ref_face_region, cv2.COLOR_RGB2GRAY),
                                         cv2.cvtColor(mod_face_region, cv2.COLOR_RGB2GRAY))
            MSSIM_RGB_dist = compute_MSSIM(ref_face_region, mod_face_region)
            BRISQUE_score_ref = brisque_model.compute(ref_face_region)
            BRISQUE_score_mod = brisque_model.compute(mod_face_region)
            print("{}, {:.6f}, {:.6f}".format(filename_all, BRISQUE_score_ref[0], BRISQUE_score_mod[0]),
                  file=out_file_BRISQUE_all)
            print('{}, {:.6f}'.format(filename_all, round(MSSIM_Y_dist[0] * 100, 2)), file=out_file_MSSIM_Y_all)
            print('{}, {:.6f}, {:.6f}, {:.6f}'.format(filename_all, round(MSSIM_RGB_dist[2] * 100, 2),
                                                      round(MSSIM_RGB_dist[1] * 100, 2),
                                                      round(MSSIM_RGB_dist[0] * 100, 2)), file=out_file_MSSIM_RGB_all)

            if args.no_face_detector:
                ref_face_blocks = get_64x64_full_image_regions(ref_face_region)
                mod_face_blocks = get_64x64_full_image_regions(mod_face_region)
            else:
                ref_face_blocks = get_64x64_face_regions(ref_face_region)
                mod_face_blocks = get_64x64_face_regions(mod_face_region)
            LPIPS_dist = 0
            for ref_face_block, mod_face_block in zip(ref_face_blocks, mod_face_blocks):
                img0 = util.im2tensor(cv2.cvtColor(ref_face_block, cv2.COLOR_BGR2RGB))  # RGB image from [-1,1]
                img1 = util.im2tensor(cv2.cvtColor(mod_face_block, cv2.COLOR_BGR2RGB))
                if args.use_gpu:
                    img0 = img0.cuda()
                    img1 = img1.cuda()
                # Compute distance
                LPIPS_dist += model.forward(img0, img1)
            LPIPS_dist /= len(ref_face_blocks)
            out_file_LPIPS_all.writelines('%s, %.6f\n' % (filename_all, LPIPS_dist))

        if DEBUG:
            processed_image = draw_faces(ref_image.copy(), cnn_faces=fd.cnn_faces, hog_faces=fd.hog_faces,
                                         haar_faces=fd.haar_faces)
            cv2.imshow('Processed image', processed_image)
            out_img_filename = "/tmp/" + reference_image_name + ".jpg"
            cv2.imwrite(out_img_filename, processed_image)
            deb_face_count = 0
            ref_face_regions = get_face_regions(ref_image, resized_faces)
            for ref_detected_face in ref_face_regions:
                out_img_filename = "/tmp/" + reference_image_name + "-_face_" + str(deb_face_count) + ".jpg"
                cv2.imwrite(out_img_filename, ref_detected_face)
                ref_face_blocks = get_64x64_face_regions(ref_detected_face)
                deb_block_count = 0
                for ref_face_block in ref_face_blocks:
                    out_img_filename = "/tmp/" + reference_image_name + "-_face_" + str(
                        deb_face_count) + "_-_block_" + str(deb_block_count) + ".jpg"
                    cv2.imwrite(out_img_filename, ref_face_block)
                    deb_block_count += 1
                deb_face_count += 1
            if cv2.waitKey(1) == 27:
                break  # esc to quit

        processed_file += 1

    if DEBUG:
        cv2.destroyAllWindows()
    out_file_LPIPS_all.close()
    out_file_MSSIM_RGB_all.close()
    out_file_MSSIM_Y_all.close()
    out_file_BRISQUE_all.close()


if __name__ == '__main__':
    main()
