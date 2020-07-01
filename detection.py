import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import csv


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='backbone network mobile0.25 or resnet50')
parser.add_argument('--cuda', default=0, type=int, help='cuda device to run on')
parser.add_argument('--origin_size', default=True, type=str, help='whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='results', type=str, help='dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='use cpu inference')
parser.add_argument('--dataset_folder', default='casia', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.1, type=float, help='nms_threshold')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--ref', default=None, type=str, help='reference file in resources folder')
parser.add_argument('--skip_n_first', default=0, type=int, help='number of files to skip')
args = parser.parse_args()


os.makedirs("results", exist_ok=True)
os.makedirs("resources", exist_ok=True)
os.makedirs("images", exist_ok=True)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device(f"cuda:{args.cuda}")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def rect_intersect_area(rect1, rect2):
    x11, y11, x12, y12 = rect1
    x21, y21, x22, y22 = rect2
    x_overlap = max(0, min(x12, x22) - max(x11, x21))
    y_overlap = max(0, min(y12, y22) - max(y11, y21))
    return x_overlap * y_overlap


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else f"cuda:{args.cuda}")
    net = net.to(device)

    # testing dataset
    testset_folder = f"images/{args.dataset_folder}/"
    testset_list = f"resources/{args.ref}"

    test_dataset = np.genfromtxt(testset_list, delimiter=",", dtype=np.str)
    test_dataset = np.atleast_1d(test_dataset)
    num_images = test_dataset.shape[0]

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    known_bounding_box = True if test_dataset.ndim > 1 else False

    # testing begin
    for i, img in enumerate(test_dataset):
        if i < args.skip_n_first:
            continue

        if known_bounding_box:
            image_path = img[0]
            full_image_path = testset_folder + image_path
            true_x1, true_y1, true_x2, true_y2 = np.array(img[1:5], dtype=int)
        else:
            image_path = img
            full_image_path = testset_folder + img
            true_x1, true_y1, true_x2, true_y2 = -1, -1, -1, -1
        

        if not os.path.exists(full_image_path):
            print(f"File {image_path} not found")
            continue

        img_raw = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        
        height, width, channels = img_raw.shape
        img_size_min = min(height, width)
        img_size_max = max(height, width)
        
        resize = float(target_size) / float(img_size_min)

        # prevent bigger axis from being more than max_size:
        if np.round(resize * img_size_max) > max_size:
            resize = float(max_size) / float(img_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        # forward pass
        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  
        _t['forward_pass'].toc()
        
        # decode 
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        bounding_boxes = []
        landmarks = []
        areas = []

        for det in dets:
            
            x1, y1, x2, y2 = [int(det[d]) for d in range(4)]
            confidence = str(det[4])

            bounding_boxes.append([x1, y1, x2, y2, confidence])
            landmarks.append([int(det[d]) for d in range(5, 15)])

            if known_bounding_box:
                area = rect_intersect_area([x1, y1, x2, y2], [true_x1, true_y1, true_x2, true_y2])
                areas.append(area)
            else:
                break

        # save boxes and landmarks

        if len(areas) > 0:

            max_area_idx = np.argmax(areas)
            
            boxes = bounding_boxes[max_area_idx]
            landmarks = landmarks[max_area_idx]

        elif len(bounding_boxes) > 0:

            boxes = bounding_boxes[0]
            landmarks = landmarks[0]

        else:

            boxes = [true_x1, true_y1, true_x2, true_y2, -1]
            landmarks = [*([-1 for _ in range(5, 15)])]

            with open(f'{args.save_folder}/{args.dataset_folder}_not_found.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([image_path])
                

        with open(f'{args.save_folder}/{args.dataset_folder}_boxes.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([image_path, *boxes])

        with open(f'{args.save_folder}/{args.dataset_folder}_landmarks.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([image_path, *landmarks])


        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # save image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms BGR order
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            
            # save image
            paths = full_image_path.split('/')
            if len(paths) > 1:
                path_to_file = "results/" + '/'.join(path for path in paths[:len(paths)-1])
                os.makedirs(path_to_file, exist_ok=True)
                file_name = f'{path_to_file}/{paths[len(paths)-1]}'
            else:
                file_name = "results/" + full_image_path
            
            cv2.imwrite(file_name, img_raw)