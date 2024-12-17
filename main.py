import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from lib.qort_former import QORT_Former
import cv2
import torchvision.transforms as T
from PIL import Image

def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument('--source', default='', type=str)
    parser.add_argument('--model_path', default='h2o_model.pth', type=str)
    parser.add_argument('--img_size', default=(540, 960), type=tuple)
    return parser

def visualize(cv_img, img_points, mode='left'):
    if mode == 'left':
        color = (255,0,0)
    else:
        color = (0,0,255)
    line_thickness = 2
    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[2][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[3][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[4][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[5][:-1]), tuple(
        img_points[6][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[6][:-1]), tuple(
        img_points[7][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[7][:-1]), tuple(
        img_points[8][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[9][:-1]), tuple(
        img_points[10][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[10][:-1]), tuple(
        img_points[11][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[11][:-1]), tuple(
        img_points[12][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[13][:-1]), tuple(
        img_points[14][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[14][:-1]), tuple(
        img_points[15][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[15][:-1]), tuple(
        img_points[16][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[17][:-1]), tuple(
        img_points[18][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[18][:-1]), tuple(
        img_points[19][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[19][:-1]), tuple(
        img_points[20][:-1]), color, line_thickness)

    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[1][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[5][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[9][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[13][:-1]), color, line_thickness)
    cv2.line(cv_img, tuple(img_points[0][:-1]), tuple(
        img_points[17][:-1]), color, line_thickness)

    return cv_img


def visualize_obj(cv_img, img_points):
    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[2][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[3][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[4][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[4][:-1]), tuple(
        img_points[1][:-1]), (0, 255, 0), 5)

    cv2.line(cv_img, tuple(img_points[1][:-1]), tuple(
        img_points[5][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[2][:-1]), tuple(
        img_points[6][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[3][:-1]), tuple(
        img_points[7][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[4][:-1]), tuple(
        img_points[8][:-1]), (0, 255, 0), 5)

    cv2.line(cv_img, tuple(img_points[5][:-1]), tuple(
        img_points[6][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[6][:-1]), tuple(
        img_points[7][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[7][:-1]), tuple(
        img_points[8][:-1]), (0, 255, 0), 5)
    cv2.line(cv_img, tuple(img_points[8][:-1]), tuple(
        img_points[5][:-1]), (0, 255, 0), 5)
    return cv_img

def inference(model, source, device, img_size):
    model.eval()
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = T.Compose([
        T.Resize(img_size),
        normalize
    ])
    files = os.listdir(source)
    for fl in tqdm(files):
        if os.path.isdir(os.path.join(source,fl)):
            continue
        source_img = cv2.imread(os.path.join(source,fl))
        img_tensor = transform(Image.open(os.path.join(source,fl)).convert('RGB')).to(device).unsqueeze(0)
        outputs = model(img_tensor)
        out_logits, pred_keypoints, pred_obj_keypoints = outputs['pred_logits'], outputs['pred_keypoints'], \
            outputs['pred_obj_keypoints']
        prob = out_logits.sigmoid()
        B, num_queries, num_classes = prob.shape
        best_score = torch.zeros(B).to(device)
        obj_idx = torch.zeros(B).to(device).to(torch.long)
        for i in range(1, 9):
            score, idx = torch.max(prob[:, :, i], dim=-1)
            obj_idx[best_score < score] = idx[best_score < score]
            best_score[best_score < score] = score[best_score < score]
        left_idx = torch.argmax(prob[:, :, 9], dim=-1)
        right_idx = torch.argmax(prob[:, :, 10], dim=-1)
        keep = torch.cat([left_idx[:, None], right_idx[:, None], obj_idx[:, None]], dim=-1)
        labels = torch.gather(out_logits, 1, keep.unsqueeze(2).repeat(1, 1, num_classes)).softmax(dim=-1)
        left_kp = torch.gather(pred_keypoints[0], 1, left_idx.unsqueeze(1).unsqueeze(1).repeat(1, 1, 63)).reshape(
            B, 21, 3)
        right_kp = torch.gather(pred_keypoints[1], 1,
                                right_idx.unsqueeze(1).unsqueeze(1).repeat(1, 1, 63)).reshape(B, 21, 3)
        obj_kp = torch.gather(pred_obj_keypoints, 1,
                              obj_idx.unsqueeze(1).unsqueeze(1).repeat(1, 1, 63)).reshape(B, 21, 3)
        # denormalize
        im_h, im_w = torch.as_tensor(source_img.shape[0]).unsqueeze(0), torch.as_tensor(source_img.shape[1]).unsqueeze(0)
        target_sizes = torch.cat([im_w.unsqueeze(-1), im_h.unsqueeze(-1)], dim=-1)
        target_sizes = target_sizes.to(device)
        left_kp[..., :2] *= target_sizes.unsqueeze(1)
        left_kp[..., 2] *= 1000
        right_kp[..., :2] *= target_sizes.unsqueeze(1)
        right_kp[..., 2] *= 1000
        obj_kp[..., :2] *= target_sizes.unsqueeze(1)
        obj_kp[..., 2] *= 1000
        key_points = torch.cat([left_kp.unsqueeze(1), right_kp.unsqueeze(1), obj_kp.unsqueeze(1)], dim=1)
        source_img = visualize(source_img, key_points[0][0].detach().cpu().numpy().astype(np.int32), 'left')
        source_img = visualize(source_img, key_points[0][1].detach().cpu().numpy().astype(np.int32), 'right')
        source_img = visualize_obj(source_img,key_points[0][2].detach().cpu().numpy().astype(np.int32))
        output_dir = os.path.join(source, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, fl)
        cv2.imwrite(img_path, source_img)





if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model_path, weights_only=True)
    model = QORT_Former(num_classes=11).to(device)
    model.load_state_dict(checkpoint)
    inference(model, args.source, device, args.img_size)