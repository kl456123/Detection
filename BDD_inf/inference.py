import os
import cv2
import numpy
import torch
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision import transforms
from collections import OrderedDict
from models.encoder import DataEncoder

#IMG_PATH = '/home/mark/mount/nas2/mercury/zhongguancun/20180710-150000-00708/keyframes'
IMG_PATH='/data/zhengzhou/raw_image/'
SAVE_PATH = './zhongguancun'
MODEL_PATH = './weights/model_epoch10.pkl'
OUT_IMG = True
OUT_TXT = False


class Infer(object):

    def __init__(self, model_name, weight_path=None, data_type='coco80'):
        self.input_shape = None
        self.det_model = None
        self.obj_names = None
        self.colormap = None
        self.encoder = None
        self.id_list = None

        self.data_transfer = self.__set_transfer()

        if model_name == 'retina_dla_bdd':
            from cfgs.model_cfgs.retina_dla_bdd_cfg import ModelCFG
            self.__init_prnet(ModelCFG, weight_path)
            self.input_shape = ModelCFG['input_shape']

        if data_type == 'bdd':
            from cfgs.data_cfgs.bdd100k_cfg import DataCFG
            self.colormap = DataCFG['colormap']
            self.obj_names = DataCFG['obj_names']
            self.anno_type = DataCFG['anno_type']
            self.id_list = DataCFG['id_list']

    def __init_prnet(self, model_cfg, model_path=None, is_local=False):
        from models.PRNet import PRNet
        self.det_model = PRNet(model_cfg)
        self.encoder = DataEncoder(model_cfg, anchor_type=model_cfg['anchor_type'], infer_mode=True)
        if model_path is not None:
            self.__load_weight(model_path, is_local=is_local)
        self.__set_test_mode()

    def inference(self, array_img, save_name=None):
        img = cv2.resize(array_img, self.input_shape)
        img = Image.fromarray(img)

        with torch.no_grad():
            image = self.data_transfer(img).unsqueeze(0).cuda()
            loc1_preds, loc2_preds, os_preds, cls_preds = self.det_model.forward(image)
            boxes, lbls, scores, has_obj = self.encoder.decode(
                loc2_preds.data.squeeze(0), F.softmax(cls_preds.squeeze(0), dim=1).data, os_preds.squeeze(0), Nt=0.5)

        if has_obj:
            boxes = boxes.cpu().numpy()
            boxes = self.__decode_box(boxes, array_img.shape)
            lbls = lbls.cpu().numpy()
            scores = scores.cpu().numpy()
            if OUT_IMG:
                array_img = self.__draw_box(array_img, boxes, lbls, scores)
                cv2.imwrite(save_name, array_img)
        else:
            if OUT_IMG:
                array_img = cv2.cvtColor(array_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(save_name, array_img)

        return boxes, lbls, scores, has_obj

    def predict_in_folder(self, img_path, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        names = sorted(os.listdir(img_path))
        for n in tqdm(names, smoothing=0.):
            img_name = os.path.join(img_path, n)
            save_name = os.path.join(save_path, n)
            img = cv2.cvtColor(cv2.imread(img_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            self.inference(img, save_name)

    def single_infer(self, img_path, save_path):
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        self.inference(img, save_path)

    def update_weight(self, new_weight):
        self.det_model = new_weight

    @staticmethod
    def __set_transfer():
        inference_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize([0.290101, 0.328081, 0.286964],
                                                                        [0.182954, 0.186566, 0.184475])])
        return inference_transforms

    def __set_test_mode(self):
        self.det_model.cuda()
        self.det_model.eval()

    def __load_weight(self, model_path, is_local=False):
        print("loading pre-trained weight")
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)

        if is_local:
            self.det_model.load_state_dict(weight)
        else:
            new_state_dict = OrderedDict()
            for k, v in weight.items():
                name = k[7:]      # remove `module.`
                new_state_dict[name] = v
            self.det_model.load_state_dict(new_state_dict)

    def __draw_box(self, img, box_list, label_list, conf, threshold=0.):
        for bbox, label, cf in zip(box_list, label_list, conf):
            if cf < threshold:
                continue

            xmin = int(bbox[0])
            xmax = int(bbox[2])
            ymin = int(bbox[1])
            ymax = int(bbox[3])

            class_name = self.obj_names[label - 1] + '%.2f' % float(cf)
            c = self.colormap[label - 1]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=c, thickness=2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, class_name, (xmin + 5, ymax - 5), font, fontScale=0.5, color=c, thickness=2)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def __save_json(self, json_file, img_name, box_list, label_list, conf, threshold=0.):
        for bbox, label, cf in zip(box_list, label_list, conf):
            if cf < threshold:
                continue
            xmin = int(bbox[0])
            xmax = int(bbox[2])
            ymin = int(bbox[1])
            ymax = int(bbox[3])
            img_id = img_name.split('/')[-1][:-4]

            obj_dict = {
                "image_id": img_id,
                "category_id": self.id_list[label - 1],
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "score": float(cf)
            }
            json_file.append(obj_dict)

    @staticmethod
    def __decode_box(bboxes, img_shape):
        bboxes = numpy.clip(bboxes, 0., 1.)
        bboxes[:, 0] *= img_shape[1]
        bboxes[:, 2] *= img_shape[1]
        bboxes[:, 1] *= img_shape[0]
        bboxes[:, 3] *= img_shape[0]

        # bboxes[:, 0] -= box_shift[0]
        # bboxes[:, 2] -= box_shift[0]
        # bboxes[:, 1] -= box_shift[1]
        # bboxes[:, 3] -= box_shift[1]
        #
        # bboxes[:, 0] /= img_shape[1]
        # bboxes[:, 2] /= img_shape[1]
        # bboxes[:, 1] /= img_shape[0]
        # bboxes[:, 3] /= img_shape[0]
        #
        # bboxes = numpy.clip(bboxes, 0., 1.)
        #
        # bboxes[:, 0] *= img_shape[1]
        # bboxes[:, 2] *= img_shape[1]
        # bboxes[:, 1] *= img_shape[0]
        # bboxes[:, 3] *= img_shape[0]

        return bboxes

    @staticmethod
    def pad_img(img):
        w, h = img.size
        a = max(w, h)
        box_shift = [0., 0.]

        if w < a:
            w_pl = int((a - w) / 2)
            w_pr = (a - w) - w_pl
            img = ImageOps.expand(img, border=(w_pl, 0, w_pr, 0), fill=(124, 117, 104))
            box_shift[0] = w_pl

        elif h < a:
            h_pu = int((a - h) / 2)
            h_pd = (a - h) - h_pu
            img = ImageOps.expand(img, border=(0, h_pu, 0, h_pd), fill=(124, 117, 104))
            box_shift[1] = h_pu

        return img, box_shift


if __name__ == '__main__':
    infer = Infer(model_name='retina_dla_bdd', weight_path=MODEL_PATH, data_type='bdd')
    # infer.single_infer('/home/mark/OSNet/toy.jpg', './test.jpg')
    infer.predict_in_folder(IMG_PATH, SAVE_PATH)
    # infer.predict_folders(IMG_PATH, SAVE_PATH)
