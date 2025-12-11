
import numpy as np
from   agents.zeroshot.unigoal.utils.visualization.semantic_prediction import SemanticPredMaskRCNN
from   agents.zeroshot.unigoal.configs.categories                                     import name2index
from agents.zeroshot.unigoal.utils.llm import LLM
from   PIL                                 import Image
import re

class PreProcessing():
    def __init__(self,args):
        self.args = args
        self.info = {}

        self.instance_imagegoal = None
        self.sem_pred = SemanticPredMaskRCNN(args)
        self.gt_goal_idx              = None
        self.llm = LLM(self.args.base_url, self.args.api_key, self.args.llm_model)

        self.name2index = name2index
        self.index2name = {v: k for k, v in self.name2index.items()}
        self.prompt_text2object = '"chair: 0, sofa: 1, plant: 2, bed: 3, toilet: 4, tv_monitor: 5" The above are the labels corresponding to each category. Which object is described in the following text? Only response the number of the label and not include other text.\nText: {text}'


    def set_goal_cat_id(self,idx): # LLM infers and passes the index of the object
        self.gt_goal_idx = idx
        return


    def get_goal_name_temp(self):
        self.info['goal_cat_id'] = self.gt_goal_idx
        if self.info['goal_cat_id'] is not None:
            self.info['goal_name'] = self.index2name[self.gt_goal_idx]
            return self.info['goal_name']

        return None

    def get_goal_name(self):

        if self.args.goal_type == 'ins_image':
            self.instance_imagegoal = np.array(Image.open(self.args.image_goal_path))
        elif self.args.goal_type == 'text':
            self.text_goal = self.args.text_goal_prompt

        idx = self.get_goal_cat_id()

        if idx is not None:
            self.set_goal_cat_id(idx)
        return self.get_goal_name_temp()

    def get_gt_goal_idx(self):
        return self.gt_goal_idx

    def pred_sem(self, rgb, depth=None, use_seg=True, pred_bbox=False):
        if pred_bbox:
            semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
            return self.pred_box, seg_predictions
        else:
            if use_seg:
                semantic_pred, self.rgb_vis, self.pred_box, seg_predictions = self.sem_pred.get_prediction(rgb)
                semantic_pred = semantic_pred.astype(np.float32)
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
            return semantic_pred, seg_predictions

    def get_goal_cat_id(self):
        if self.args.goal_type == 'ins_image':
            instance_whwh, seg_predictions = self.pred_sem(self.instance_imagegoal.astype(np.uint8), None, pred_bbox=True)


            ins_whwh = [instance_whwh[i] for i in range(len(instance_whwh)) \
                if (instance_whwh[i][2][3]-instance_whwh[i][2][1])>1/6*self.instance_imagegoal.shape[0] or \
                    (instance_whwh[i][2][2]-instance_whwh[i][2][0])>1/6*self.instance_imagegoal.shape[1]]
            if ins_whwh != []:
                ins_whwh = sorted(ins_whwh,  \
                    key=lambda s: ((s[2][0]+s[2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((s[2][1]+s[2][3]-self.instance_imagegoal.shape[0])/2)**2 \
                    )
                if ((ins_whwh[0][2][0]+ins_whwh[0][2][2]-self.instance_imagegoal.shape[1])/2)**2 \
                        +((ins_whwh[0][2][1]+ins_whwh[0][2][3]-self.instance_imagegoal.shape[0])/2)**2 < \
                            ((self.instance_imagegoal.shape[1] / 6)**2 )*2:
                    return int(ins_whwh[0][0])
            return None
        elif self.args.goal_type == 'text':
            for i in range(10):
                if isinstance(self.text_goal, dict) and 'intrinsic_attributes' in self.text_goal:
                    text_goal = self.text_goal['intrinsic_attributes']
                else:
                    text_goal = self.text_goal

                text_goal_id = self.llm(self.prompt_text2object.replace('{text}', text_goal))


                try:
                    text_goal_id = re.findall(r'\d+', text_goal_id)[0]
                    text_goal_id = int(text_goal_id)
                    if 0 <= text_goal_id < 7:
                        return text_goal_id
                except:
                    pass
            return 0
