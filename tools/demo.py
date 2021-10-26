from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import ipdb

from mmdet.apis.inference import show_result_pyplot
config_file = 'configs/1_ZSOR/faster_rcnn_mobilenetv2_fpn_baseline_ZSOR.py'
checkpoint_file = 'work_dirs/faster_rcnn_mobilenetv2_fpn_baseline_ZSOR/latest.pth'

# ipdb.set_trace()
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
imgs_path = []
for _,_,files in os.walk("datasets/ZSOR/test_data_A/Novels",topdown=True):
    for file in files:
        imgs_path.append(os.path.join("datasets/ZSOR/test_data_A/Novels", file))
imgs = []
for img_path in imgs_path:
    imgs.append(mmcv.imread(img_path))

# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, imgs[0]) 
# visualize the results in a new window
# show_result_pyplot(img, result, model.CLASSES)
# or save the visualization results to image files
show_result_pyplot(imgs[0], result, model.CLASSES, out_file='result.jpg')
pass