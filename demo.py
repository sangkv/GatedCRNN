import os
import time

import torch
import torchvision.transforms as transforms
from PIL import Image

import utils
from model import GatedCRNN


# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL
list_of_characters = """aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ îç¤²€û°ëœ"""

converter = utils.CTCLabelConverter(list_of_characters)

num_class = len(converter.character)
input_channel = 1
model = GatedCRNN(num_classes=num_class, input_channel=input_channel)
model.load_state_dict(torch.load('pretrained_model/best_norm_ED_final.pth'))
model.to(device)
model.eval()

# DATA
class ResizeNormalize(object):

    def __init__(self, imgH=32, interpolation=Image.BICUBIC):
        self.imgH = imgH
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        (w, h) = img.size
        imgW = int(w * (self.imgH/h))
        img = img.resize((imgW, self.imgH), self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img.unsqueeze(0)

transforms = ResizeNormalize(imgH=32)

path_test = 'data/test'
list_image = os.listdir(path_test)

# PREDICT
sum_inference_time = 0
with torch.no_grad():
    for image_path in list_image:
        img_path = os.path.join(path_test, image_path)
        img = Image.open(img_path).convert('L')
        image_tensors = transforms(img)
        batch_size = image_tensors.size(0)
        image_tensors = image_tensors.to(device)

        t0 = time.time()
        preds = model(image_tensors)
        t1 = time.time()
        sum_inference_time += (t1-t0)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, preds_size)
        print("\nImage: ", image_path)
        print("Predict: ", preds_str)

time_one_image = sum_inference_time/len(list_image)
print('Time cost for one image: ', time_one_image)
fps = float(1/time_one_image)
print("FPS = {} ".format(fps, '.1f') )

