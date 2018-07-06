import sys
import cv2

sys.path.append('/workspace/mnt/group/face-det/zhubin/caffe/python')
import caffe
import numpy as np
import random
import cPickle as pickle
import os


# def random_crop(img, size, cop_size, flag, xy=[]):
#     if flag == 0:
#         h_off = random.randint(0, size - cop_size)
#         w_off = random.randint(0, size - cop_size)
#         xy = [xy[0] - w_off, xy[1] - h_off, xy[2] - w_off, xy[3] - h_off, xy[4] - w_off, xy[5] - h_off, xy[6] - w_off,
#               xy[7] - h_off, xy[8] - w_off, xy[9] - h_off]
#         crop_img = img[h_off:h_off + cop_size, w_off:w_off + cop_size]
#         return crop_img, xy
#     if flag == 1:
#         h_off = random.randint(0, size - cop_size)
#         w_off = random.randint(0, size - cop_size)
#         crop_img = img[h_off:h_off + cop_size, w_off:w_off + cop_size]
#         return crop_img

def random_crop(img, im_size, crop_size, flag=1):
    # if flag==1 use random_crop
    # if falg==0 use mid_crop
    if flag == 1:
        h_off = random.randint(0, im_size - crop_size)
        w_off = random.randint(0, im_size - crop_size)
    else:
        h_off = (im_size-crop_size)/2
        w_off = (im_size-crop_size)/2
    crop_img = img[h_off: h_off + crop_size, w_off: w_off + crop_size]
    return crop_img


def mirror(img):
    img = cv2.flip(img, 1)
    return img


def illumination(img):
    r0 = random.uniform(0.0, 1.0)
    r1 = random.uniform(0.0, 1.0)
    hue = r0 * 0.1 + 1
    exposure = r1 * 0.5 + 1
    if random.uniform(0.0, 1.0) > 0.5:
        exposure = 1.0 / exposure
    if random.uniform(0.0, 1.0) > 0.5:
        hue = 1.0 / hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] * exposure
    v = np.clip(v, 0, 255)
    # v = v.astype(np.int32)
    h = hsv[:, :, 0] * hue
    h = np.clip(h, 0, 255)
    # h = h.astype(np.int32)

    hsv[:, :, 2] = v.astype(np.uint8)
    hsv[:, :, 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# read img list
def readSrcFile(src_file):
        f = open(src_file, 'r')
        imgTuple = []
        for line in f.readlines():
            temp = line.split(" ")
            imgTuple.append((temp[0], int(temp[1]), int(temp[2])))
        return imgTuple

################################################################################
#########################Train Data Layer By Python#############################
################################################################################
class Data_Layer_train(caffe.Layer):

    def setup(self, bottom, top):
        if len(top) != 3:
            raise Exception("Need to define tops (data, label, pts)")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom")

        self.mean = 127.5
        self.scale = 1 / 128.0

        params = eval(self.param_str)
        self.mirror = params["mirror"]
        self.illumination = params["illumination"]

        self.im_size = params["im_size"]
        self.crop_size = params["crop_size"]

        self.batch_size = params["batch_size"]
        self.src_file = params['src_file']
        self.basepath = params['img_basepath']

        self.imgTuples = readSrcFile(self.src_file)
        self._cur = 0  # use this to check if we need to restart the list of images

        self.data_aug_type = ["normal"]
        if self.mirror == True:
            self.data_aug_type.append("mirror")
        if self.illumination == True:
            self.data_aug_type.append("illumination")
        if ("mirror" in self.data_aug_type) and ("illumination" in self.data_aug_type):
            self.data_aug_type.append("mirror_illumination")
        top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)
        top[1].reshape(self.batch_size, 1)
        top[2].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, label, pts = self.load_next_image()

            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
            top[2].data[itt, ...] = pts

    def backward(self, top, propagate_down, bottom):
        pass

    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        if self._cur == len(self.imgTuples):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgTuples)
        img_path, label, pts = self.imgTuples[self._cur]
        self._cur += 1
        # bgr
        image = cv2.imread(os.path.join(self.basepath, img_path))
        h = image.shape[0]
        w = image.shape[1]
        if h != w:
            raise Exception("image height not equal width")
        if h != self.im_size:
            raise Exception("image height not equal the prototxt input size")
        image = self.data_augment(image)

        # normalization
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image -= self.mean
        image *= self.scale
        #print os.path.join(self.basepath, img_path), label, pts
        
        if pts != -1:
            pts = pts / 70.0

        return image, label, pts

    # mirror, illumination, mirror+illumination
    def data_augment(self, image):
        # crop
        image = random_crop(image, self.im_size, self.crop_size, flag=1)

        # choose a type of data augment
        idx = random.randint(0, len(self.data_aug_type) - 1)

        if self.data_aug_type[idx] == 'mirror':
            image = mirror(image)
        elif self.data_aug_type[idx] == 'illumination':
            image = illumination(image)
        elif self.data_aug_type[idx] == 'mirror_illumination':
            image = illumination(image)
            image = mirror(image)
        else:
            image = image
        return image

    

################################################################################
#########################Validation Data Layer By Python########################
################################################################################

class Data_Layer_validation(caffe.Layer):

    def setup(self, bottom, top):
        if len(top) != 3:
            raise Exception("Need to define tops (data, label, pts)")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom")

        self.mean = 127.5
        self.scale = 1 / 128.0

        params = eval(self.param_str)

        self.mid_crop = params["mid_crop"]
        self.use_ensemble = params["use_ensemble"]  # average of normal, mirror, illumination...

        self.im_size = params["im_size"]
        self.crop_size = params["crop_size"]

        self.batch_size = params["batch_size"]
        self.src_file = params['src_file']
        self.basepath = params['img_basepath']

        self.imgTuples = readSrcFile(self.src_file)
        self._cur = 0  # use this to check if we need to restart the list of images

        top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)
        top[1].reshape(self.batch_size, 1)
        top[2].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, label, pts = self.load_next_image()
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
            top[2].data[itt, ...] = pts

    def backward(self, top, propagate_down, bottom):
        pass

    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        if self._cur == len(self.imgTuples):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgTuples)
        img_path, label, pts = self.imgTuples[self._cur]
        self._cur += 1
        # bgr
        image = cv2.imread(os.path.join(self.basepath, img_path))
        h = image.shape[0]
        w = image.shape[1]
        if h != w:
            raise Exception("image height not equal width")
        if h != self.im_size:
            raise Exception("image height not equal the prototxt input size")

        # use mid_crop 
        if self.mid_crop == True:
            image = random_crop(image, self.im_size, self.crop_size, flag=0)
        else:
            raise Exception("mid_crop must be True in Validation or Test")

        # normalization
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        image -= self.mean
        image *= self.scale
        
        pts = pts / 70.0

        return image, label, pts

    # ensmeble
    def ensmble(self, image):
        pass
   


################################################################################
#############################Filter Layer By Python###########################
################################################################################

class filter_Layer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs(bottom)")
        if len(top) != 2:
            raise Exception("Need 2 Outputs(top)")

    def reshape(self, bottom, top):
        pts = bottom[1].data[:, 0]
        self.valid_index = np.where(pts != -1)[0]
        self.count = len(self.valid_index)
        if (self.count > 0):
            top[0].reshape(self.count, 1)
            top[1].reshape(self.count, 1)
        else:
            top[0].reshape(1, 1)
            top[1].reshape(1, 1)

    def forward(self, bottom, top):
        if (self.count > 0):
            top[0].data[0:self.count] = bottom[0].data[self.valid_index]
            top[1].data[0:self.count] = bottom[1].data[self.valid_index]
        else:
            top[0].data[...][...] = 0
            top[1].data[...][...] = 0

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0] and self.count != 0:
            bottom[0].diff[...] = 0
            bottom[0].diff[self.valid_index] = top[0].diff[...]
        if propagate_down[1] and self.count != 0:
            bottom[1].diff[...] = 0
            bottom[1].diff[self.valid_index] = top[1].diff[...]
        if self.count == 0:
            bottom[0].diff[...] = 0
            bottom[1].diff[...] = 0




