import sys
import os
#sys.path.append('/workspace/mnt/group/face-det/zhubin/caffe/python')
curr_path = os.path.abspath(".")
sys.path.append(curr_path)

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('/workspace/mnt/group/face-det/zhubin/face_resnet18/solver.prototxt')
solver.net.copy_from('/workspace/mnt/group/face-det/zhubin/train_file/resnet-18.caffemodel')
solver.solve()
