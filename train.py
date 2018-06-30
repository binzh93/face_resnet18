import sys
sys.path.append('/workspace/mnt/group/face-det/zhubin/caffe/python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('/workspace/mnt/group/face-det/zhubin/face_resnet18/solver.prototxt')
solver.net.copy_from('/workspace/mnt/group/face-det/zhubin/train_file/resnet18-priv.caffemodel')
solver.solve()
