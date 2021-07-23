import tensorflow as tf
import os
import sys
import subprocess as sp

'''
Returns total memory capacity of all gpus available (list)
'''
def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
  COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
  memory_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_values = [int(x.split()[0]) for i, x in enumerate(memory_info)]
  return memory_values

def configure_gpu(gpu_no, memory_fraction, gpu_memories):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
  memory_limit = gpu_memories[int(gpu_no)] * memory_fraction
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(gpus[0],\
     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
  logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  return gpus, logical_gpus


gpu_no = str(sys.argv[1])
memory_fraction = float(sys.argv[2])
gpu_memories = get_gpu_memory()
# memory_limit = gpu_memories[int(gpu_no)] * memory_fraction
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],\
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
# logical_gpus = tf.config.experimental.list_logical_devices('GPU')
gpus, logical_gpus = configure_gpu(gpu_no, memory_fraction, gpu_memories)
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")





# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],\
#      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
# logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

a = tf.constant(1)
b = tf.constant(2)
while True:
    c = a + b
    print(c)

# python GpuTest.py 4 0.5