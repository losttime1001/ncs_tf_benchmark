#-*- coding:utf-8 -*-
import mvnc.mvncapi as ncs
import numpy as np
import cv2
import sys
import time
 
def run(input_image):

    ncs_names = ncs.EnumerateDevices()
    if (len(ncs_names) < 1):
        print("Error - no NCS devices detected.")
        quit()

    dev = ncs.Device(ncs_names[0])
 
    dev.OpenDevice()
 
    with open('graph', 'rb') as f:
        graph = dev.AllocateGraph(f.read())

    if (graph.LoadTensor(input_image.astype(np.float16), 'user object')):
        output, userobj = graph.GetResult()
        #print(userobj)
        #print 'Predict label:',np.argmax(output)
        #print(output)

    for i in range(0,1):
        start_time = time.time()
        if (graph.LoadTensor(input_image.astype(np.float16), 'user object')):
            output, userobj = graph.GetResult()
            print 'Predict label:',np.argmax(output)
            print 'Predict time:', time.time() - start_time, 's'

    graph.DeallocateGraph()
    dev.CloseDevice()
 
if __name__=="__main__":
    args = sys.argv
    if len(args) != 2:
        print('Usage: python %s filename'%(args[0]))
        quit()
    
    image = cv2.imread(args[1])
    image = cv2.resize(image, (299, 299))
    image = np.array(image)/255.0
    run(image)
