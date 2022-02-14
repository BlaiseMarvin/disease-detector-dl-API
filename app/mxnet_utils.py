import io 
from PIL import Image 
from mxnet import gluon 
import mxnet as mx 
import warnings
import numpy as np 
import cv2
import gluoncv as gcv 
from gluoncv import utils

CLASSES=['fruit_brownspot', 'fruit_healthy', 'fruit_woodiness']

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    deserialized_net = gluon.nn.SymbolBlock.imports("app/passionfruits-symbol.json", ['data'],"app/passionfruits-0000.params")

# def preprocessing(input_image,height=512,width=512):
# 	preprocessed_image=np.copy(input_image)
# 	preprocessed_image=cv2.resize(preprocessed_image,(width,height))
# 	preprocessed_image=cv2.cvtColor(preprocessed_image,cv2.COLOR_BGR2RGB)
# 	preprocessed_image=preprocessed_image.transpose((2,0,1))
# 	preprocessed_image=preprocessed_image.reshape(1,3,height,width)
# 	return mx.nd.array(preprocessed_image)

def transform_image(image_bytes):
	image=Image.open(io.BytesIO(image_bytes))
	# return preprocessing(image)
	image=mx.nd.array(image)
	return image 

def get_prediction(image):
	x,image=gcv.data.transforms.presets.ssd.transform_test(image,short=512)
	cid,score,bbox=deserialized_net(x)
	imageResult=utils.viz.bbox.cv_plot_bbox(image,bbox[0],scores=score[0],labels=cid[0],class_names=CLASSES,linewidth=2)
	return imageResult.tolist()



# def get_prediction(image_tensor):
# 	cid,score,bbox=deserialized_net(image_tensor)
# 	return {'cid':cid,'score':score,'bbox':bbox}


