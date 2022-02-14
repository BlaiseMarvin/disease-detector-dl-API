import requests 
import numpy as np 
resp=requests.post("https://zendimak-passion-image.herokuapp.com/predict",files={'file':open('fruit.jpg','rb')})
# print('cid shape: ',np.array(resp.json()['cid']).shape)
# print('score shape: ',np.array(resp.json()['score']).shape)
# print('bbox shape: ',np.array(resp.json()['bbox']).shape)
# print('image shape: ',np.array(resp.json()['image']).shape)
# print(np.array(resp.json()['image']).shape)
print(resp.json())
