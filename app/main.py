from flask import Flask,request,jsonify

from app.mxnet_utils import transform_image,get_prediction

app=Flask(__name__)

ALLOWED_EXTENSIONS=['png', 'jpg', 'jpeg']

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict',methods=['POST'])
def predict():
	if request.method=='POST':
		file=request.files.get('file')
		if file is None or file.filename=="":
			return jsonify({'error':'no file'})
		if not allowed_file(file.filename):
			return jsonify({'error':'format not supported'})

		try:
			img_bytes=file.read()
			tensor=transform_image(img_bytes)
			result=get_prediction(tensor)
			return jsonify({'results':result})
			# prediction=get_prediction(tensor)

			
			# cid=prediction['cid'].asnumpy().tolist()
			# score=prediction['score'].asnumpy().tolist()
			# bbox=prediction['bbox'].asnumpy().tolist()
			
			
			  
			# return jsonify({'cid':cid,'score':score,'bbox':bbox})
		except:
			return jsonify({'error':'error during inference'})