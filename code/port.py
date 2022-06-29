import argparse
import img_to_text as i2t
import seq2seq_pytorch as s2s
import seq2emo as s2e
import random
import string
import os
cuda = True
device = 0
i2t.cuda = cuda

 
import io
from flask import Flask, render_template, request, send_file
from flask_uploads import UploadSet, configure_uploads, IMAGES




app = Flask(__name__)
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = './static/imgs'
configure_uploads(app, photos)
r = i2t.setup_test()
global files
files = {}

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global files
    if request.method == 'POST' and 'photo' in request.files:
        filename_raw = request.files['photo'].filename
        if len(files) > 100:
            files = {}
        fn, ext = os.path.splitext(filename_raw)
        
        #result = i2t.test(r, filename_raw)
        #text = result['text']
        #pad = result['PAD']
        filename = ''.join([random.choice(string.hexdigits) for v in range(20)]) + ext 
        files[filename] = request.files['photo'].read()
        test_res = i2t.test(r, test_images=[request.files['photo']])
        #filename = photos.save(request.files['photo'])
        photos.save(request.files['photo'])
        print(request.files['photo'])
        
        img_path = '/uploads/' + filename 
        return '<html><head></head><body>' + ("<img width=400 src='%s'/><br>" % img_path) + '<br>'.join(test_res['text']) +'</body></html>'
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def get_image(filename):
    #filename ='http://127.0.0.1:5000/uploads/' + filename 
    f = files[filename]
    del files[filename]
    return send_file(io.BytesIO(f), mimetype='image/jpeg')






if __name__ == '__main__':
    app.run(host = "0.0.0.0")
