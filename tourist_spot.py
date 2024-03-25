from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
# 許可される拡張子のセット
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# クラス名のマッピング
class_names = {
    0: '厳島神社',
    1: '青い池',
    2: '金閣寺',
    3: '富士山',
    4: '東京タワー',
}

def allowed_file(filename):
    """ファイルの拡張子が許可されているかチェックする"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# モデルの読み込み（ローカルパスを指定）
model = load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="ファイルがありません。")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="ファイルが選択されていません。")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join('uploads', filename)  # 一時ファイルの保存先を指定
            file.save(img_path)

            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_class_name = class_names[predicted_class[0]]

            os.remove(img_path)  # 画像ファイルの一時ファイルを削除

            return render_template('index.html', prediction=f'これは{predicted_class_name}です。')
        else:
            return render_template('index.html', prediction="許可されていないファイル形式です。")

    return render_template('index.html')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)