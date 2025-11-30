import os
import glob
from flask import Flask, render_template, request, redirect, url_for
from stitcher import PanoramaStitcher

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

stitcher = PanoramaStitcher()

def clear_uploads():
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
        os.remove(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        clear_uploads()
        
        uploaded_files = request.files.getlist('images')
        if not uploaded_files or uploaded_files[0].filename == '':
            return redirect(request.url)

        filepaths = []
        filenames = [] # Store simple filenames for display
        
        for file in uploaded_files:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            filepaths.append(path)
            filenames.append(filename)

        if len(filepaths) < 2:
            return "Please upload at least 2 images."

        result_image = stitcher.stitch(filepaths)

        if result_image is None:
            return "Stitching failed."

        result_filename = 'panorama_result.jpg'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        # SAVE USING PILLOW (Manual stitcher helper)
        stitcher.save_image(result_image, result_path)

        return render_template('index.html', result=result_filename, input_images=filenames)

    return render_template('index.html', result=None, input_images=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)