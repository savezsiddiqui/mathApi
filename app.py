import os
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from final import result as result

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def method():

    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    if request.method == 'POST':
        some_json = request.get_json()
        imgstring = some_json[22:]
        imgdata = base64.b64decode(imgstring)
        filename = 'some_image.png'
        path = 'uploads/'
        with open(path + filename, 'wb') as f:
            f.write(imgdata)
        data = result()
        return jsonify({"your digits": data}), 201
    else:
        # return jsonify({"message": "HELLO FUCKTARD"})
        data = result()
        return jsonify({"your digits": data}), 201


if __name__ == "__main__":
    app.run(debug=True)
