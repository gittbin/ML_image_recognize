from flask import Flask, request, jsonify
from flask_cors import CORS  # Thêm import CORS
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import base64

app = Flask(__name__)

# Enable CORS
CORS(app)  # Để mở rộng CORS cho toàn bộ ứng dụng Flask

# Load mô hình MNIST đã huấn luyện


@app.route("/predict", methods=["POST"])
def predict():
    model = tf.keras.models.load_model("model_aug.h5")
    try:
        # Nhận ảnh dưới dạng base64 từ frontend
        data = request.get_json()
        image_data = data["image"]

        print(str(image_data))

        # Xử lý base64 thành ảnh
        img_data = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(img_data))
        # .convert(
        #     "L"
        # )  # Chuyển ảnh thành ảnh xám (grayscale)

        # Thêm background trắng
        if image.mode == "RGBA":
            white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            white_background.paste(image, (0, 0), image)
            image = white_background.convert("L")
        else:
            image = image.convert("L")

        # Đổi nền trắng thành đen, nét đen thành trắng
        image = 255 - np.array(image)
        image = Image.fromarray(image)

        # Resize ảnh về kích thước 28x28
        image = image.resize((28, 28))

        # Chuyển ảnh thành array NumPy và chuẩn hóa
        img_array = np.array(image) / 255.0  # Chuẩn hóa giá trị pixel

        img_array = img_array.reshape(
            1, 28 * 28, 1
        )  # Thêm chiều batch và kênh (1, 28, 28, 1)

        # Dự đoán
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)  # Lấy nhãn có xác suất cao nhất
        prediction_probabilities = prediction[0]  # Các xác suất cho các lớp

        print("Prediction Probabilities: ", prediction_probabilities)
        print("Predicted Class: ", predicted_class)

        return jsonify(
            {
                "prediction": str(predicted_class),
                "probabilities": prediction_probabilities.tolist(),
            }
        )

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 400
@app.route("/predict2", methods=["POST"])
def predict2():
    model = tf.keras.models.load_model("/mnist_model.h5")
    try:
        # Nhận ảnh dưới dạng base64 từ frontend
        data = request.get_json()
        image_data = data["image"]

        print(f"Received image data of length: {len(image_data)}")

        # Xử lý base64 thành ảnh
        img_data = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(img_data))

        # Thêm background trắng nếu cần
        if image.mode == "RGBA":
            white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            white_background.paste(image, (0, 0), image)
            image = white_background.convert("L")
        else:
            image = image.convert("L")

        # Đổi nền trắng thành đen
        image = 255 - np.array(image)

        # Resize ảnh về kích thước 28x28
        image = Image.fromarray(image).resize((28, 28))

        # Chuyển ảnh thành array NumPy và chuẩn hóa
        img_array = np.array(image) / 255.0  # Chuẩn hóa giá trị pixel

        # Thêm chiều kênh màu (1) để có shape (28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        # Dự đoán
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))  # Lấy nhãn có xác suất cao nhất
        prediction_probabilities = np.round(prediction[0], 4).tolist()

        print("Prediction Probabilities: ", prediction_probabilities)
        print("Predicted Class: ", predicted_class)

        return jsonify(
            {
                "prediction": str(predicted_class),
                "probabilities": prediction_probabilities,
            }
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True)
