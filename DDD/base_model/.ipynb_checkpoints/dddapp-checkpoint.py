import cv2
import flet as ft
import threading
import base64
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import model  # from model.py

class VideoCaptureApp:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.is_running = False
        self.image_control = ft.Image()
        self.model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_320x320/saved_model')
        self.category_index = self.load_labels('mscoco_label_map.pbtxt')

    def load_labels(self, path):
        from object_detection.utils import label_map_util
        return label_map_util.create_category_index_from_labelmap(path, use_display_name=True)

    def start_capture(self):
        self.is_running = True
        self.capture_thread = threading.Thread(target=self.update_frame)
        self.capture_thread.start()

    def stop_capture(self):
        self.is_running = False
        self.capture_thread.join()

    def update_frame(self):
        while self.is_running:
            ret, frame = self.capture.read()
            if ret:
                frame = self.run_inference(frame)
                # Convert the frame to a format Flet can use
                _, buffer = cv2.imencode('.jpg', frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                self.image_control.src_base64 = jpg_as_text
                self.image_control.update()
            time.sleep(0.03)  # Introduce a small delay for smoother video

    def run_inference(self, image):
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)
        detections = self.model(input_tensor)

        bboxes = detections['detection_boxes'][0].numpy()
        class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        for i in range(len(bboxes)):
            if scores[i] >= 0.5:
                bbox = bboxes[i]
                class_id = class_ids[i]
                display_str = self.category_index[class_id]['name']
                ymin, xmin, ymax, xmax = bbox
                im_height, im_width, _ = image.shape
                left, right, top, bottom = (xmin * im_width, xmax * im_width,
                                            ymin * im_height, ymax * im_height)

                cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                cv2.putText(image, display_str, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image

    def build(self):
        return ft.Column([
            self.image_control,
            ft.Row([
                ft.ElevatedButton("Start", on_click=lambda _: self.start_capture()),
                ft.ElevatedButton("Stop", on_click=lambda _: self.stop_capture()),
            ])
        ])

def train_and_test_model():
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        '../Datasets/train_data/',
        target_size=(128, 128),
        batch_size=32,
        shuffle=True,
        subset='training',
        class_mode='binary')

    validation_generator = train_datagen.flow_from_directory(
        '../Datasets/train_data/',
        target_size=(128, 128),
        batch_size=32,
        shuffle=True,
        subset='validation',
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        '../Datasets/train_data/',
        target_size=(128, 128),
        batch_size=32,
        shuffle=True,
        class_mode='binary')

    # Load model
    base_model = model.getModel()

    # Callbacks
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='model_weights.{epoch:02d}-{val_loss:.2f}.weights.h5',
                                           save_weights_only=True,
                                           monitor='val_accuracy'),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs', update_freq='batch', profile_batch=0)
    ]

    # Train the model
    hist = base_model.fit(train_generator,
                          epochs=20,
                          batch_size=train_generator.samples,
                          validation_data=validation_generator,
                          validation_steps=validation_generator.samples,
                          verbose=1,
                          callbacks=my_callbacks)

    # Test the model
    score = base_model.evaluate(test_generator, verbose=1)
    print('Test accuracy:', score[1])

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def main(page: ft.Page):
    app = VideoCaptureApp()
    page.add(app.build())
    # Add buttons to train and test the model
    page.add(
        ft.Row([
            ft.ElevatedButton("Train and Test Model", on_click=lambda _: train_and_test_model())
        ])
    )

ft.app(target=main)
