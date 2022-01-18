import os
import numpy as np
import time
import cv2

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

from configuration import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def load_model(path_to_model):
    start_time = time.time()
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(path_to_model)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time: ' + str(elapsed_time) + 's')
    print("MODEL LOADED.")
    
    return detect_fn

def main(path_to_model, path_to_labelmap):
    detect_fn = load_model(path_to_model)
    vision = cv2.VideoCapture(-1)

    while vision.isOpened():
        ret, frame = vision.read()

        elapsed = []
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(image_np, 0)
        start_time = time.time()
        detections = detect_fn(input_tensor)
        end_time = time.time()
        elapsed.append(end_time - start_time)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.int32),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.30,
                agnostic_mode=False)
        cv2.imshow("Object Detection", frame)

        mean_elapsed = sum(elapsed) / float(len(elapsed))
        # print('Elapsed time: ' + str(mean_elapsed) + ' second per image')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vision.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(MODEL_DIR, LABEL_DIR)
