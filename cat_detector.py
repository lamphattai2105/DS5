import cv2
import numpy as np
from utils import decode_netout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from losses import YoloLoss
from callbacks import MapEvaluation
from data_generator import BatchGenerator, parse_annotation_xml
from tensorflow.keras.optimizers import Adam
 

class CatDetector:
    def __init__(self):
        self.model = None
        self.anchors = [1.0, 1.0]
        self.labels = ['cat']
        self.num_classes = len(self.labels)
        self.batch_size = 16

    def build_model(self):
        
        num_anchors = len(self.anchors)
        
        backend = InceptionV3(
            include_top = False,
            input_shape = [500, 500, 3],
            weights = 'pretrained/inception_backend.h5'
        )
        
        conv_layer_1 = Conv2D(filters = num_anchors * 6, kernel_size = [1, 1])(backend.output)
        
        self.model = Model(backend.input, conv_layer_1)
        self.model.summary()
        
        loss = YoloLoss(
            self.anchors,
            [14, 14],
            self.batch_size,
            lambda_obj = 5.0 
        )
        
        optimizer = Adam(learning_rate = 1e-4)
        self.model.compile(loss = loss, optimizer = optimizer)
        
        list_train_images, _ = parse_annotation_xml(
                        'datasets/voc/train/anns',
                        'datasets/voc/train/images'
        )
        
        list_valid_images, _ = parse_annotation_xml(
                        'datasets/voc/valid/anns',
                        'datasets/voc/valid/images'
        )
        
        generator_config = {
             'IMAGE_H' : 500,
             'IMAGE_W' : 500,
             'IMAGE_C' : 3,
             'GRID_H'  : 14,
             'GRID_W'  : 14,
             'BOX'     : num_anchors,
             'LABELS'  : self.labels,
             'CLASS'   : self.num_classes,
             'ANCHORS' : self.anchors,
             'BATCH_SIZE' : self.batch_size
        }
        
        valid_generator = BatchGenerator(
            list_valid_images, generator_config, preprocess_input = preprocess_input
        )
        valid_generator = BatchGenerator(
            list_train_images, generator_config, preprocess_input = preprocess_input
        )
    
        map_evaluation = MapEvaluation(
            self, valid_generator,
            iou_threshold = 0.6,
            save_best = True,
            save_path = 'models/cat.best_map.hdf5'
        )
        
        self.model.fit_generator(
            train_generator,
            steps_per_epoch = len(train_generator) * 0.1,
            epochs = 100,
            validation_data = valid_generator,
            validation_steps = len(valid_generator) * 0.1,
            callbacks[map_evaluation]
        )

    def save_model(self):
        pass  # delete this line and replace yours

    def load_model(self):
        pass  # delete this line and replace yours

    def train(self, **kwargs):
        pass  # delete this line and replace yours

    def predict(self, image):
        """
        Autotest will call this function
        :param image: a PIL Image object
        :return: a list of boxes, each item is a tuple of (x_min, y_min, x_max, y_max)
        """
        return []  # delete this line and replace yours

    def preprocess_input(self, image):
        return image

    def infer(self, image, iou_threshold=0.5, score_threshold=0.5):
        image = cv2.resize(image, (416, 416))
        image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        image = self.preprocess_input(image)
        if len(image.shape) == 3:
            input_image = image[np.newaxis, :]
        else:
            input_image = image[np.newaxis, ..., np.newaxis]

        netout = self.model.predict(input_image)[0]

        boxes = decode_netout(netout, self.anchors, self.num_classes, score_threshold, iou_threshold)

        return boxes
