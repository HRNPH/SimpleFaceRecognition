from typing import Dict, List, Tuple, Union
import onnxruntime as ort
import numpy as np
import cv2 as cv
from app.libs.algorithm.facedet.box_utils import predict

ort.set_default_logger_severity(3)  # Disable onnx logger


class ImagePreprocessor:
    @staticmethod
    def __resize(img: np.ndarray, dim: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """Convert to RGB, Resize"""
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, dim)
        return img

    @staticmethod
    def face_det(img: np.ndarray, dim: Tuple[int, int] = (640, 480)) -> np.ndarray:
        image = ImagePreprocessor.__resize(img, dim)
        image_mean = np.array([127, 127, 127])  # Mean of the image
        image = (image - image_mean) / 128  # Normalize
        image = np.transpose(image, [2, 0, 1])  # Change data layout from HWC to CHW
        image = np.expand_dims(image, axis=0)
        return image.astype(
            np.float32
        )  # MUST BE FLOAT32, did this at this process to reduce computation resources (lower bit)

    @staticmethod
    def face_net(img: np.ndarray, dim: Tuple[int, int] = (160, 160)) -> np.ndarray:
        image = ImagePreprocessor.__resize(img, dim)
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)


class FaceRecognition:
    def __init__(
        self,
        face_detect_model: str = "app/weights/Ultra-lightweight-RFB-640.onnx",
        face_net_model: str = "app/weights/faceNet.onnx",
    ):
        self.face_det = ort.InferenceSession(face_detect_model)
        self.face_net = ort.InferenceSession(face_net_model)

    def _face_detect(
        self, image: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray[np.ndarray]:
        """Low level face detection function, Offers more control"""
        x = ImagePreprocessor.face_det(image)
        input_name = self.face_det.get_inputs()[0].name
        confidences, boxes = self.face_det.run(None, {input_name: x})
        boxes, labels, probs = predict(
            image.shape[1], image.shape[0], confidences, boxes, threshold
        )  # Predict as orginal image size
        return boxes, labels, probs

    def _face_net(self, image: np.ndarray) -> np.ndarray[np.ndarray]:
        """Low level face landmark function, Offers more control"""
        x = ImagePreprocessor.face_net(image)
        predicted = self.face_net.run(None, {"image_input": x})
        return predicted

    # Wrappers, for easy use
    def face_exists(
        self, image: np.ndarray, threshold: float = 0.5, allow_multiple: bool = False
    ) -> bool:
        """Wrapper for face detection, checks if face exists in the given image"""
        boxes, labels, probs = self._face_detect(image, threshold)
        is_face_exists = FaceRecognition.Utility.is_face_exists(
            boxes, allow_multiple=allow_multiple
        )
        return is_face_exists

    class Utility:
        @staticmethod
        def scale(box):
            """scale current rectangle to box"""
            width = box[2] - box[0]
            height = box[3] - box[1]
            maximum = max(width, height)
            dx = int((maximum - width) / 2)
            dy = int((maximum - height) / 2)

            bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
            return bboxes

        @staticmethod
        def draw_boxes(
            boxes: np.ndarray,
            image: np.ndarray,
            color: Tuple[int, int, int] = (0, 255, 0),
        ) -> np.ndarray:
            """Draw boxes on the given image
            Args:
                boxes: List of boxes to draw (can have multiple boxes)
                image: Image to draw the boxes on (Only one image)
                color: Color of the boxes (BGR Scale)
            """
            for i in range(boxes.shape[0]):
                box = FaceRecognition.Utility.scale(boxes[i, :])
                image = cv.rectangle(
                    image, (box[0], box[1]), (box[2], box[3]), color, 4
                )
            return image

        @staticmethod
        def face_crop(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
            """Crop the face from the given image using the given boxesm Handling out of bound issue

            Args:
                image (np.ndarray): image to crop the face from
                boxes (np.ndarray): List of boxes to crop the face from

            Returns:
                np.ndarray: an image of the cropped face
            """
            box = boxes[0]  # Get the first box form the list
            box = FaceRecognition.Utility.scale(box)
            # Handling out of bound issue, Max min
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(image.shape[1], box[2])
            box[3] = min(image.shape[0], box[3])
            return image[box[1] : box[3], box[0] : box[2]]  # Crop the image

        @staticmethod
        def is_face_exists(boxes: np.ndarray, allow_multiple: bool = False) -> bool:
            """Check if face exists in the given boxes
            Args:
                boxes: List of boxes to check
                threshold: Minimum threshold to consider as a face
                allow_multiple: Allow multiple faces
            """
            return len(boxes) > 0 if allow_multiple else len(boxes) == 1

        @staticmethod
        def is_the_same_face(
            source: Union[np.ndarray, np.ndarray[np.ndarray]],
            target: np.ndarray,
            threshold: float = 0.95,
        ) -> bool:
            """Check if faces are the same, Using cosine similarity after flattening the array
            Args:
                source: Source vector to compare with the target, if multiple faces, would compute as mean
                target: Second face
                threshold: Minimum threshold to consider as the same face
            """
            if not isinstance(source, np.ndarray):
                raise ValueError("Invalid source type")

            def cosine_similarity(source: np.ndarray, target: np.ndarray) -> float:
                source, target = (source.flatten(), target.flatten())
                return np.dot(source, target) / (
                    np.linalg.norm(source) * np.linalg.norm(target)
                )  # Formula: A.B / |A|.|B|

            if source.ndim == 2:
                similarity = np.array(
                    [cosine_similarity(face, target) for face in source]
                ).mean()
            elif source.ndim == 1:
                similarity = cosine_similarity(source, target)
            else:
                raise ValueError("Invalid source shape, Expected 1D or 2D array")

            print(similarity)
            return similarity > threshold, similarity, threshold
