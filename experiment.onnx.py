import os
from typing import Dict, List, Tuple, Union
import onnxruntime as ort
import numpy as np
import cv2 as cv
from tqdm import tqdm

from app.utils.algorithm.facedet.box_utils import predict


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


class FaceDatabase:
    def __init__(self):
        self.faces: Dict[str, np.ndarray[np.ndarray]] = {}

    def add_face(self, key: str, face: np.ndarray):
        self.faces[key] = face if key not in self.faces else self.faces[key]

    def remove_face(self, key: str):
        for i, face in enumerate(self.faces):
            if key in face:
                self.faces.pop(i)
                break

    def get_faces(self, key: str) -> np.ndarray[np.ndarray]:
        """Get the faces from the database given the key"""
        if key is not None:
            return self.faces[key] if key in self.faces else None
        return self.faces

    def clear(self):
        self.faces = []

    def mock_init(self, from_path: str = "local/experiment/faces"):
        print("Initialize the database")
        model = FaceRecognition()
        # check if empty
        if len(self.faces) > 0:
            print("Database is not empty")
            return

        for folder in (pbar_folder := tqdm(os.listdir(from_path))):
            pbar_folder.set_description(f"Processing Folder: {folder}")
            for file in (
                pbar_file := tqdm(os.listdir(os.path.join(from_path, folder)))
            ):
                pbar_file.set_description(f"- Files: {file}")
                if file.endswith(".jpg") or file.endswith(".png"):
                    image = cv.imread(os.path.join(from_path, folder, file))
                    is_face_exists = model.face_exists(image)
                    if not is_face_exists:
                        raise Exception(f"No face found in the image {file}")
                    boxs, label, prob = model._face_detect(image)
                    face_crop = model.Utility.face_crop(image, boxs)
                    face_vector = model._face_net(face_crop)[0]
                    self.add_face("test", face_vector)

        if len(self.faces) == 0:
            raise Exception("No faces found in the database")


class Debug:
    def __init__(self, local_database: str = "local/experiment/faces"):
        self.database = FaceDatabase()
        self.database.mock_init(local_database)

    def debug(self):
        model = FaceRecognition()
        cam = cv.VideoCapture(0)
        print("Camera ready!")
        while True:
            ready, frame = cam.read()

            if ready and frame is not None:
                # --- 1. Face Detection
                boxes, labels, probs = model._face_detect(frame)
                is_face_exists = FaceRecognition.Utility.is_face_exists(
                    boxes, allow_multiple=False
                )
                drawned = FaceRecognition.Utility.draw_boxes(boxes, frame)

                if is_face_exists:  # SKIP FACE RECOGNITION IF FACE NOT EXISTS
                    # --- 2. Face Recognition
                    # Crop the Face
                    cropped_face = FaceRecognition.Utility.face_crop(frame, boxes)
                    # print(boxes)
                    cv.imshow("Face", cropped_face)
                    predicted = model._face_net(cropped_face)

                    # --- 3. Verify the face
                    source_face = self.database.get_faces("test")
                    is_same_face, score, thresh = (
                        FaceRecognition.Utility.is_the_same_face(
                            source_face, predicted[0], threshold=0.95
                        )
                    )
                    # ---- 3.1 Shown Keyword
                    text, color = (
                        ("Authenticated", (0, 255, 0))
                        if is_same_face
                        else ("Not Authenticated", (0, 0, 255))
                    )

                    # Show the frame
                    cv.putText(
                        drawned, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2
                    )
                    cv.putText(
                        drawned,
                        f"Similarity: {score:.2f}",
                        (10, 60),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                    )
                    cv.imshow("Result", drawned)
                else:
                    cv.putText(
                        drawned,
                        "Face not exists/Multiple faces Detected",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv.imshow("Result", drawned)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    cv.destroyAllWindows()
                    break

            else:
                print("Camera not ready")
                break


def main():
    runner = Debug()
    runner.debug()


if __name__ == "__main__":
    main()
