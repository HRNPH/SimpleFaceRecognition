import random
from typing import Optional, Tuple
from litestar import Response
import numpy as np
from libs.algorithm.facedet.box_utils import predict
from libs.algorithm.pipeline import ImagePreprocessor
from libs.algorithm.pipeline import FaceRecognition
from models.schemas import (
    FaceRecognitionRequest,
    DefaultResponse,
    UserCreationRequest,
)
from db.prisma import db
import base64
import cv2 as cv
from prisma import Base64


class FaceRecognitionController:
    def __init__(self):
        """
        Controller for handling face recognition requests.
        Initializes face detection, face recognition models, and the face database.
        """
        self.recognition_model = FaceRecognition()

    def process_face_recognition(
        self, request: FaceRecognitionRequest
    ) -> DefaultResponse:
        """
        Process the face recognition request by detecting and verifying faces in the image.

        Args:
            request (FaceRecognitionRequest): The request containing the image and parameters.

        Returns:
            FaceRecognitionResponse: The response with results.
        """
        # Decode the image from the request
        image = self._decode_image(request.image_base64)

        # Step 1: Detect Faces
        boxes, labels, probs = self.recognition_model._face_detect(image)
        is_face_exists = FaceRecognition.Utility.is_face_exists(
            boxes, allow_multiple=False
        )

        if not is_face_exists:
            return DefaultResponse(
                message="No face detected in the image",
                success=False,
            )

        # Step 2: Crop the First Detected Face
        cropped_face = FaceRecognition.Utility.face_crop(image, boxes)

        # Step 3: Generate Face Embedding
        face_embedding = self.recognition_model._face_net(cropped_face)[0]

        # Step 4: Verify Against Database
        # source_face = self.database.get_faces(request.rfid)
        source_face = True  # Mock
        if source_face is None:
            return DefaultResponse(
                message="Identity not found in the database",
                success=False,
            )

        # is_same_face, similarity, threshold = FaceRecognition.Utility.is_the_same_face(
        #     source_face, face_embedding, threshold=request.threshold
        # )

        is_same_face, similarity = (
            random.choice([True, False]),
            random.uniform(0, 1),
        )  # Mock

        # Step 5: Generate Response
        if is_same_face:
            return DefaultResponse(
                message="Face verified successfully",
                success=True,
                similarity=similarity,
            )
        else:
            return DefaultResponse(
                message="Face does not match",
                success=False,
                similarity=similarity,
            )

    async def add_face_to_database(self, request: UserCreationRequest) -> Response:
        """
        Add a face to the face database.

        Args:
            request (FaceRecognitionRequest): The request containing the image and parameters.

        Returns:
            FaceRecognitionResponse: The response with results.
        """
        # Decode the image from the request
        image = self._decode_image(request.image_base64)

        # Step 1: Detect Faces
        boxes, labels, probs = self.recognition_model._face_detect(image)
        is_face_exists = FaceRecognition.Utility.is_face_exists(
            boxes, allow_multiple=False
        )

        if not is_face_exists:
            return DefaultResponse(
                message="No face detected in the image",
                success=False,
            )

        # Step 2: Crop the First Detected Face
        cropped_face = FaceRecognition.Utility.face_crop(image, boxes)
        # Step 3: Generate Face Embedding
        face_embedding: np.ndarray = self.recognition_model._face_net(cropped_face)[0]
        # Step 4: Add to Database
        embedding_bytes = face_embedding.tobytes()

        async with db:
            await db.user.create(
                data={
                    "rfid": request.rfid,
                    "faceDatabases": {
                        "create": {"vector": Base64.encode(embedding_bytes)}
                    },
                }
            )
            return DefaultResponse(
                message="Face added to the database",
                success=True,
            )

    @staticmethod
    def _decode_image(image_data_base64: str) -> np.ndarray:
        """
        Decode the image from the request payload base64 string into an OpenCV-compatible array.

        Args:
            image_data (str): The base64 encoded image data.

        Returns:
            np.ndarray: The decoded image as a numpy array.
        """
        # Decode the Base64 string
        image_data = base64.b64decode(image_data_base64, validate=True)
        # Convert the byte data into a numpy array
        np_array = np.frombuffer(image_data, np.uint8)
        return cv.imdecode(np_array, cv.IMREAD_COLOR)  # BGR format
