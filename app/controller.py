import random
from typing import Optional, Tuple
from litestar import Response
import numpy as np
from app.libs.algorithm.facedet.box_utils import predict
from app.libs.algorithm.pipeline import ImagePreprocessor
from app.libs.algorithm.pipeline import FaceRecognition
from app.models.schemas import (
    FaceRecognitionRequest,
    DefaultResponse,
    UserCreationRequest,
)
from app.db.prisma import db
import base64
import cv2 as cv
from prisma import Base64
from app.libs.logutils import logger


class FaceRecognitionController:
    def __init__(self):
        """
        Controller for handling face recognition requests.
        Initializes face detection, face recognition models, and the face database.
        """
        self.recognition_model = FaceRecognition()

    async def process_face_recognition(
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
                rfid=request.rfid,
            )

        # Step 2: Crop the First Detected Face
        cropped_face = FaceRecognition.Utility.face_crop(image, boxes)

        # Step 3: Generate Input, Face Embedding
        face_embedding = self.recognition_model._face_net(cropped_face)[0]

        # Step 4: Verify Against Database, Grab Precomputed Face Embedding
        async with db:
            user = await db.user.find_unique(
                where={"rfid": request.rfid},
                include={"faceDatabases": True},
            )
            if (
                user is None
                or user.faceDatabases is None
                or len(user.faceDatabases) == 0
            ):
                return DefaultResponse(
                    message="Identity not found in the database",
                    success=False,
                )

            source_vector = np.frombuffer(
                Base64.decode(user.faceDatabases[0].vector), np.float32
            )

            is_same_face, similarity, threshold = (
                FaceRecognition.Utility.is_the_same_face(source_vector, face_embedding)
            )

            # Step 5: Generate Response
            message, success = (
                ("Face does not match", False)
                if not is_same_face
                else ("Face verified successfully", True)
            )
            return DefaultResponse(
                message=message,
                success=success,
                similarity=similarity,
                threshold=threshold,
                temperature=request.temperature,
                rfid=request.rfid,
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
