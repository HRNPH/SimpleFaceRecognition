import json
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
from dotenv import load_dotenv
from models.schemas import FirebaseLog
from libs.logging import logger
import os

load_dotenv()


class FirebaseAuthenticationLogs:
    def __init__(self, log_path: str = "embed/v1"):
        # load from .env as environment variables, Inject as file
        self.__cred_json = json.loads((os.environ["FIREBASE_CREDENTIALS"]))
        self.__cert = credentials.Certificate(self.__cred_json)
        self._app_name = "FaceRecognitionAPI"
        if len(firebase_admin._apps) == 0:
            logger.info(f"Initializing Firebase App, App Name: {self._app_name}")
            firebase_admin.initialize_app(self.__cert, name=self._app_name)

        self.log_path = log_path
        self.ref = db.reference(
            path=self.log_path,
            url=os.environ["FIREBASE_DATABASE_URL"],
            app=firebase_admin.get_app(self._app_name),
        )

    def log(self, data: FirebaseLog):
        """Log the authentication request to firebase"""
        return self.ref.push(data.to_firebase_dict())
