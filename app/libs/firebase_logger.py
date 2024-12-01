import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from app.models.schemas import FirebaseLog
from app.libs.logutils import logger
import os
import base64

load_dotenv()


class EnvironmentConfig(BaseModel):
    type: str = Field(..., description="The type of the configuration")
    project_id: str = Field(..., description="The project ID")
    private_key_id: str = Field(..., description="The private key ID")
    private_key: str = Field(..., description="The private key")
    client_email: str = Field(..., description="The client email address")
    client_id: str = Field(..., description="The client ID")
    auth_uri: str = Field(..., description="The authentication URI")
    token_uri: str = Field(..., description="The token URI")
    auth_provider_x509_cert_url: str = Field(
        ..., description="The X509 certificate URL for the auth provider"
    )
    client_x509_cert_url: str = Field(
        ..., description="The X509 certificate URL for the client"
    )
    universe_domain: str = Field(..., description="The universe domain")


# Load the environment variables into the Pydantic model
def load_env_to_dict():
    env_config = EnvironmentConfig(
        type=os.getenv("TYPE"),
        project_id=os.getenv("PROJECT_ID"),
        private_key_id=os.getenv("PRIVATE_KEY_ID"),
        private_key=base64.b64decode(os.getenv("PRIVATE_KEY")).decode("utf-8"),
        client_email=os.getenv("CLIENT_EMAIL"),
        client_id=os.getenv("CLIENT_ID"),
        auth_uri=os.getenv("AUTH_URI"),
        token_uri=os.getenv("TOKEN_URI"),
        auth_provider_x509_cert_url=os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
        client_x509_cert_url=os.getenv("CLIENT_X509_CERT_URL"),
        universe_domain=os.getenv("UNIVERSE_DOMAIN"),
    )
    return env_config.model_dump()


class FirebaseAuthenticationLogs:
    def __init__(
        self, log_path: str = "embed/v1", cred_dict: dict = load_env_to_dict()
    ):
        # load from .env as environment variables, Inject as file
        self.__cred_dict = cred_dict
        self.__cert = credentials.Certificate(self.__cred_dict)
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
