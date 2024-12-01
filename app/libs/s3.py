import os
import base64
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from app.libs.logutils import logger


class S3Uploader:
    """Handles S3 upload functionality."""

    def __init__(
        self,
        bucket_name: str = os.environ["S3_BUCKET_NAME"],
        region_name: str = "auto",  # R2 uses "auto" for region
        access_key: str = os.environ["S3_ACCESS_KEY_ID"],
        secret_key: str = os.environ["S3_SECRET_ACCESS_KEY"],
        endpoint_url: str = os.environ["S3_ENDPOINT_URL"],
    ):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,  # Custom endpoint for R2
        )

    def upload(self, image_base64: str, file_path: str) -> str:
        """
        Uploads a base64-encoded image to the S3 bucket.

        :param image_base64: The base64-encoded image string.
        :param file_path: The desired S3 object key.
        :return: The public URL of the uploaded file.
        """
        try:
            # Decode the base64 image
            image_binary = base64.b64decode(image_base64)

            # Upload the file to the bucket
            result = self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=image_binary,
                ContentType="image/png",
            )
            logger.debug(f"Upload result: {result}")

            # Generate the public URL
            s3_url = f"{os.environ['S3_ENDPOINT_URL_PUB']}/{file_path}"
            logger.info(f"Uploaded to S3: {s3_url}")
            return s3_url
        except (BotoCoreError, ClientError) as e:
            logger.error(f"S3 Upload Failed: {str(e)}")
            raise Exception(f"S3 upload failed: {str(e)}")

    @staticmethod
    def _get_mime_type(base64_data: str) -> str:
        """
        Extracts the MIME type from a base64-encoded image string.

        :param base64_data: The base64-encoded image string.
        :return: The MIME type of the image.
        """
        header = base64_data.split(",")[0]
        if "data:" in header and ";base64" in header:
            return header.split(":")[1].split(";")[0]
        raise ValueError("Invalid base64 image data format")
