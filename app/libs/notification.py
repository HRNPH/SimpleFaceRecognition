import os
from typing import Union
import requests
import base64
from app.libs.logutils import logger
from io import BytesIO


class LineNotify:
    def __init__(self, token: str = os.environ["LINE_NOTIFY_PERSONAL_ACCESS_TOKEN"]):
        self.url = "https://notify-api.line.me/api/notify"
        self.token = token
        self.headers = {
            "content-type": "application/x-www-form-urlencoded",
            "Authorization": f"Bearer {self.token}",
        }

    def send(
        self,
        message: str,
    ):
        r = requests.post(
            self.url,
            headers=self.headers,
            data={
                "message": f"{message}",
            },
        )
        logger.info(f"Line Notify Response: {r.text}")
        return r.text

    def template(
        self,
        name: str,
        success: bool,
        timestamp: str,
        image_url: Union[str, None] = None,
    ) -> str:
        return f"\n{name} Authentication {'Success' if success else 'Failed'} at {timestamp}\nView more information at https://embed-regis.hrnph.dev{f'\nView Image: {image_url}' if image_url else ''}"

    def _test_template(self):
        msg = self.template("test", True, "2021-10-10 10:10:10")
        self.send(msg)


if __name__ == "__main__":
    client = LineNotify()
    client._test_template()
