import base64
import random
import requests


class APIClient:
    def __init__(self, api_url, api_key):
        """
        Initialize the API client with the base URL and API key.
        """
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "accept": "application/json",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def encode_image_to_base64(image_path):
        """
        Encode an image from the given file path to a base64 string.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Error encoding image: {e}")

    def create_entry(self, rfid, name, image_path, url_path: str = "create"):
        """
        Send a POST request to create an entry with the provided details.
        """

        # Encode the image to base64
        image_base64 = self.encode_image_to_base64(image_path)

        # Prepare the payload
        payload = {"rfid": rfid, "name": name, "image_base64": image_base64}

        # Send POST request
        response = requests.post(
            f"{self.api_url}{url_path}", json=payload, headers=self.headers
        )

        # Return the response as JSON
        return response.json()

    def create_inference(self, rfid, image_path, url_path: str = "inference"):
        """
        Send a POST request to create an entry with the provided details.
        """
        # Encode the image to base64
        image_base64 = self.encode_image_to_base64(image_path)

        # Prepare the payload
        payload = {
            "rfid": rfid,
            "image_base64": image_base64,
            "temperature": random.uniform(32.0, 37.5),
        }

        # Send POST request
        response = requests.post(
            f"{self.api_url}{url_path}", json=payload, headers=self.headers
        )

        # Return the response as JSON
        return response.json()


# Example usage
if __name__ == "__main__":
    api_url = "http://localhost:8000/api/v1/"
    api_key = "kuy"
    # image_path = "local/experiment/faces/01aa8ed9ba72fb48d3d17db18a8c4e13/focus.png"
    image_path = "local/experiment/faces/guide/1.png"
    rfid = "test"
    name = "focus"

    client = APIClient(api_url, api_key)
    # response = client.create_entry(rfid, name, image_path)
    response = client.create_inference(rfid, image_path)
    print(response)
