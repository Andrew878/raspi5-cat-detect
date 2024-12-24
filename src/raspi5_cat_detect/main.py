import time
import os
import signal
import sys
from pathlib import Path
from typing import Tuple

import requests
import boto3
from picamera2 import Picamera2
import moondream as md
from twilio.rest import Client
from PIL import Image
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(),override=True)

# Global flag to indicate we should keep running (until a signal says "stop")
keep_running = True


def handle_termination_signal(signum, frame):
    """Signal handler to gracefully exit on Ctrl+C or kill."""
    global keep_running
    keep_running = False
    print("\nReceived termination signal. Stopping after current iteration...")


class CatDetector:
    def __init__(self, image_dir: str = "./cat_images", model_path: str = "moondream-2b-int8.mf"):
        """Initialize the cat detector with camera, model, Twilio, and S3 config"""
        # 1. Initialize Raspberry Pi Camera
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration()
        self.picam2.configure(config)

        # 2. Set image directory
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # 3. Initialize Moondream model
        self.model = md.vl(model=model_path)

        # 4. Twilio WhatsApp configuration
        self.twilio_from_number = os.getenv('TWILIO_FROM_NUMBER')  # e.g., "whatsapp:+14155238886"
        self.your_number = os.getenv('YOUR_WHATSAPP_NUMBER')       # e.g., "whatsapp:+12345556789"

        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        if not all([self.twilio_account_sid, self.twilio_auth_token]):
            raise ValueError("Missing required environment variables for Twilio (SID, Auth).")

        self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)

        # 5. S3 configuration
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("Missing S3_BUCKET_NAME in environment variables.")

        self.s3_client = boto3.client("s3")  # uses credentials from environment or aws config

    def capture_image(self) -> Path:
        """Capture an image using Raspberry Pi Camera to local disk."""
        self.picam2.start()
        time.sleep(2)  # Allow camera to adjust

        image_path = self.image_dir / f'cat_image_{int(time.time())}.jpg'
        self.picam2.capture_file(str(image_path))
        self.picam2.stop()

        return image_path

    def generate_presigned_put_url(self, object_key: str, expiration=1800) -> str:
        """
        Generate a presigned URL that allows a PUT request (upload) to S3.
        Valid for `expiration` seconds, and includes ContentType='image/jpeg'
        so the object is stored as a JPEG.
        """
        try:
            response = self.s3_client.generate_presigned_url(
                ClientMethod='put_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_key,
                    'ContentType': 'image/jpeg'
                },
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            print(f"Error generating presigned PUT URL: {e}")
            return None

    def generate_presigned_get_url(self, object_key: str, expiration=1800) -> str:
        """
        Generate a presigned URL that allows a GET request (download) from S3.
        Valid for `expiration` seconds.
        """
        try:
            response = self.s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_key
                },
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            print(f"Error generating presigned GET URL: {e}")
            return None

    def upload_file_via_presigned_url(self, presigned_url: str, local_file_path: Path) -> bool:
        """
        Uploads `local_file_path` to S3 using the presigned PUT URL (HTTP PUT).
        Returns True if upload is successful (HTTP 200), False otherwise.
        Must include headers={'Content-Type': 'image/jpeg'} to match our presigned URL.
        """
        try:
            with open(local_file_path, 'rb') as file_data:
                response = requests.put(
                    presigned_url,
                    data=file_data,
                    headers={'Content-Type': 'image/jpeg'}  # Must match 'ContentType' param above
                )
            print(f"Upload status code: {response.status_code}")
            print(f"Upload response text: {response.text}")
            return (response.status_code == 200)
        except Exception as e:
            print(f"Error uploading via presigned URL: {e}")
            return False

    def upload_and_get_s3_url(self, local_file_path: Path) -> str:
        """
        1) Generate a presigned PUT URL.
        2) Upload local file to S3 via that URL.
        3) Generate and return a presigned GET URL for Twilio to access.

        Returns the presigned GET URL or None if something failed.
        """
        object_key = local_file_path.name  # Use the same filename in S3

        put_url = self.generate_presigned_put_url(object_key, expiration=1800)
        if not put_url:
            print("Failed to generate presigned PUT URL.")
            return None

        print(f"Presigned PUT URL (for uploading): {put_url}")

        success = self.upload_file_via_presigned_url(put_url, local_file_path)
        if not success:
            print("Upload via presigned URL failed.")
            return None

        # Now generate a presigned GET URL so Twilio can fetch this image
        get_url = self.generate_presigned_get_url(object_key, expiration=1800)
        print(f"Presigned GET URL (for Twilio): {get_url}")
        return get_url

    def detect_cat(self, image_path: Path) -> Tuple[bool,str]:
        """Example logic: Detect if there's a cat (or 'books') in the image using Moondream."""
        image = Image.open(image_path)
        encoded_image = self.model.encode_image(image)

        # 1) Basic caption
        caption = self.model.caption(encoded_image)["caption"]
        print(f"Image caption: {caption}")

        # 2) Query for 'are there books' (just as a placeholder question)
        answer = self.model.query(
            encoded_image,
            "Are there books in this image? Answer with just 'yes' or 'no'."
        )["answer"]
        print(f"Query answer: {answer}")

        return answer.lower().strip() == "yes", caption

    def send_whatsapp(self, image_get_url: str, caption: str = None) -> bool:
        """
        Send image via WhatsApp using Twilio, referencing the presigned GET URL.
        Twilio must be able to fetch the image at `image_get_url`.
        """
        try:
            message_body = 'Cat detected!'
            if caption:
                message_body += f'\n{caption}'

            message = self.twilio_client.messages.create(
                from_=self.twilio_from_number,
                body=message_body,
                to=self.your_number,
                media_url=[image_get_url]  # Must be http(s) accessible by Twilio
            )
            print(f"WhatsApp message sent successfully! SID: {message.sid}")
            return True
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
            return False

    def run(self, interval: int = 60, max_iterations: int = None):
        """Main loop with graceful shutdown and optional iteration limit."""
        global keep_running

        print("Starting Cat Detection Service...")
        print(f"Using model for inference...")

        iteration_count = 0

        while keep_running:
            try:
                iteration_count += 1
                print(f"--- Iteration {iteration_count} ---")

                # 1) Capture the image locally
                print("Capturing image...")
                image_path = self.capture_image()
                print(f"Saved image: {image_path}")

                # 2) Upload image to S3 (private) and get a presigned GET URL
                print("Uploading to S3 via presigned URL...")
                s3_get_url = self.upload_and_get_s3_url(image_path)
                if not s3_get_url:
                    print("Failed to upload image to S3 or generate GET URL.")
                    # Optionally delete local file if something failed
                    image_path.unlink(missing_ok=True)
                    # Sleep and continue
                    time.sleep(interval)
                    continue

                # 3) Analyze image (detect cat or 'books')
                print("Analyzing image with Moondream...")
                is_cat, caption = self.detect_cat(image_path)
                if is_cat:
                    print("Cat detected! Sending WhatsApp message...")
                    self.send_whatsapp(s3_get_url, caption)
                else:
                    print("No cat detected. Deleting local image, leaving S3 copy for debugging if desired.")
                    image_path.unlink(missing_ok=True)

                # 4) Check iteration limit
                if max_iterations is not None and iteration_count >= max_iterations:
                    print(f"Reached max_iterations={max_iterations}. Exiting loop.")
                    break

                print(f"Waiting {interval} seconds before next capture...")
                time.sleep(interval)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(interval)

        print("Graceful shutdown complete.")


def main():
    # Attach signal handlers for graceful termination
    signal.signal(signal.SIGINT, handle_termination_signal)
    signal.signal(signal.SIGTERM, handle_termination_signal)

    # Decide which model path to use based on environment variable
    is_small = os.getenv('MODEL_TYPE') == 'small'
    model_path = os.getenv('MOONDREAM_MODEL_PATH_SMALL') if is_small else os.getenv('MOONDREAM_MODEL_PATH_LARGE')
    print(f"Model path is {model_path}")

    detector = CatDetector(model_path=model_path)

    # Example: run for 1 iteration (for testing). Increase as needed.
    detector.run(interval=60, max_iterations=1)


if __name__ == "__main__":
    main()