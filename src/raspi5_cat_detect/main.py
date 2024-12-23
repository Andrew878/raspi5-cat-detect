import time
import os
from pathlib import Path
from picamera2 import Picamera2

import moondream as md
from twilio.rest import Client
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CatDetector:
    def __init__(self, image_dir: str = "./cat_images", model_path: str = "moondream-2b-int8.mf"):
        """Initialize the cat detector with camera, model, and messaging setup"""
        # Initialize Raspberry Pi Camera
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration()
        self.picam2.configure(config)

        # Set image directory
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Moondream model
        self.model = md.vl(model=model_path)

        # Twilio WhatsApp configuration
        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.twilio_from_number = os.getenv('TWILIO_FROM_NUMBER')
        self.your_number = os.getenv('YOUR_WHATSAPP_NUMBER')

        if not all([self.twilio_account_sid, self.twilio_auth_token,
                    self.twilio_from_number, self.your_number]):
            raise ValueError("Missing required environment variables for Twilio")

        self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)

    def capture_image(self) -> Path:
        """Capture an image using Raspberry Pi Camera"""
        self.picam2.start()
        time.sleep(2)  # Allow camera to adjust

        image_path = self.image_dir / f'cat_image_{int(time.time())}.jpg'
        self.picam2.capture_file(str(image_path))
        self.picam2.stop()

        return image_path

    def detect_cat(self, image_path: Path) -> bool:
        """Detect if a cat is present in the image using Moondream"""
        # Load and encode image
        image = Image.open(image_path)
        encoded_image = self.model.encode_image(image)

        # First get a general caption to confirm image quality
        caption = self.model.caption(encoded_image)["caption"]
        print(f"Image caption: {caption}")

        # Then specifically query for cat presence
        answer = self.model.query(
            encoded_image,
            "Are there books in this image? Answer with just 'yes' or 'no'."
        )["answer"]
        print(f"Query answer: {answer}")

        return answer.lower().strip() == "yes"

    def send_whatsapp(self, image_path: Path, caption: str = None) -> bool:
        """Send image via WhatsApp using Twilio"""
        try:
            message_body = 'Cat detected!'
            if caption:
                message_body += f'\nImage description: {caption}'

            message = self.twilio_client.messages.create(
                from_=self.twilio_from_number,
                body=message_body,
                to=self.your_number,
                media_url=[f'file://{image_path.absolute()}']
            )
            print(f"WhatsApp message sent successfully! Message ID: {message.sid}")
            return True
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
            return False

    def run(self, interval: int = 60):
        """Main execution method"""
        print("Starting Cat Detection Service...")
        print(f"Using model for inference...")

        while True:
            try:
                print("Capturing image...")
                image_path = self.capture_image()
                print(f"Attempted to capture image to: {image_path}")
                print(f"Directory exists? {self.image_dir.exists()}")
                print(f"File now exists? {image_path.exists()}")

                print("Analyzing image...")
                if self.detect_cat(image_path):
                    print("Cat detected! Sending WhatsApp message...")
                    self.send_whatsapp(image_path)
                else:
                    print("No cat detected in this image.")
                    image_path.unlink()  # Delete image if no cat detected

                print(f"Waiting {interval} seconds before next capture...")
                time.sleep(interval)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(interval)


def main():
    # Use environment variable for model path if provided
    is_small = os.getenv('MODEL_TYPE') == 'small'
    model_path = os.getenv('MOONDREAM_MODEL_PATH_SMALL') if is_small else os.getenv('MOONDREAM_MODEL_PATH_LARGE')
    print(f"Model path is {model_path}")
    detector = CatDetector(model_path=model_path)
    detector.run()


if __name__ == "__main__":
    main()