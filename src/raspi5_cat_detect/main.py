import time
import os
import signal
import sys
from pathlib import Path

from picamera2 import Picamera2
import moondream as md
from twilio.rest import Client
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global flag to indicate we should keep running (until a signal says "stop")
keep_running = True


def handle_termination_signal(signum, frame):
    """Signal handler to gracefully exit on Ctrl+C or kill."""
    global keep_running
    keep_running = False
    print("\nReceived termination signal. Stopping after current iteration...")


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
        self.twilio_from_number = os.getenv('TWILIO_FROM_NUMBER')
        self.your_number = os.getenv('YOUR_WHATSAPP_NUMBER')

  
        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_account_auth=os.getenv('TWILIO_AUTH_TOKEN')

        if not all([self.twilio_account_sid,self.twilio_account_auth]):
            raise ValueError("Missing required environment variables for Twilio")

        # Initialize Twilio client with API key + secret
        # self.twilio_client = Client(
        #     self.twilio_api_key_sid,
          #   self.twilio_api_key_secret,
            # account_sid=self.twilio_account_sid)
        print(self.twilio_account_sid, self.twilio_account_auth)
        self.twilio_client=Client(self.twilio_account_sid, self.twilio_account_auth)

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

        # Then specifically query for cat presence (or your custom prompt)
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
                media_url=[f'https://images.unsplash.com/photo-1516280030429-27679b3dc9cf?q=80&w=2970&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D']
            )
            print(f"WhatsApp message sent successfully! Message ID: {message.sid}")
            return True
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
            return False

    def run(self, interval: int = 60, max_iterations: int = None):
        """Main execution method with optional iteration limit."""
        global keep_running  # We'll check this flag each loop

        print("Starting Cat Detection Service...")
        print(f"Using model for inference...")

        iteration_count = 0

        while keep_running:
            try:
                iteration_count += 1
                print(f"--- Iteration {iteration_count} ---")

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

                # Check if we've hit the max_iterations limit
                if max_iterations is not None and iteration_count >= max_iterations:
                    print(f"Reached maximum number of iterations ({max_iterations}). Exiting.")
                    break

                print(f"Waiting {interval} seconds before next capture...")
                time.sleep(interval)

            except Exception as e:
                print(f"Error: {e}")
                # Optionally break out on error, or just wait and continue
                time.sleep(interval)

        print("Graceful shutdown complete.")


def main():
    # Attach signal handlers for graceful termination
    signal.signal(signal.SIGINT, handle_termination_signal)
    signal.signal(signal.SIGTERM, handle_termination_signal)

    # Use environment variable for model path if provided
    is_small = os.getenv('MODEL_TYPE') == 'small'
    model_path = os.getenv('MOONDREAM_MODEL_PATH_SMALL') if is_small else os.getenv('MOONDREAM_MODEL_PATH_LARGE')
    print(f"Model path is {model_path}")

    # Create the detector instance
    detector = CatDetector(
        model_path=model_path
        # Optionally specify a custom image_dir here if needed
    )

    # Run with interval=60 seconds, up to 10 iterations as an example
    detector.run(interval=60, max_iterations=1)


if __name__ == "__main__":
    main()