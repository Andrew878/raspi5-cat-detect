import os
import signal
import sys
import time
from pathlib import Path
from typing import Tuple, List
import json
import requests
import boto3
from picamera2 import Picamera2
import moondream as md
from twilio.rest import Client
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from datetime import datetime, timedelta

# SendGrid imports
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Content

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Global flag for graceful shutdown
keep_running = True

def handle_termination_signal(signum, frame):
    """Signal handler to gracefully exit on Ctrl+C or kill."""
    global keep_running
    keep_running = False
    print("\nReceived termination signal. Stopping after current iteration...")

class CatDetector:
    def __init__(
        self, 
        base_image_dir: str = "./cat_images",
        model_path: str = "moondream-2b-int8.mf",
        message_cooldown_hours: float = 1.0  # e.g. send at most once per hour
    ):
        """
        Initialize the cat detector with camera, model, Twilio, S3, and optional SendGrid config.
        
        :param base_image_dir: Base directory where date-based subfolders are created.
        :param model_path: Path to the Moondream model.
        :param message_cooldown_hours: Number of hours to wait between WhatsApp messages (and email).
        """
        # 1. Initialize Raspberry Pi Camera
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration()
        self.picam2.configure(config)

        # 2. Set base directory for images
        self.base_image_dir = Path(base_image_dir)
        self.base_image_dir.mkdir(parents=True, exist_ok=True)

        # 3. Initialize Moondream model
        self.model = md.vl(model=model_path)

        # 4. Twilio WhatsApp configuration
        self.twilio_from_number = os.getenv('TWILIO_FROM_NUMBER')  # e.g., "whatsapp:+14155238886"
        self.your_number = os.getenv('YOUR_WHATSAPP_NUMBER')       # e.g., "whatsapp:+12345556789"
        self.second_number = os.getenv('SECOND_WHATSAPP_NUMBER')   # e.g., "whatsapp:+9876543210"
        
        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        if not all([self.twilio_account_sid, self.twilio_auth_token]):
            raise ValueError("Missing required environment variables for Twilio (SID, Auth).")

        self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)

        # 5. S3 configuration
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("Missing S3_BUCKET_NAME in environment variables.")
        self.s3_client = boto3.client("s3")

        # 6. Message throttling
        self.message_cooldown_hours = message_cooldown_hours
        self.last_message_time = datetime.min  # track last time we sent a message
        self.cat_detections: List[dict] = []
        # Each entry in `self.cat_detections` is a dict:
        # {
        #    "timestamp": datetime,
        #    "s3_url": str,
        #    "local_path": Path,
        #    "cat_count": int
        # }

        # 7. SendGrid configuration
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")  # If None, skip email
        self.sendgrid_from_email = os.getenv("SENDGRID_FROM_EMAIL") 
        self.sendgrid_to_email = os.getenv("SENDGRID_TO_EMAIL") 

    def capture_image(self) -> Path:
        """
        Capture an image using Raspberry Pi Camera and store it 
        in a date-based subfolder with a timestamp filename.
        """
        # Prepare subfolder by date
        date_folder = self.base_image_dir / time.strftime('%Y-%m-%d')
        date_folder.mkdir(parents=True, exist_ok=True)

        # Construct a timestamp-based filename
        timestamp_str = time.strftime('%H-%M-%S')
        image_path = date_folder / f"{timestamp_str}.jpg"

        self.picam2.start()
        time.sleep(2)  # Allow camera to adjust
        self.picam2.capture_file(str(image_path))
        self.picam2.stop()

        return image_path

    def generate_presigned_put_url(self, object_key: str, expiration=1800) -> str:
        """Generate a presigned URL that allows a PUT request (upload) to S3."""
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
        """Generate a presigned URL that allows a GET request (download) from S3."""
        try:
            response = self.s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
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
        """
        try:
            with open(local_file_path, 'rb') as file_data:
                response = requests.put(
                    presigned_url,
                    data=file_data,
                    headers={'Content-Type': 'image/jpeg'}  
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
        3) Generate and return a presigned GET URL for Twilio (or anyone) to access.
        """
        # Include the subfolder structure in S3 if you want 
        # (here, we just use the same local relative path as the object key)
        object_key = str(local_file_path.relative_to(self.base_image_dir))  
        
        put_url = self.generate_presigned_put_url(object_key, expiration=1800)
        if not put_url:
            print("Failed to generate presigned PUT URL.")
            return None

        print(f"Presigned PUT URL (for uploading): {put_url}")

        success = self.upload_file_via_presigned_url(put_url, local_file_path)
        if not success:
            print("Upload via presigned URL failed.")
            return None

        # Now generate a presigned GET URL so Twilio/Email can fetch this image
        get_url = self.generate_presigned_get_url(object_key, expiration=1800)
        print(f"Presigned GET URL (for Twilio/Email): {get_url}")
        return get_url

    def detect_cat_and_count(self, image_path: Path) -> int:
        """
        Detect how many cats are in the image using Moondream.

        We'll do a simple approach:
         1) Encode image with self.model.encode_image(...)
         2) Query "How many cats are in this image? Provide a number."
         3) Attempt to parse an integer from the result.

        If the text cannot be parsed, fallback to 0.
        """
        try:
            image = Image.open(image_path)
            encoded_image = self.model.encode_image(image)

            # For demonstration, we prompt for the count of cats:
            cat_count_answer = self.model.detect(
                encoded_image,
                "cats"
            )["objects"]
            print(f"Moondream cat count answer: {cat_count_answer}")

            # Attempt to parse a number from the answer
            cat_count = len(cat_count_answer)
            return cat_count

        except Exception as e:
            print(f"Error in detect_cat_and_count: {e}")
            return 0

    def maybe_send_whatsapp_and_email(self):
        """
        Sends a WhatsApp message (and Email) if enough time has passed (>= message_cooldown_hours)
        and if we have cat detections queued.

        We send up to 3 images (evenly spaced in time) among the queued cat detections.
        The message says "Cats detected! (N)" where N is sum of the cat counts in those images.

        We send the identical message to BOTH self.your_number and self.second_number (if set).
        Then, we also send an email via SendGrid if SENDGRID_API_KEY is defined.
        """
        now = datetime.now()
        hours_since_last = (now - self.last_message_time).total_seconds() / 3600.0
        if hours_since_last < self.message_cooldown_hours:
            # Too soon to send again
            return

        if not self.cat_detections:
            # Nothing to send
            return

        # 1) Decide which entries to include: up to 3, evenly spaced
        if len(self.cat_detections) <= 3:
            selected = self.cat_detections
        else:
            # pick the earliest, middle, latest
            selected = [
                self.cat_detections[0],
                self.cat_detections[len(self.cat_detections)//2],
                self.cat_detections[-1]
            ]

        # 2) Construct the Twilio/Email message
        cat_sum = sum(d["cat_count"] for d in selected)
        message_body = f"Cats detected! ({cat_sum})"
        media_urls = [d["s3_url"] for d in selected]

        # 3) Send identical WhatsApp messages to your_number and second_number
        self._send_whatsapp_notifications(message_body, media_urls)

        # 4) Also send an email if configured
        if self.sendgrid_api_key:
            # For email, we can embed the presigned GET URLs as links in HTML
            self.send_email_via_sendgrid(
                subject="Cat Detector Alert",
                body_html=self._build_email_html(message_body, media_urls),
                to_email=self.sendgrid_to_email
            )

        # 5) Update last_message_time and clear out the cat detections queue
        self.last_message_time = now
        self.cat_detections.clear()

    def _send_whatsapp_notifications(self, message_body: str, media_urls: List[str]):
        """
        Helper method to send WhatsApp notifications to the primary and (optional) second number.
        """
        recipients = [self.your_number]
        if self.second_number:
            recipients.append(self.second_number)

        for r in recipients:
            try:
                msg = self.twilio_client.messages.create(
                    from_=self.twilio_from_number,
                    body=message_body,
                    to=r,
                    media_url=media_urls  # up to 10 media URLs are allowed by Twilio
                )
                print(f"WhatsApp to {r} sent successfully! SID: {msg.sid}")
            except Exception as e:
                print(f"Error sending WhatsApp to {r}: {e}")

    def _build_email_html(self, text_body: str, media_urls: List[str]) -> str:
        """
        Given the text message and a list of S3 presigned URLs,
        build an HTML body that includes clickable links (and optional <img> tags).
        """
        # Start with a simple text
        html_content = f"<p>{text_body}</p>"

        # Optionally embed them as images or just links
        for i, url in enumerate(media_urls, start=1):
            # As links
            html_content += f'<p>Image {i}: <a href="{url}">{url}</a></p>'
            # Or embed as <img>
            # html_content += f'<p><img src="{url}" alt="Cat Image {i}" width="400"></p>'

        return html_content

    def send_email_via_sendgrid(self, subject: str, body_html: str, to_email: str) -> None:
        """
        Send an email via SendGrid. 
        If the environment variable SENDGRID_API_KEY is set, we can send emails.
        """
        try:
            message = Mail(
                from_email=self.sendgrid_from_email,
                to_emails=to_email,
                subject=subject,
                html_content=body_html
            )
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)
            print(f"Email sent to {to_email} via SendGrid. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error sending email via SendGrid: {e}")

    def run(self, interval: int = 60, max_iterations: int = None):
        """
        Main loop. 
        1) Capture an image
        2) Upload to S3
        3) Detect cat count
        4) If count > 0, queue detection record
        5) Possibly send WhatsApp/Email if cooldown has expired
        6) Wait `interval` seconds
        """
        global keep_running

        print("Starting Cat Detection Service...")
        print(f"Using model for inference... (cooldown={self.message_cooldown_hours} hr)")

        iteration_count = 0

        while keep_running:
            try:
                iteration_count += 1
                print(f"--- Iteration {iteration_count} ---")

                # 1) Capture the image locally
                print("Capturing image...")
                image_path = self.capture_image()
                print(f"Saved image: {image_path}")

                # 2) Upload image to S3
                print("Uploading to S3 via presigned URL...")
                s3_get_url = self.upload_and_get_s3_url(image_path)
                if not s3_get_url:
                    print("Failed to upload image to S3 or generate GET URL.")
                    image_path.unlink(missing_ok=True)
                    time.sleep(interval)
                    continue

                # 3) Detect cat count
                print("Analyzing image with Moondream for cat count...")
                cat_count = self.detect_cat_and_count(image_path)
                if cat_count > 0:
                    print(f"Detected {cat_count} cat(s). Queuing for next message send.")
                    detection_record = {
                        "timestamp": datetime.now(),
                        "s3_url": s3_get_url,
                        "local_path": image_path,
                        "cat_count": cat_count
                    }
                    self.cat_detections.append(detection_record)
                else:
                    print("No cats detected. Deleting local image.")
                    image_path.unlink(missing_ok=True)

                # 4) Maybe send a WhatsApp message (and email) if cooldown expired
                self.maybe_send_whatsapp_and_email()

                # 5) Check iteration limit
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

    # Example usage:
    detector = CatDetector(
        model_path=model_path,
        message_cooldown_hours=3.0/20  # or any other value you like
    )

    # Example: run for 2 iterations (for a quick test)
    detector.run(interval=60, max_iterations=10)

if __name__ == "__main__":
    main()