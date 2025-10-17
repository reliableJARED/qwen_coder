# combined.py
import ngrok
import multiprocessing
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
import os
import json
load_dotenv()
"""A simple HTTP server with ngrok tunneling and OAuth-based access control.
RESOURCES:
- Ngrok OAuth Documentation: https://ngrok.com/docs/ngrok-agent/configuration/traffic-policies/oauth
- Ngrok Python SDK: https://ngrok.com/docs/getting-started/python
- Ngrok Quick Reference: https://ngrok.com/blog-post/the-ngrok-cheat-sheet
"""

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # If authorized, serve the content
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"Hello from local Python HTTP Server!".encode())


def run_service():
    """Runs in separate process"""
    server = HTTPServer(("localhost", 8080), SimpleHandler)
    print("Service running on localhost:8080")
    server.serve_forever()

def run_ngrok():
    """Runs in main process"""
    # Secure your endpoint with a traffic policy.
    # traffic_policy could also be saved as a policy.yaml traffic policy file. Since it's simple, we define inline here.
    # https://ngrok.com/blog-post/the-ngrok-cheat-sheet
    allowed_emails_str=os.getenv("AUTHORIZED_EMAILS","[]")# emails are in a list in .env file
    allowed_emails = json.loads(allowed_emails_str)
    allowed_emails_expr  = ",".join([f"'{email}'" for email in allowed_emails])

    traffic_policy = {
        "on_http_request": [
            {
                "actions": [
                    {
                        "type": "oauth",
                        "config": {
                            "provider": "google"
                        }
                    }
                ]
            },
            {
                "expressions": [
                    f"!(actions.ngrok.oauth.identity.email in [{allowed_emails_expr}])"
                ],
                "actions": [
                    {
                        "type": "custom-response",
                        "config": {
                            "status_code": 401,
                            "content": "text/plain",
                            "body": "Unauthorized access bro. Your email is not allowed to access this resource."
                        }
                    }
                ]
            }
        ]
    }

    listener = ngrok.forward(
        # The port your app is running on.
        8080,
        authtoken=os.getenv("NGROK_AUTHTOKEN"),
        domain=os.getenv("NGROK_DOMAIN"),
        traffic_policy=json.dumps(traffic_policy)
        )
    print(f"Ingress established at: {listener.url()}")

if __name__ == "__main__":
    # Start service in separate process
    service_process = multiprocessing.Process(target=run_service, daemon=True)
    service_process.start()
    
    # Give service time to start
    import time
    time.sleep(1)
    
    # Start ngrok
    run_ngrok()
    
    # Keep alive
    input("Press Enter to stop...")