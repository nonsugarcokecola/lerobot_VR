import http.server
import ssl
import os
import json
from datetime import datetime

PORT = 8000
CERT_FILE = './localhost+2.pem'
KEY_FILE = './localhost+2-key.pem'
DATA_FILE = './controller_data.json'

# Change directory to where the HTML file is located for serving static files
STATIC_DIR = './'

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_POST(self):
        if self.path == '/updatedata':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                # Add a server-side timestamp for when data was received
                data['server_timestamp'] = datetime.utcnow().isoformat() + 'Z'

                with open(DATA_FILE, 'w') as f:
                    json.dump(data, f, indent=4)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*') # Allow CORS for local dev
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'message': 'Data received'}).encode('utf-8'))
                # print(f"Received and wrote to {DATA_FILE}: {data}") # Optional: server-side logging
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Invalid JSON'}).encode('utf-8'))
                print("Error: Invalid JSON received")
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
                print(f"Error processing request: {e}")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Endpoint not found')

    def do_OPTIONS(self): # Handle preflight requests for CORS
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == '__main__':
    # Ensure the static directory exists if it's different from CWD, though here it's /home/ubuntu
    # os.chdir(STATIC_DIR) # Not strictly needed if RequestHandler directory is set and script is run from anywhere
    
    #httpd = http.server.HTTPServer(('192.168.221.106', PORT), RequestHandler)
    httpd = http.server.HTTPServer(('0.0.0.0', PORT), RequestHandler)
    # 创建一个 SSLContext 对象
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"Serving HTTPS on port {PORT} from directory {STATIC_DIR}...")
    print(f"Access your page at: https://localhost:{PORT}/webxr_quest_input.html")
    print(f"Controller data will be saved to: {DATA_FILE}")
    print("Listening for POST requests on /updatedata")
    httpd.serve_forever()
