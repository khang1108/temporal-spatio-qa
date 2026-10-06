#!/usr/bin/env python3
"""
Dedicated HTTP Server for SpatialMQA Survey Dashboard.
Serves survey.html directly on port 3000 (http://localhost:3000).
"""
import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse

PORT = 3000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SurveyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def translate_path(self, path):
        # Strip query strings and hash anchors
        clean_path = urlparse(path).path
        if clean_path in ('/', '', '/index.html'):
            clean_path = '/survey.html'
        return super().translate_path(clean_path)

def run():
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), SurveyHandler) as httpd:
            print("=" * 60)
            print(f"🚀 Survey Dashboard is running at: http://localhost:{PORT}")
            print(f"📄 Automatically serving: survey.html")
            print(f"🛑 Press Ctrl+C in this terminal to stop the server")
            print("=" * 60)
            sys.stdout.flush()
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️ Port {PORT} đang được sử dụng. Đang giải phóng port {PORT}...")
            os.system(f"fuser -k {PORT}/tcp 2>/dev/null")
            with socketserver.TCPServer(("", PORT), SurveyHandler) as httpd:
                print(f"🚀 Đã khởi động lại Survey Dashboard tại: http://localhost:{PORT}")
                sys.stdout.flush()
                httpd.serve_forever()
        else:
            raise e

if __name__ == '__main__':
    run()
