
import socket
from nicegui import ui

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

# Get current IP
camera_ip = "172.20.10.2"  # Camera IP - keep this separate
local_ip = get_local_ip()   # Server IP - detect automatically

# Dynamic iframe that shows both server and camera status
iframe = f'<iframe width="350" height="260" src="http://{camera_ip}/" allowfullscreen></iframe>'
status_html = f'<p>Server running on: <a href="http://{local_ip}:5000">http://{local_ip}:5000</a></p>'

@ui.page('/')
def main():
    ui.add_body_html(iframe)
    ui.add_body_html(status_html)
    ui.label('Your baby is tired!').style('color: #Ff8c00; font-size: 200%; font-weight: 300')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0", port=8080)  # Run on all interfaces

