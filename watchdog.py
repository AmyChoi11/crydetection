
import asyncio
import json
import logging
import time
import websockets
import requests
import socket
from datetime import datetime

# Configuration
WEBSOCKET_URL = "ws://192.168.77.141:5000/ws"  # Replace with your server's IP on the hotspot
WATCHY_IP = "192.168.77.141"  # Replace with your Watchy's IP on the hotspot
CONFIDENCE_THRESHOLD = 0.3
TIME_THRESHOLD = 10
NOTIFICATION_COOLDOWN = 60

# Only notify for these specific baby states
ALERT_STATES = ["crying", "uncomfortable", "tired"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('watchdog-notifier')

# Rest of your class and functions...
class WatchdogNotifier:
    def __init__(self):
        self.last_notification_time = 0
        self.high_confidence_start = 0
        self.current_reason = None
        self.current_confidence = 0
        self.sustained_high_confidence = False
        self.connected = False

    async def connect_websocket(self):
        """Connect to the cry detection system WebSocket"""
        while True:
            try:
                logger.info(f"Connecting to WebSocket at {WEBSOCKET_URL}")
                async with websockets.connect(WEBSOCKET_URL) as websocket:
                    logger.info("WebSocket connection established")
                    self.connected = True
                    
                    # Send initial message to keep connection alive
                    await websocket.send("ping")
                    
                    # Process incoming messages
                    await self.process_messages(websocket)
            except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
                logger.error(f"WebSocket connection error: {str(e)}")
                self.connected = False
                logger.info("Trying to reconnect in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                await asyncio.sleep(5)

    async def process_messages(self, websocket):
        """Process incoming WebSocket messages"""
        while True:
            try:
                # Receive message from WebSocket
                message = await websocket.recv()
                data = json.loads(message)
                
                # Process prediction data
                if data.get("status") == "success":
                    reason = data.get("reason")
                    confidence = data.get("confidence", 0.0)
                    
                    # Skip silent predictions and non-alert states
                    if reason == "silent" or reason not in ALERT_STATES:
                        logger.info(f"Ignoring non-alert state: {reason}")
                        self.reset_tracking()
                        continue
                    
                    # Track prediction
                    self.track_prediction(reason, confidence)
                
                # Keep connection alive by sending ping every 30 seconds
                await websocket.send("ping")
                
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON message")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                raise

    def track_prediction(self, reason, confidence):
        """Track predictions and trigger notifications when threshold is met"""
        current_time = time.time()
        
        # Log the current prediction
        logger.info(f"Alert state detected: {reason}, Confidence: {confidence:.2f}")
        
        # Update current prediction
        self.current_reason = reason
        self.current_confidence = confidence
        
        # Check if confidence exceeds threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            # If this is the start of high confidence, record the time
            if self.high_confidence_start == 0:
                self.high_confidence_start = current_time
                logger.info(f"Started tracking high confidence alert: {reason}")
            
            # Check if high confidence has been sustained for the required time
            elapsed = current_time - self.high_confidence_start
            if elapsed >= TIME_THRESHOLD and not self.sustained_high_confidence:
                self.sustained_high_confidence = True
                logger.info(f"High confidence sustained for {elapsed:.1f} seconds")
                
                # Check cooldown period
                if current_time - self.last_notification_time >= NOTIFICATION_COOLDOWN:
                    # Trigger notification
                    self.notify_watchy(reason, confidence)
                    self.last_notification_time = current_time
                else:
                    logger.info("Notification cooldown period active, skipping notification")
        else:
            # Reset tracking if confidence drops below threshold
            self.reset_tracking()

    def reset_tracking(self):
        """Reset high confidence tracking"""
        if self.high_confidence_start > 0:
            logger.info("Confidence dropped below threshold, resetting tracking")
        self.high_confidence_start = 0
        self.sustained_high_confidence = False

    def notify_watchy(self, reason, confidence):
        """Send notification to Watchy device"""
        try:
            logger.info(f"Sending vibration notification to Watchy for: {reason}")
            url = f"http://{WATCHY_IP}/vibrate"
            
            # Try multiple notification methods
            
            # Method 1: Simple POST (no data)
            try:
                response = requests.post(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Notification sent successfully via simple POST")
                    return
            except Exception as e:
                logger.warning(f"Simple POST failed: {str(e)}")
                
            # Method 2: Send as form data
            try:
                response = requests.post(
                    url, 
                    data={"reason": reason},
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"Notification sent successfully via form data")
                    return
            except Exception as e:
                logger.warning(f"Form data POST failed: {str(e)}")
                
            # Method 3: Try with query parameters
            try:
                response = requests.post(
                    f"{url}?reason={reason}",
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"Notification sent successfully via URL parameters")
                    return
            except Exception as e:
                logger.warning(f"URL parameter POST failed: {str(e)}")
                
            logger.error(f"All notification methods failed for Watchy")
        except requests.RequestException as e:
            logger.error(f"Error sending notification to Watchy: {str(e)}")
def scan_network():
    base_ip = "192.168.77."
    print(f"Scanning network {base_ip}x for devices...")
    open_devices = []
    
    for i in range(1, 255):
        ip = f"{base_ip}{i}"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)
            result = s.connect_ex((ip, 80))
            if result == 0:
                print(f"Found device at {ip} with open port 80!")
                open_devices.append(ip)
            s.close()
        except:
            pass
    
    return open_devices  # Return the list of discovered devices
def find_watchy_mdns():
    """Try to find Watchy using mDNS service discovery"""
    try:
        import zeroconf
        from zeroconf import ServiceBrowser, Zeroconf

        class WatchyListener:
            def __init__(self):
                self.watchy_address = None
            
            def add_service(self, zc, type, name):
                info = zc.get_service_info(type, name)
                if info and ('watchy' in name.lower() or 'esp32' in name.lower()):
                    self.watchy_address = socket.inet_ntoa(info.addresses[0])
                    print(f"Found Watchy at {self.watchy_address}")
        
        listener = WatchyListener()
        zc = Zeroconf()
        browser = ServiceBrowser(zc, "_http._tcp.local.", listener)
        
        # Wait a bit for discovery
        print("Searching for Watchy via mDNS (5 seconds)...")
        time.sleep(5)
        zc.close()
        
        return listener.watchy_address
    except ImportError:
        print("zeroconf package not installed, skipping mDNS discovery")
        return None
    except Exception as e:
        print(f"Error during mDNS discovery: {e}")
        return None
    
async def main():
    """Main entry point"""
    logger.info("Starting Watchdog Notification Service")
    logger.info(f"WebSocket URL: {WEBSOCKET_URL}")
    logger.info(f"Watchy IP: {WATCHY_IP}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD * 100}%")
    logger.info(f"Time threshold: {TIME_THRESHOLD} seconds")
    logger.info(f"Alert states: {', '.join(ALERT_STATES)}")
    
    notifier = WatchdogNotifier()
    await notifier.connect_websocket()


async def test_connection():
    """Test connection to the services"""
    
    # Test cry detection server connection
    server_parts = WEBSOCKET_URL.replace("ws://", "").replace("wss://", "").split("/")[0]
    server_host = server_parts.split(":")[0]
    server_port = int(server_parts.split(":")[1]) if ":" in server_parts else 80
    
    logger.info(f"Testing connection to detection server: {server_host}:{server_port}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((server_host, server_port))
        if result == 0:
            logger.info("✅ Server port is open and reachable")
        else:
            logger.error(f"❌ Cannot connect to server port: {result}")
        sock.close()
    except Exception as e:
        logger.error(f"❌ Error testing server connection: {e}")
    
    # Test Watchy connection
    logger.info(f"Testing connection to Watchy: {WATCHY_IP}")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((WATCHY_IP, 80))
        if result == 0:
            logger.info("✅ Watchy is reachable")
        else:
            logger.error(f"❌ Cannot connect to Watchy: {result}")
        sock.close()
    except Exception as e:
        logger.error(f"❌ Error testing Watchy connection: {e}")

if __name__ == "__main__":
    # Run network scan first to find Watchy
    print("Scanning network to find Watchy...")
    devices = scan_network()
    
    if devices:
        print(f"Found {len(devices)} potential devices with web servers:")
        for ip in devices:
            print(f"  - {ip}")
        
        # Filter out known server IP
        watchy_candidates = [ip for ip in devices if ip != "192.168.77.141"]
        if watchy_candidates:
            print(f"\nLikely Watchy IP: {watchy_candidates[0]}")
            print(f"Update WATCHY_IP in your code to use this IP")
            
            # Option to automatically use the first candidate
            use_ip = input(f"Use {watchy_candidates[0]} as Watchy IP? (y/n): ").lower()
            if use_ip == 'y':
                WATCHY_IP = watchy_candidates[0]
                print(f"Using {WATCHY_IP} for Watchy")
    else:
        print("No devices with web servers found on the network")
    
    # Run connection tests
    asyncio.run(test_connection())
    
    # Run main program
    asyncio.run(main())