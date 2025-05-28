#!/bin/bash

# Get local IP address
get_ip() {
    if command -v ip &> /dev/null; then
        ip addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n1
    elif command -v ifconfig &> /dev/null; then
        ifconfig | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n1
    else
        echo "Unknown IP"
    fi
}

SERVER_IP=$(get_ip)

echo "====================================="
echo "Baby Cry Detection System"
echo "====================================="
echo "Starting server on IP: $SERVER_IP"

cd "$(dirname "$0")/app"

# Export IP as environment variable
export SERVER_IP="$SERVER_IP"
echo "Starting Baby Cry Detection System..."
python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload &

# Wait for server to start
sleep 2

echo "====================================="
echo "Server started successfully!"
echo "====================================="
echo "Local Access: http://localhost:5000"
echo "Network Access: http://$SERVER_IP:5000" 
echo "Configuration: http://$SERVER_IP:5000/system-config"
echo "====================================="