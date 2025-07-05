#!/bin/bash
# Debug script để kiểm tra Lua script và OpenResty config

echo "🔍 Debugging MCCVA System..."
echo "================================"

# 1. Kiểm tra OpenResty config
echo "📋 Checking OpenResty config..."
if [ -f "/usr/local/openresty/nginx/conf/nginx.conf" ]; then
    echo "✅ nginx.conf exists"
    grep -n "mccva_routing.lua" /usr/local/openresty/nginx/conf/nginx.conf || echo "❌ Lua script not found in nginx.conf"
else
    echo "❌ nginx.conf not found"
fi

# 2. Kiểm tra Lua script
echo ""
echo "📋 Checking Lua script..."
if [ -f "/opt/mccva/lua/mccva_routing.lua" ]; then
    echo "✅ Lua script exists"
    echo "Last modified: $(stat -c %y /opt/mccva/lua/mccva_routing.lua)"
    
    # Kiểm tra format data gửi đến ML Service
    echo ""
    echo "🔍 Checking ML Service request format..."
    grep -A 10 "features = {" /opt/mccva/lua/mccva_routing.lua || echo "❌ features format not found"
else
    echo "❌ Lua script not found"
fi

# 3. Kiểm tra OpenResty status
echo ""
echo "📋 Checking OpenResty status..."
sudo systemctl status openresty --no-pager

# 4. Test ML Service trực tiếp
echo ""
echo "🤖 Testing ML Service directly..."
curl -s -X POST http://localhost:5000/predict/makespan \
  -H "Content-Type: application/json" \
  -d '{"features": [4, 8, 100, 1000, 3]}' | jq . || echo "❌ ML Service test failed"

# 5. Test OpenResty endpoint
echo ""
echo "🌐 Testing OpenResty endpoint..."
curl -s -X POST http://localhost/mccva/route \
  -H "Content-Type: application/json" \
  -d '{"cpu_cores": 4, "memory": 8, "storage": 100, "network_bandwidth": 1000, "priority": 3}' | jq . || echo "❌ OpenResty test failed"

# 6. Kiểm tra OpenResty logs
echo ""
echo "📋 Recent OpenResty logs..."
sudo journalctl -u openresty --no-pager | tail -10

echo ""
echo "✅ Debug completed!" 