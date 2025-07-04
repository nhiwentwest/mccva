#!/bin/bash
# setup_lua_scripts.sh - Copy Lua scripts vào đúng vị trí cho OpenResty

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }

# Tạo thư mục Lua
LUA_DIR="/usr/local/openresty/nginx/conf/lua"
log "Tạo thư mục Lua: $LUA_DIR"
sudo mkdir -p "$LUA_DIR"
sudo chown -R ubuntu:ubuntu "$LUA_DIR"

# Copy Lua scripts
log "Copy Lua scripts..."

# MCCVA Routing script
cat > "$LUA_DIR/mccva_routing.lua" << 'EOF'
#!/usr/bin/env lua
-- mccva_routing.lua - MCCVA Algorithm Implementation
-- Triển khai thuật toán MCCVA với retry/fallback logic và improved load balancing

local http = require "resty.http"
local cjson = require "cjson.safe"
local ngx = ngx

-- Shared memory zones
local ml_cache = ngx.shared.ml_cache
local vm_info = ngx.shared.vm_info
local mccva_stats = ngx.shared.mccva_stats

-- Configuration
local ML_SERVICE_URL = "http://127.0.0.1:5000"
local MAX_RETRIES = 3
local RETRY_DELAY = 1  -- seconds

-- Mock server endpoints với load balancing weights
local MOCK_SERVERS = {
    [1] = {url = "http://127.0.0.1:8081", weight = 3, priority = 1},
    [2] = {url = "http://127.0.0.1:8082", weight = 2, priority = 2},
    [3] = {url = "http://127.0.0.1:8083", weight = 3, priority = 3},
    [4] = {url = "http://127.0.0.1:8084", weight = 2, priority = 4},
    [5] = {url = "http://127.0.0.1:8085", weight = 3, priority = 5},
    [6] = {url = "http://127.0.0.1:8086", weight = 2, priority = 6},
    [7] = {url = "http://127.0.0.1:8087", weight = 3, priority = 7},
    [8] = {url = "http://127.0.0.1:8088", weight = 2, priority = 8}
}

-- Utility functions
local function log_info(msg)
    ngx.log(ngx.INFO, "[MCCVA] " .. msg)
end

local function log_error(msg)
    ngx.log(ngx.ERR, "[MCCVA] " .. msg)
end

local function get_request_id()
    return ngx.var.request_id or ngx.var.connection .. "_" .. ngx.var.connection_requests
end

-- HTTP request helper với retry logic
local function http_request_with_retry(url, method, body, headers, max_retries)
    max_retries = max_retries or MAX_RETRIES
    local request_id = get_request_id()
    
    for attempt = 1, max_retries do
        log_info(string.format("[%s] Attempt %d/%d: %s %s", request_id, attempt, max_retries, method, url))
        
        local httpc = http.new()
        httpc:set_timeout(10000)  -- 10 seconds timeout
        
        local res, err = httpc:request_uri(url, {
            method = method,
            body = body,
            headers = headers or {}
        })
        
        if res then
            log_info(string.format("[%s] Success: %d", request_id, res.status))
            return res
        else
            log_error(string.format("[%s] Attempt %d failed: %s", request_id, attempt, err))
            
            if attempt < max_retries then
                log_info(string.format("[%s] Retrying in %d seconds...", request_id, RETRY_DELAY))
                ngx.sleep(RETRY_DELAY)
            end
        end
    end
    
    log_error(string.format("[%s] All %d attempts failed", request_id, max_retries))
    return nil, "All retry attempts failed"
end

-- Get ML prediction
local function get_ml_prediction(features)
    local request_id = get_request_id()
    log_info(string.format("[%s] Getting ML prediction for features: %s", request_id, cjson.encode(features)))
    
    local ml_request = {
        features = features
    }
    
    local res, err = http_request_with_retry(
        ML_SERVICE_URL .. "/predict/makespan",
        "POST",
        cjson.encode(ml_request),
        {["Content-Type"] = "application/json"}
    )
    
    if not res then
        log_error(string.format("[%s] ML prediction failed: %s", request_id, err))
        return nil, err
    end
    
    if res.status ~= 200 then
        log_error(string.format("[%s] ML prediction returned status %d: %s", request_id, res.status, res.body))
        return nil, "ML service error: " .. res.status
    end
    
    local prediction = cjson.decode(res.body)
    if not prediction then
        log_error(string.format("[%s] Failed to decode ML prediction", request_id))
        return nil, "Invalid ML response"
    end
    
    log_info(string.format("[%s] ML prediction: %s (confidence: %s)", request_id, prediction.makespan, prediction.confidence))
    return prediction
end

-- Improved load balancing selection
local function select_server_with_load_balancing(makespan, priority, confidence)
    local request_id = get_request_id()
    
    -- Get current server loads from shared memory
    local server_loads = {}
    for server_id, server_info in pairs(MOCK_SERVERS) do
        local load_key = "load_" .. server_id
        local current_load = mccva_stats:get(load_key) or 0
        server_loads[server_id] = {
            id = server_id,
            url = server_info.url,
            weight = server_info.weight,
            priority = server_info.priority,
            current_load = current_load,
            score = 0
        }
    end
    
    -- Calculate server scores based on multiple factors
    for server_id, server_info in pairs(server_loads) do
        local score = 0
        
        -- Factor 1: Makespan-based routing (40% weight)
        if makespan == "small" and server_id <= 2 then
            score = score + 40
        elseif makespan == "medium" and server_id >= 3 and server_id <= 5 then
            score = score + 40
        elseif makespan == "large" and server_id >= 6 then
            score = score + 40
        else
            score = score + 20  -- Partial score for other cases
        end
        
        -- Factor 2: Priority-based routing (30% weight)
        if server_info.priority == priority then
            score = score + 30
        elseif math.abs(server_info.priority - priority) == 1 then
            score = score + 20
        else
            score = score + 10
        end
        
        -- Factor 3: Load balancing (20% weight)
        local load_factor = math.max(0, 20 - (server_info.current_load * 2))
        score = score + load_factor
        
        -- Factor 4: Server weight (10% weight)
        score = score + (server_info.weight * 2)
        
        server_info.score = score
    end
    
    -- Sort servers by score (descending)
    local sorted_servers = {}
    for _, server_info in pairs(server_loads) do
        table.insert(sorted_servers, server_info)
    end
    table.sort(sorted_servers, function(a, b) return a.score > b.score end)
    
    -- Select top 3 servers and use weighted random selection
    local candidates = {}
    local total_weight = 0
    
    for i = 1, math.min(3, #sorted_servers) do
        local server = sorted_servers[i]
        local weight = server.weight * (1 / (1 + server.current_load * 0.1))
        table.insert(candidates, {server = server, weight = weight})
        total_weight = total_weight + weight
    end
    
    -- Weighted random selection
    local random_value = math.random() * total_weight
    local current_weight = 0
    
    for _, candidate in ipairs(candidates) do
        current_weight = current_weight + candidate.weight
        if random_value <= current_weight then
            local selected_server = candidate.server
            
            -- Update server load
            local load_key = "load_" .. selected_server.id
            local new_load = (mccva_stats:get(load_key) or 0) + 1
            mccva_stats:set(load_key, new_load)
            
            log_info(string.format("[%s] Selected server %d (score: %.2f, load: %d)", 
                                 request_id, selected_server.id, selected_server.score, new_load))
            
            return selected_server.id, selected_server.url
        end
    end
    
    -- Fallback to first server
    local fallback_server = sorted_servers[1]
    local load_key = "load_" .. fallback_server.id
    local new_load = (mccva_stats:get(load_key) or 0) + 1
    mccva_stats:set(load_key, new_load)
    
    log_info(string.format("[%s] Fallback to server %d", request_id, fallback_server.id))
    return fallback_server.id, fallback_server.url
end

-- Route request to appropriate server
local function route_request(server_number, request_data)
    local request_id = get_request_id()
    local server_url = MOCK_SERVERS[server_number].url
    
    if not server_url then
        log_error(string.format("[%s] Invalid server number: %d", request_id, server_number))
        return nil, "Invalid server number"
    end
    
    log_info(string.format("[%s] Routing to server %d: %s", request_id, server_number, server_url))
    
    local res, err = http_request_with_retry(
        server_url .. "/process",
        "POST",
        cjson.encode(request_data),
        {["Content-Type"] = "application/json"}
    )
    
    if not res then
        log_error(string.format("[%s] Server %d request failed: %s", request_id, server_number, err))
        return nil, err
    end
    
    if res.status ~= 200 then
        log_error(string.format("[%s] Server %d returned status %d", request_id, server_number, res.status))
        return nil, "Server error: " .. res.status
    end
    
    local response = cjson.decode(res.body)
    if not response then
        log_error(string.format("[%s] Failed to decode server %d response", request_id, server_number))
        return nil, "Invalid server response"
    end
    
    log_info(string.format("[%s] Server %d response: %s", request_id, server_number, cjson.encode(response)))
    return response
end

-- Main MCCVA routing logic
local function mccva_route()
    local request_id = get_request_id()
    log_info(string.format("[%s] Starting MCCVA routing", request_id))
    
    -- Set response headers
    ngx.header.content_type = "application/json"
    
    -- Parse request body
    ngx.req.read_body()
    local body = ngx.req.get_body_data()
    
    if not body then
        log_error(string.format("[%s] No request body", request_id))
        ngx.status = 400
        ngx.say(cjson.encode({error = "No request body"}))
        return
    end
    
    local request_data = cjson.decode(body)
    if not request_data then
        log_error(string.format("[%s] Invalid JSON in request body", request_id))
        ngx.status = 400
        ngx.say(cjson.encode({error = "Invalid JSON"}))
        return
    end
    
    -- Validate required fields
    local required_fields = {"cpu_cores", "memory", "storage", "network_bandwidth", "priority"}
    for _, field in ipairs(required_fields) do
        if not request_data[field] then
            log_error(string.format("[%s] Missing required field: %s", request_id, field))
            ngx.status = 400
            ngx.say(cjson.encode({error = "Missing required field: " .. field}))
            return
        end
    end
    
    -- Prepare features for ML prediction
    local features = {
        request_data.cpu_cores,
        request_data.memory,
        request_data.storage,
        request_data.network_bandwidth,
        request_data.priority
    }
    
    -- Get ML prediction
    local prediction, err = get_ml_prediction(features)
    if not prediction then
        log_error(string.format("[%s] ML prediction failed: %s", request_id, err))
        
        -- Fallback: use priority-based routing
        log_info(string.format("[%s] Using fallback routing based on priority", request_id))
        local fallback_server = math.min(request_data.priority, #MOCK_SERVERS)
        prediction = {makespan = "fallback", confidence = 0.5}
        
        local response, route_err = route_request(fallback_server, request_data)
        if not response then
            ngx.status = 500
            ngx.say(cjson.encode({error = "Fallback routing failed: " .. route_err}))
            return
        end
        
        ngx.say(cjson.encode({
            server = "fallback_" .. fallback_server,
            prediction = prediction,
            response = response,
            fallback = true
        }))
        return
    end
    
    -- Select server using improved load balancing
    local server_number, server_url = select_server_with_load_balancing(
        prediction.makespan, 
        request_data.priority, 
        prediction.confidence
    )
    
    -- Route request to selected server
    local response, err = route_request(server_number, request_data)
    if not response then
        log_error(string.format("[%s] Routing failed: %s", request_id, err))
        ngx.status = 500
        ngx.say(cjson.encode({error = "Routing failed: " .. err}))
        return
    end
    
    -- Update statistics
    local stats_key = "requests_" .. server_number
    local current_stats = mccva_stats:get(stats_key) or 0
    mccva_stats:set(stats_key, current_stats + 1)
    
    -- Return response
    local result = {
        server = "server_" .. server_number,
        prediction = prediction,
        response = response,
        request_id = request_id,
        load_balancing_info = {
            selected_server = server_number,
            server_url = server_url,
            selection_method = "improved_load_balancing"
        }
    }
    
    log_info(string.format("[%s] MCCVA routing completed successfully", request_id))
    ngx.say(cjson.encode(result))
end

-- Execute main function
mccva_route()
EOF

# Predict makespan script
cat > "$LUA_DIR/predict_makespan.lua" << 'EOF'
#!/usr/bin/env lua
-- predict_makespan.lua - Proxy to ML Service for makespan prediction

local http = require "resty.http"
local cjson = require "cjson.safe"
local ngx = ngx

local function proxy_to_ml()
    ngx.header.content_type = "application/json"
    
    -- Read request body
    ngx.req.read_body()
    local body = ngx.req.get_body_data()
    
    if not body then
        ngx.status = 400
        ngx.say(cjson.encode({error = "No request body"}))
        return
    end
    
    -- Forward to ML service
    local httpc = http.new()
    httpc:set_timeout(10000)
    
    local res, err = httpc:request_uri("http://127.0.0.1:5000/predict/makespan", {
        method = "POST",
        body = body,
        headers = {
            ["Content-Type"] = "application/json"
        }
    })
    
    if not res then
        ngx.status = 500
        ngx.say(cjson.encode({error = "ML service error: " .. (err or "unknown")}))
        return
    end
    
    ngx.status = res.status
    ngx.say(res.body)
end

proxy_to_ml()
EOF

# Predict VM cluster script
cat > "$LUA_DIR/predict_vm_cluster.lua" << 'EOF'
#!/usr/bin/env lua
-- predict_vm_cluster.lua - Proxy to ML Service for VM cluster prediction

local http = require "resty.http"
local cjson = require "cjson.safe"
local ngx = ngx

local function proxy_to_ml()
    ngx.header.content_type = "application/json"
    
    -- Read request body
    ngx.req.read_body()
    local body = ngx.req.get_body_data()
    
    if not body then
        ngx.status = 400
        ngx.say(cjson.encode({error = "No request body"}))
        return
    end
    
    -- Forward to ML service
    local httpc = http.new()
    httpc:set_timeout(10000)
    
    local res, err = httpc:request_uri("http://127.0.0.1:5000/predict/vm_cluster", {
        method = "POST",
        body = body,
        headers = {
            ["Content-Type"] = "application/json"
        }
    })
    
    if not res then
        ngx.status = 500
        ngx.say(cjson.encode({error = "ML service error: " .. (err or "unknown")}))
        return
    end
    
    ngx.status = res.status
    ngx.say(res.body)
end

proxy_to_ml()
EOF

# Set permissions
log "Set permissions cho Lua scripts..."
sudo chmod +x "$LUA_DIR"/*.lua
sudo chown -R ubuntu:ubuntu "$LUA_DIR"

# Test nginx config
log "Test nginx configuration..."
if sudo /usr/local/openresty/nginx/sbin/nginx -t; then
    log "✅ Nginx configuration is valid"
else
    error "❌ Nginx configuration is invalid"
    exit 1
fi

# Reload OpenResty
log "Reload OpenResty..."
sudo systemctl reload openresty

log "✅ Lua scripts setup completed!"
log "Lua scripts location: $LUA_DIR"
ls -la "$LUA_DIR" 