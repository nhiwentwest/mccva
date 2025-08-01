-- mccva_routing.lua - MCCVA Algorithm Implementation

local http = require "resty.http"
local cjson = require "cjson.safe"
local cache = ngx.shared.ml_cache

-- CORS headers
ngx.header["Access-Control-Allow-Origin"] = "*"
ngx.header["Access-Control-Allow-Methods"] = "POST, OPTIONS"
ngx.header["Access-Control-Allow-Headers"] = "Content-Type"

-- Handle preflight requests
if ngx.req.get_method() == "OPTIONS" then
    ngx.status = 204
    ngx.exit(204)
end

-- MCCVA Server Mapping Configuration
-- Phân phối yêu cầu tới máy ảo phù hợp dựa trên AI prediction
local mccva_server_mapping = {
    -- Makespan-based routing (SVM Classification)
    ["small"] = "http://mccva-vm-small:8081",
    ["medium"] = "http://mccva-vm-medium:8083",
    ["large"] = "http://mccva-vm-large:8085",
}

-- Log function
local function log(level, message)
    ngx.log(level, "MCCVA: ", message)
end

-- Main prediction function
local function get_prediction(workload)
    -- Create HTTP client
    local httpc = http.new()
    httpc:set_timeout(5000)  -- 5 second timeout
    
    -- Debug log the request
    log(ngx.INFO, "Sending prediction request to ML service with payload: " .. cjson.encode(workload))
    
    -- Call ML service for prediction
    local res, err = httpc:request_uri("http://mccva-ml-service:5000/predict/mccva_complete", {
        method = "POST",
        body = cjson.encode(workload),
        headers = {
            ["Content-Type"] = "application/json",
        }
    })
    
    -- Handle errors
    if not res then
        log(ngx.ERR, "Failed to call ML service: " .. (err or "unknown error"))
        return nil, "ML service connection failed"
    end
    
    if res.status ~= 200 then
        log(ngx.ERR, "ML service returned non-200 status: " .. res.status)
        log(ngx.ERR, "Response body: " .. (res.body or "empty"))
        
        -- Special handling for 403 Forbidden errors
        if res.status == 403 then
            log(ngx.WARN, "ML service returned 403 Forbidden, possibly an authentication issue")
            return nil, "ML service authentication failed (403 Forbidden)"
        end
        
        return nil, "ML service error: " .. res.status
    end
    
    -- Debug log the response
    log(ngx.INFO, "ML service response: " .. res.body)
    
    -- Parse response
    local prediction, err = cjson.decode(res.body)
    if not prediction then
        log(ngx.ERR, "Failed to parse ML response: " .. (err or "unknown error"))
        log(ngx.ERR, "Raw response body: " .. (res.body or "empty"))
        return nil, "Invalid ML service response: " .. (err or "JSON parse error")
    end
    
    -- Validate response structure
    if not prediction.makespan then
        log(ngx.ERR, "ML response missing required 'makespan' field")
        log(ngx.ERR, "Raw response: " .. (res.body or "empty"))
        return nil, "Invalid ML service response: missing makespan field"
    end
    
    return prediction, nil
end

-- Main request handler
if ngx.req.get_method() == "POST" then
    -- Read request body
    ngx.req.read_body()
    local body = ngx.req.get_body_data()
    
    if not body then
        ngx.status = 400
        ngx.say(cjson.encode({error = "Missing request body"}))
        return ngx.exit(400)
    end
    
    -- Parse JSON body
    local workload, err = cjson.decode(body)
    if not workload then
        ngx.status = 400
        ngx.say(cjson.encode({error = "Invalid JSON: " .. (err or "unknown error")}))
        return ngx.exit(400)
    end
    
    -- Get prediction from ML service
    local prediction, err = get_prediction(workload)
    if not prediction then
        -- Fallback mechanism when ML service fails
        log(ngx.WARN, "ML service failed, using fallback mechanism: " .. (err or "unknown error"))
        
        -- Enhanced fallback mechanism
        local cpu = workload.cpu_cores or 4
        local memory = workload.memory_gb or 8
        local storage = workload.storage_gb or 100
        local network = workload.network_bandwidth or 1000
        local priority = workload.priority or 3
        
        -- Error-aware fallback strategy
        local vm_type = "medium"  -- Default to medium
        local confidence = 0.6    -- Default confidence
        local fallback_reason = err or "ML service error"
        
        -- Specialized fallback based on error type
        if err and string.find(err, "403 Forbidden") then
            log(ngx.WARN, "Using authentication failure fallback strategy")
            -- On authentication failure, we use a more conservative approach
            if priority >= 4 then
                vm_type = "large"
                confidence = 0.7
            elseif cpu <= 2 and memory <= 4 then
                vm_type = "small"
                confidence = 0.7
            end
        else
            -- Standard fallback for other errors
            if cpu <= 2 and memory <= 4 and storage <= 50 then
                vm_type = "small"
            elseif cpu >= 8 or memory >= 16 or storage >= 500 or priority >= 5 then
                vm_type = "large"
            end
        end
        
        prediction = {
            makespan = vm_type,
            confidence = confidence,
            method = "fallback_rule_based",
            stage_results = {
                fallback = true,
                reason = err or "ML service error"
            }
        }
        
        log(ngx.WARN, "Using fallback prediction: " .. vm_type .. " with confidence " .. confidence)
    end
    
    -- Get target VM based on prediction
    local vm_type = prediction.makespan or "small"  -- Default to small if no prediction
    
    -- Debug log the VM type
    log(ngx.INFO, "Selected VM type: " .. vm_type)
    
    local target_server = mccva_server_mapping[vm_type]
    
    if not target_server then
        log(ngx.ERR, "Invalid VM type returned: " .. vm_type)
        ngx.status = 500
        ngx.say(cjson.encode({error = "Invalid VM type: " .. vm_type}))
        return ngx.exit(500)
    end
    
    -- Forward request to target VM
    local httpc = http.new()
    httpc:set_timeout(10000)  -- 10 second timeout
    
    log(ngx.INFO, "Forwarding request to target VM: " .. target_server)
    
    local res, err = httpc:request_uri(target_server .. "/process", {
                method = "POST",
                body = body,
                headers = {
                    ["Content-Type"] = "application/json",
            ["X-MCCVA-Method"] = "mccva_complete",
            ["X-Makespan"] = vm_type,
            ["X-Confidence"] = prediction.confidence or "0",
        }
    })
    
    -- Handle errors
    if not res then
        log(ngx.ERR, "Failed to connect to VM: " .. (err or "unknown error"))
        ngx.status = 502
        ngx.say(cjson.encode({error = "Failed to connect to VM: " .. (err or "unknown error")}))
        return ngx.exit(502)
    end
    
    -- Debug log the VM response
    log(ngx.INFO, "VM response status: " .. res.status)
    log(ngx.INFO, "VM response body: " .. (res.body or "empty"))
    
    -- Return response from VM
    ngx.status = res.status
    
    -- Add MCCVA metadata headers
    ngx.header["X-MCCVA-VM-Type"] = vm_type
    ngx.header["X-MCCVA-Confidence"] = prediction.confidence or "unknown"
    
    -- Log successful routing
    log(ngx.INFO, "Successfully routed request to " .. vm_type .. " VM with confidence " .. (prediction.confidence or "unknown"))
    
    -- Return VM response
    ngx.say(res.body)
    
else
    -- Method not allowed
    ngx.status = 405
    ngx.say(cjson.encode({error = "Method not allowed"}))
    ngx.exit(405)
end 