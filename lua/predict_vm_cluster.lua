-- predict_vm_cluster.lua - Endpoint dự đoán VM cluster với caching
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

-- Validate request method
if ngx.req.get_method() ~= "POST" then
    ngx.status = 405
    ngx.say(cjson.encode({error = "Method not allowed"}))
    return
end

-- Get request body
ngx.req.read_body()
local body = ngx.req.get_body_data()

if not body then
    ngx.status = 400
    ngx.say(cjson.encode({error = "No request body"}))
    return
end

-- Parse JSON
local data = cjson.decode(body)
if not data or not data.vm_features then
    ngx.status = 400
    ngx.say(cjson.encode({error = "Missing 'vm_features' field"}))
    return
end

-- Create cache key from VM features
local cache_key = "vm_cluster:" .. cjson.encode(data.vm_features)
local cache_ttl = 300  -- 5 minutes

-- Check cache first
local cached_result = cache:get(cache_key)
if cached_result then
    ngx.header.content_type = "application/json"
    ngx.say(cached_result)
    return
end

-- Call ML service
local client = http.new()
local res, err = client:request_uri("http://127.0.0.1:5000/predict/vm_cluster", {
    method = "POST",
    body = body,
    headers = {
        ["Content-Type"] = "application/json",
        ["Connection"] = "keep-alive"
    },
    keepalive = 10000,
    keepalive_timeout = 60,
    keepalive_pool = 10
})

if not res then
    ngx.status = 500
    ngx.say(cjson.encode({error = "Failed to connect to ML service: " .. (err or "unknown error")}))
    return
end

if res.status ~= 200 then
    ngx.status = res.status
    ngx.say(res.body)
    return
end

-- Cache the result
cache:set(cache_key, res.body, cache_ttl)

-- Return response
ngx.header.content_type = "application/json"
ngx.say(res.body) 