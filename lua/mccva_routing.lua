-- mccva_routing.lua - MCCVA Algorithm Implementation
-- Makespan Classification & Clustering VM Algorithm
-- Triển khai thuật toán MCCVA với OpenResty + LuaJIT
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
    makespan = {
        small = {
            primary = "http://127.0.0.1:8081",  -- VM tải thấp
            backup = "http://127.0.0.1:8082",   -- VM tải thấp backup
            weight = 0.7  -- 70% traffic to primary
        },
        medium = {
            primary = "http://127.0.0.1:8083",  -- VM tải trung bình
            backup = "http://127.0.0.1:8084",   -- VM tải trung bình backup
            weight = 0.6  -- 60% traffic to primary
        },
        large = {
            primary = "http://127.0.0.1:8085",  -- VM tải cao
            backup = "http://127.0.0.1:8086",   -- VM tải cao backup
            weight = 0.8  -- 80% traffic to primary
        }
    },
    
    -- Cluster-based routing (K-Means Clustering)
    cluster = {
        [0] = {  -- Low resource usage cluster (K-Means predicted)
            primary = "http://127.0.0.1:8081",  -- VM tải thấp
            backup = "http://127.0.0.1:8082",   -- VM tải thấp backup
            weight = 0.9
        },
        [1] = {  -- Medium resource usage cluster (K-Means predicted)
            primary = "http://127.0.0.1:8083",  -- VM tải trung bình
            backup = "http://127.0.0.1:8084",   -- VM tải trung bình backup
            weight = 0.7
        },
        [2] = {  -- High resource usage cluster (K-Means predicted)
            primary = "http://127.0.0.1:8085",  -- VM tải cao
            backup = "http://127.0.0.1:8086",   -- VM tải cao backup
            weight = 0.5  -- More load balancing for high usage
        },
        [3] = {  -- Balanced cluster (K-Means predicted)
            primary = "http://127.0.0.1:8087",  -- VM cân bằng
            backup = "http://127.0.0.1:8088",   -- VM cân bằng backup
            weight = 0.6
        },
        [4] = {  -- CPU intensive cluster (K-Means predicted)
            primary = "http://127.0.0.1:8085",  -- VM tải cao
            backup = "http://127.0.0.1:8086",   -- VM tải cao backup
            weight = 0.8
        },
        [5] = {  -- Storage intensive cluster (K-Means predicted)
            primary = "http://127.0.0.1:8083",  -- VM tải trung bình
            backup = "http://127.0.0.1:8084",   -- VM tải trung bình backup
            weight = 0.7
        }
    }
}

-- MCCVA Algorithm: Chọn VM dựa trên AI prediction
local function mccva_select_vm(makespan, cluster, confidence, vm_features)
    local selected_vm = nil
    local routing_info = {}
    
    -- Priority 1: High confidence makespan routing (SVM-based)
    if confidence and confidence > 2.0 then
        local makespan_config = mccva_server_mapping.makespan[makespan]
        if makespan_config then
            -- Weighted random selection based on SVM confidence
            local rand = math.random()
            local adjusted_weight = makespan_config.weight
            
            -- Adjust weight based on SVM confidence
            if confidence > 3.0 then
                adjusted_weight = adjusted_weight + 0.1  -- Higher confidence = more primary
            elseif confidence < 2.5 then
                adjusted_weight = adjusted_weight - 0.1  -- Lower confidence = more backup
            end
            
            if rand <= adjusted_weight then
                selected_vm = makespan_config.primary
                routing_info.method = "mccva_svm_primary"
            else
                selected_vm = makespan_config.backup
                routing_info.method = "mccva_svm_backup"
            end
            routing_info.confidence = confidence
            routing_info.makespan = makespan
            routing_info.algorithm = "SVM Classification"
        end
    end
    
    -- Priority 2: Cluster-based routing (K-Means-based fallback)
    if not selected_vm then
        local cluster_config = mccva_server_mapping.cluster[cluster]
        if cluster_config then
            local rand = math.random()
            local adjusted_weight = cluster_config.weight
            
            -- Adjust weight based on VM features (K-Means insights)
            if vm_features then
                local cpu_usage = vm_features[1] or 0
                local ram_usage = vm_features[2] or 0
                local storage_usage = vm_features[3] or 0
                
                -- K-Means-based weight adjustment
                if cpu_usage > 0.8 then
                    adjusted_weight = adjusted_weight + 0.1  -- High CPU = more primary
                elseif ram_usage > 0.8 then
                    adjusted_weight = adjusted_weight + 0.05  -- High RAM = slight preference
                elseif storage_usage > 0.8 then
                    adjusted_weight = adjusted_weight - 0.05  -- High storage = slight backup preference
                end
            end
            
            if rand <= adjusted_weight then
                selected_vm = cluster_config.primary
                routing_info.method = "mccva_kmeans_primary"
            else
                selected_vm = cluster_config.backup
                routing_info.method = "mccva_kmeans_backup"
            end
            routing_info.confidence = confidence or 0
            routing_info.cluster = cluster
            routing_info.algorithm = "K-Means Clustering"
        end
    end
    
    -- Priority 3: MCCVA ensemble decision (combine SVM + K-Means)
    if not selected_vm then
        -- Use both algorithms to make final decision
        local ensemble_score = 0
        
        if makespan == "small" then
            ensemble_score = ensemble_score + 1
        elseif makespan == "medium" then
            ensemble_score = ensemble_score + 2
        elseif makespan == "large" then
            ensemble_score = ensemble_score + 3
        end
        
        if cluster >= 0 and cluster <= 2 then
            ensemble_score = ensemble_score + 1  -- Low/medium clusters
        elseif cluster >= 3 and cluster <= 5 then
            ensemble_score = ensemble_score + 2  -- High clusters
        end
        
        -- MCCVA ensemble decision
        if ensemble_score <= 2 then
            selected_vm = "http://127.0.0.1:8081"  -- VM tải thấp
            routing_info.method = "mccva_ensemble_low"
        elseif ensemble_score <= 4 then
            selected_vm = "http://127.0.0.1:8083"  -- VM tải trung bình
            routing_info.method = "mccva_ensemble_medium"
        else
            selected_vm = "http://127.0.0.1:8085"  -- VM tải cao
            routing_info.method = "mccva_ensemble_high"
        end
        
        routing_info.confidence = confidence or 0
        routing_info.ensemble_score = ensemble_score
        routing_info.algorithm = "MCCVA Ensemble (SVM + K-Means)"
    end
    
    return selected_vm, routing_info
end

-- Main MCCVA Algorithm Implementation
if ngx.req.get_method() == "POST" then
    ngx.req.read_body()
    local body = ngx.req.get_body_data()
    
    if body then
        local data = cjson.decode(body)
        
        -- Step 1: Enhanced AI Prediction - Dùng ensemble learning
        local features = {
            data.cpu_cores or 4,
            data.memory or 8,
            data.storage or 100,
            data.network_bandwidth or 1000,
            data.priority or 3,
            data.task_complexity or 2,    -- New feature: task complexity (1-5)
            data.data_size or 50,         -- New feature: data size in GB
            data.io_intensity or 2,       -- New feature: I/O intensity (1-5)
            data.parallel_degree or 1,    -- New feature: parallel processing degree
            data.deadline_urgency or 3    -- New feature: deadline urgency (1-5)
        }
        
        local vm_features = data.vm_features or {0.5, 0.5, 0.5}  -- Default VM features
        
        local enhanced_request = {
            features = features,
            vm_features = vm_features
        }
        
        -- Dùng enhanced prediction endpoint
        local enhanced_response, err = http.new():request_uri("http://localhost:5000/predict/enhanced", {
            method = "POST",
            body = cjson.encode(enhanced_request),
            headers = { ["Content-Type"] = "application/json" }
        })
        
        local makespan = "medium"  -- default
        local cluster = 0  -- default
        local confidence = 0
        
        if enhanced_response and enhanced_response.status == 200 then
            local enhanced_result = cjson.decode(enhanced_response.body)
            makespan = enhanced_result.makespan
            cluster = enhanced_result.cluster
            confidence = enhanced_result.confidence
            
            -- Log enhanced prediction details
            ngx.log(ngx.INFO, "Enhanced prediction: makespan=" .. makespan .. 
                   ", cluster=" .. cluster .. ", confidence=" .. confidence)
        else
            -- Fallback to basic prediction if enhanced fails
            local ml_request = {
                features = features
            }
            
            local makespan_response, err = http.new():request_uri("http://localhost:5000/predict/makespan", {
                method = "POST",
                body = cjson.encode(ml_request),
                headers = { ["Content-Type"] = "application/json" }
            })
            
            if makespan_response and makespan_response.status == 200 then
                local makespan_result = cjson.decode(makespan_response.body)
                makespan = makespan_result.makespan
                confidence = makespan_result.confidence
            end
        end
        
        -- Step 2: MCCVA Algorithm - Chọn VM tối ưu
        local target_vm, routing_info = mccva_select_vm(makespan, cluster, confidence, vm_features)
        
        -- Step 3: Forward request to selected VM with retry/fallback
        local tried_vms = {}
        local function try_forward(vm_url, method_label)
            local client = http.new()
            local res, err = client:request_uri(vm_url, {
                method = "POST",
                body = body,
                headers = {
                    ["Content-Type"] = "application/json",
                    ["X-MCCVA-Method"] = routing_info.method .. (method_label or ""),
                    ["X-Makespan"] = makespan,
                    ["X-Cluster"] = tostring(cluster),
                    ["X-Confidence"] = tostring(confidence),
                    ["X-Algorithm"] = routing_info.algorithm or "unknown"
                }
            })
            table.insert(tried_vms, {vm=vm_url, err=err})
            return res, err
        end

        local res, err = try_forward(target_vm, "")
        
        -- Success case: Return response with prediction data
        if res and res.status < 500 then
            local response_data = nil
            if res.body then
                local ok, parsed = pcall(cjson.decode, res.body)
                if ok then
                    response_data = parsed
                else
                    ngx.log(ngx.ERR, "Failed to parse response JSON: " .. (res.body or "empty"))
                    response_data = {processed = true, error = "Invalid JSON response"}
                end
            else
                response_data = {processed = true, error = "Empty response"}
            end
            
            response_data.routing_info = routing_info
            response_data.target_vm = target_vm
            response_data.prediction = {
                makespan = makespan,
                confidence = confidence,
                features = features,
                timestamp = os.date("%Y-%m-%dT%H:%M:%S")
            }
            response_data.mccva_decision = {
                makespan_prediction = makespan,
                cluster_prediction = cluster,
                confidence_score = confidence,
                algorithm_used = routing_info.algorithm or "unknown",
                ensemble_score = routing_info.ensemble_score or 0
            }
            response_data.server = target_vm:gsub("http://127.0.0.1:", "server_")
            ngx.header.content_type = "application/json"
            ngx.say(cjson.encode(response_data))
            return
        end
        
        -- Nếu lỗi, thử backup (nếu có)
        if (not res or res.status >= 500) and routing_info then
            local backup_vm = nil
            -- Tìm backup VM từ mapping
            for _, mapping in pairs(mccva_server_mapping) do
                for k, v in pairs(mapping) do
                    if v.primary == target_vm and v.backup then
                        backup_vm = v.backup
                    end
                end
            end
            if backup_vm and backup_vm ~= target_vm then
                local res2, err2 = try_forward(backup_vm, "_backup")
                if res2 then
                    local response_data = nil
                    if res2.body then
                        local ok, parsed = pcall(cjson.decode, res2.body)
                        if ok then
                            response_data = parsed
                        else
                            ngx.log(ngx.ERR, "Failed to parse backup response JSON: " .. (res2.body or "empty"))
                            response_data = {processed = true, error = "Invalid JSON response"}
                        end
                    else
                        response_data = {processed = true, error = "Empty response"}
                    end
                    
                    response_data.routing_info = routing_info
                    response_data.target_vm = backup_vm
                    response_data.mccva_decision = {
                        makespan_prediction = makespan,
                        cluster_prediction = cluster,
                        confidence_score = confidence,
                        algorithm_used = routing_info.algorithm or "unknown",
                        ensemble_score = routing_info.ensemble_score or 0
                    }
                    response_data.retry = true
                    response_data.tried_vms = tried_vms
                    ngx.header.content_type = "application/json"
                    ngx.say(cjson.encode(response_data))
                    return
                else
                    err = err2 or err
                end
            end
        end
        -- Nếu vẫn lỗi, trả về lỗi cuối cùng và log lại các VM đã thử
        ngx.status = 500
        ngx.say(cjson.encode({error = "MCCVA routing failed: " .. (err or "unknown error"), tried_vms = tried_vms}))
    else
        ngx.status = 400
        ngx.say(cjson.encode({error = "No request body"}))
    end
else
    ngx.status = 405
    ngx.say(cjson.encode({error = "Method not allowed"}))
end 