function [scene, latency] = describeSceneVLM(rgbFrame, modelName)
%DESCRIBESCENEVLM Ask a local Ollama VLM what is in the drone's flight path.
%   Vision-only questions the depth map can't answer: what the obstacle is and
%   how much caution it needs. Sector choice stays deterministic in the flight code.
%   scene fields: ok, obstacle, inPath, hazard ("low"|"medium"|"high"), description.
    arguments
        rgbFrame
        modelName (1,1) string
    end

    scene = struct("ok", false, "obstacle", "unknown", "inPath", false, ...
                   "hazard", "low", "description", "");

    tmpFile = [tempname, '.jpg'];
    imwrite(imresize(rgbFrame, 0.5), tmpFile, 'Quality', 80);
    fid = fopen(tmpFile, 'rb');
    rawBytes = fread(fid, inf, '*uint8');
    fclose(fid);
    delete(tmpFile);
    b64Image = matlab.net.base64encode(rawBytes);

    prompt = [ ...
        'You are the vision system of a drone flying forward at 12 m altitude along a city street. ', ...
        'Look at this forward camera image. Identify the most important object in the drone''s flight path ', ...
        '(the middle of the image, at and above the horizon) that it could collide with or should be careful around. ', ...
        'Buildings lining the sides of the street are normal and are NOT in the path unless one is directly ahead. ', ...
        'Respond ONLY with valid JSON: {"obstacle": "none|building|vehicle|construction|pole|tree|wire|person|unknown_object", ', ...
        '"in_path": true or false, "hazard": "low|medium|high", "description": "<at most 10 words>"}'];

    queryTimer = tic;
    try
        payload = struct("model", modelName, ...
                         "prompt", prompt, ...
                         "images", {{b64Image}}, ...
                         "format", "json", ...
                         "stream", false);
        opts = weboptions("MediaType", "application/json", "Timeout", 30);
        resp = webwrite("http://localhost:11434/api/generate", payload, opts);
        parsed = jsondecode(resp.response);

        if isfield(parsed, "obstacle"),    scene.obstacle    = lower(string(parsed.obstacle)); end
        if isfield(parsed, "description"), scene.description = string(parsed.description); end
        if isfield(parsed, "in_path")
            v = parsed.in_path;
            if ischar(v) || isstring(v)
                v = any(strcmpi(string(v), ["true" "yes" "1"])); % Small VLMs often quote booleans
            end
            scene.inPath = logical(v);
        end
        if isfield(parsed, "hazard")
            h = lower(string(parsed.hazard));
            if any(h == ["low" "medium" "high"])
                scene.hazard = h;
            end
        end
        scene.ok = true;
    catch ME
        fprintf("Scene query failed: %s\n", ME.message);
    end
    latency = toc(queryTimer);
end
