function [scene, latency] = describeSceneVLM(rgbFrame, modelName, variant)
%DESCRIBESCENEVLM Ask a local Ollama VLM what is in the drone's flight path.
%   Vision-only questions the depth map can't answer: is there something in the
%   street that isn't part of the mapped city, and what is it. Sector choice stays
%   deterministic in the flight code.
%
%   variant: "describe"     open-ended obstacle / in_path / hazard / description
%            "yesno"        narrow yes/no: any object in the street that isn't a building,
%                           road, sidewalk, tree or streetlight
%            "yesno_center" yesno restricted to the middle of the image, with the rule
%                           that a large object standing in the roadway is not a building
%
%   scene fields: ok, flag (non-map obstacle in the flight path), obstacle, inPath,
%   hazard ("low"|"medium"|"high"), description.
    arguments
        rgbFrame
        modelName (1,1) string
        variant (1,1) string {mustBeMember(variant, ["describe" "yesno" "yesno_center"])} = "describe"
    end

    scene = struct("ok", false, "flag", false, "obstacle", "unknown", "inPath", false, ...
                   "hazard", "low", "description", "");

    tmpFile = [tempname, '.jpg'];
    imwrite(imresize(rgbFrame, 0.5), tmpFile, 'Quality', 80);
    fid = fopen(tmpFile, 'rb');
    rawBytes = fread(fid, inf, '*uint8');
    fclose(fid);
    delete(tmpFile);
    b64Image = matlab.net.base64encode(rawBytes);

    switch variant
        case "describe"
            prompt = [ ...
                'You are the vision system of a drone flying forward at 12 m altitude along a city street. ', ...
                'Look at this forward camera image. Identify the most important object in the drone''s flight path ', ...
                '(the middle of the image, at and above the horizon) that it could collide with or should be careful around. ', ...
                'Buildings lining the sides of the street are normal and are NOT in the path unless one is directly ahead. ', ...
                'Respond ONLY with valid JSON: {"obstacle": "none|building|vehicle|construction|pole|tree|wire|person|unknown_object", ', ...
                '"in_path": true or false, "hazard": "low|medium|high", "description": "<at most 10 words>"}'];
        case "yesno"
            prompt = [ ...
                'This is the forward camera of a drone flying along a city street. ', ...
                'Is there any object in the street ahead that is NOT a building, road, sidewalk, tree, traffic light or streetlight? ', ...
                'Examples: a vehicle, barrier, box, construction equipment, person or debris. ', ...
                'Respond ONLY with valid JSON: {"object_in_street": true or false, "what": "<2-5 words, or none>"}'];
        case "yesno_center"
            prompt = [ ...
                'This is the forward camera of a drone flying along a city street. Look only at the middle third of the image, ', ...
                'where the street continues ahead. Buildings stand on the left and right sides of a street; ', ...
                'a large object standing in the middle of the roadway is not a building. ', ...
                'Is there any object in the roadway ahead that is NOT a building, road, sidewalk, tree, traffic light or streetlight? ', ...
                'Respond ONLY with valid JSON: {"object_in_street": true or false, "what": "<2-5 words, or none>"}'];
    end

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

        if variant == "describe"
            if isfield(parsed, "obstacle"),    scene.obstacle    = lower(string(parsed.obstacle)); end
            if isfield(parsed, "description"), scene.description = string(parsed.description); end
            if isfield(parsed, "in_path"),     scene.inPath      = parseBool(parsed.in_path); end
            if isfield(parsed, "hazard")
                h = lower(string(parsed.hazard));
                if any(h == ["low" "medium" "high"])
                    scene.hazard = h;
                end
            end
            % Buildings are already covered by the map and depth reflex; flag only the rest
            scene.flag = scene.inPath && scene.hazard ~= "low" && ~any(scene.obstacle == ["none" "building"]);
        else
            if isfield(parsed, "object_in_street"), scene.flag = parseBool(parsed.object_in_street); end
            if isfield(parsed, "what"),             scene.description = string(parsed.what); end
            scene.inPath   = scene.flag;
            scene.obstacle = ternary(scene.flag, "street_object", "none");
            scene.hazard   = ternary(scene.flag, "medium", "low");
        end
        scene.ok = true;
    catch ME
        fprintf("Scene query failed: %s\n", ME.message);
    end
    latency = toc(queryTimer);
end

function b = parseBool(v)
    if ischar(v) || isstring(v)
        b = any(strcmpi(string(v), ["true" "yes" "1"])); % Small VLMs often quote booleans
    else
        b = logical(v);
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
