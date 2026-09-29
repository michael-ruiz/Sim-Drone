function [sector, reason, latency] = queryVLMNavigator(rgbFrame, sectorDepths, bearingDeg, distToTarget, targetLabel, modelName, useOverlay)
%QUERYVLMNAVIGATOR Ask a local Ollama VLM which of 5 camera sectors to fly toward.
%   Returns sector 0-5 (0 = all blocked), the model's short reason, and the query
%   latency in seconds. Falls back to a depth + bearing scorer if Ollama fails.
%   useOverlay draws the sector boundaries and numbers into the image the VLM sees.
    arguments
        rgbFrame
        sectorDepths (1,5) double
        bearingDeg (1,1) double
        distToTarget (1,1) double
        targetLabel (1,1) string
        modelName (1,1) string
        useOverlay (1,1) logical = false
    end

    img = rgbFrame;
    if useOverlay
        img = drawSectorOverlay(img, round(linspace(40, 600, 6)));
    end

    % Encode frame as Base64 JPEG for the VLM API
    tmpFile = [tempname, '.jpg'];
    imwrite(imresize(img, 0.5), tmpFile, 'Quality', 80);
    fid = fopen(tmpFile, 'rb');
    rawBytes = fread(fid, inf, '*uint8');
    fclose(fid);
    delete(tmpFile);
    b64Image = matlab.net.base64encode(rawBytes);

    overlayNote = '';
    if useOverlay
        overlayNote = 'Yellow vertical lines in the image divide it into these 5 sectors, each labeled with its number at the top. ';
    end
    prompt = sprintf([ ...
        'You are an autonomous urban drone navigator flying at 12 m altitude. The camera view is divided from left to right ', ...
        'into 5 sectors: 1 (Far Left, -28 deg), 2 (Left, -14 deg), 3 (Center, 0 deg), 4 (Right, +14 deg), 5 (Far Right, +28 deg). ', ...
        '%s', ...
        'Measured depth clearance in meters for sectors [1..5] is [%.0f, %.0f, %.0f, %.0f, %.0f] (45 means 45 m or more). ', ...
        'Your %s is %.1f meters away at relative bearing %.0f degrees (+ is Right, - is Left). ', ...
        'Pick the open sector (1-5) that avoids buildings and is closest to that bearing. ', ...
        'If every sector is blocked by a building or wall closer than 8 m, answer sector 0. ', ...
        'Respond ONLY with valid JSON: {"sector": <int 0-5>, "reason": "<short 6-word explanation>"}'], ...
        overlayNote, sectorDepths(1), sectorDepths(2), sectorDepths(3), sectorDepths(4), sectorDepths(5), ...
        targetLabel, distToTarget, bearingDeg);

    queryTimer = tic;
    try
        % Call local Ollama server (http://localhost:11434/api/generate)
        payload = struct("model", modelName, ...
                         "prompt", prompt, ...
                         "images", {{b64Image}}, ...
                         "format", "json", ...
                         "stream", false);
        opts = weboptions("MediaType", "application/json", "Timeout", 30);
        resp = webwrite("http://localhost:11434/api/generate", payload, opts);
        parsed = jsondecode(resp.response);
        sec = parsed.sector;
        if ischar(sec) || isstring(sec)
            sec = str2double(sec); % Small VLMs often return "3" instead of 3
        end
        if ~isscalar(sec) || isnan(sec)
            error("VLM returned invalid sector: %s", resp.response);
        end
        sector = max(0, min(5, round(double(sec))));
        reason = string(parsed.reason);
    catch ME
        fprintf("VLM query failed, using fallback: %s\n", ME.message);
        % Fallback if Ollama is unavailable: score sectors by depth clearance + bearing alignment
        if max(sectorDepths) < 8
            sector = 0;
            reason = "All sectors blocked (Local Fallback)";
        else
            sectorAnglesDeg = [-28, -14, 0, 14, 28];
            scores = sectorDepths - 0.35 * abs(sectorAnglesDeg - bearingDeg);
            [~, sector] = max(scores);
            reason = "Open corridor toward target (Local Fallback)";
        end
    end
    latency = toc(queryTimer);
end

function img = drawSectorOverlay(img, colEdges)
    % Yellow boundary lines plus a black label box with a bitmap digit per sector
    yellow = uint8([255 220 0]);
    for e = colEdges
        cols = max(1, e - 1):min(size(img, 2), e + 1);
        for c = 1:3
            img(:, cols, c) = yellow(c);
        end
    end

    glyphs = digitGlyphs();
    scale = 5;
    for k = 1:5
        g = logical(kron(glyphs{k}, ones(scale)));
        [gh, gw] = size(g);
        r0 = 10;
        c0 = round(mean(colEdges(k:k+1))) - floor(gw / 2);
        img(r0-5:r0+gh+4, c0-5:c0+gw+4, :) = 0;
        for c = 1:3
            ch = img(r0:r0+gh-1, c0:c0+gw-1, c);
            ch(g) = yellow(c);
            img(r0:r0+gh-1, c0:c0+gw-1, c) = ch;
        end
    end
end

function glyphs = digitGlyphs()
    % 5x7 bitmap font for digits 1-5
    rows = { ...
        ["00100" "01100" "00100" "00100" "00100" "00100" "01110"], ...
        ["01110" "10001" "00001" "00010" "00100" "01000" "11111"], ...
        ["11110" "00001" "00001" "01110" "00001" "00001" "11110"], ...
        ["00010" "00110" "01010" "10010" "11111" "00010" "00010"], ...
        ["11111" "10000" "11110" "00001" "00001" "10001" "01110"]};
    glyphs = cellfun(@(r) char(r') == '1', rows, UniformOutput=false);
end
