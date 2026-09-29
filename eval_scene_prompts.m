%% Offline Scene-Prompt Benchmark
% Replays logged frames (vlm_log/*) through describeSceneVLM prompt variants and scores
% the "non-map obstacle in path" flag against geometric ground truth: the unmapped box's
% footprint projected into each frame's camera using the logged position and heading.
%   positive  = box 2-40 m ahead and overlapping the middle +/-10 deg of the view
%   negative  = box not visible at all (behind, or outside the +/-35 deg field of view)
%   side      = box visible but off to the side (reported separately; flagging is debatable)
clear;

variants = ["describe" "yesno" "yesno_center"];
model    = "qwen2.5vl:3b";
if ~isempty(getenv("VLM_MODEL")), model = string(getenv("VLM_MODEL")); end

box      = [40 2 4 8];                               % [x y sizeX sizeY] of the unmapped obstacle
route    = [75 6; 80 110];                           % Route waypoints in the logged runs
halfFov  = atand(320 / 450);                         % ~35 deg

%% Gather frames with position + heading
frames = struct("file", {}, "pos", {}, "yaw", {});
runs = dir("vlm_log");
runs = runs([runs.isdir] & ~startsWith({runs.name}, "."));
for r = 1:numel(runs)
    runDir = fullfile(runs(r).folder, runs(r).name);
    if isfile(fullfile(runDir, "scenes.mat"))
        S = load(fullfile(runDir, "scenes.mat"), "sceneLog");
        for k = 1:numel(S.sceneLog)
            f = fullfile(runDir, sprintf("s%03d.png", k));
            if isfile(f)
                frames(end+1) = struct("file", f, "pos", S.sceneLog(k).pos(:)', "yaw", S.sceneLog(k).yaw); %#ok<SAGROW>
            end
        end
    end
    if isfile(fullfile(runDir, "queries.mat"))
        Q = load(fullfile(runDir, "queries.mat"), "queryLog");
        for k = 1:numel(Q.queryLog)
            f = fullfile(runDir, sprintf("q%03d.png", k));
            ql = Q.queryLog(k);
            % Planner logs have bearing to the current waypoint, not yaw: find the waypoint
            % whose distance matches, then yaw = heading to it - bearing
            [err, w] = min(abs(vecnorm(route - ql.pos(:)', 2, 2) - ql.dist));
            if isfile(f) && err < 0.5
                d = route(w, :) - ql.pos(:)';
                frames(end+1) = struct("file", f, "pos", ql.pos(:)', ...
                                       "yaw", atan2(d(2), d(1)) - deg2rad(ql.bearing)); %#ok<SAGROW>
            end
        end
    end
end
nF = numel(frames);

%% Ground truth from the box footprint
[gx, gy] = ndgrid(box(1) + (-box(3)/2:0.5:box(3)/2), box(2) + (-box(4)/2:0.5:box(4)/2));
pts = [gx(:), gy(:)];
label = strings(1, nF); boxDist = nan(1, nF);
for i = 1:nF
    d   = pts - frames(i).pos;
    fwd = d * [cos(frames(i).yaw); sin(frames(i).yaw)];
    rgt = d * [-sin(frames(i).yaw); cos(frames(i).yaw)];
    ang = atan2d(rgt, fwd);
    ahead = fwd > 2;
    if any(ahead & abs(ang) <= 10 & fwd <= 40)
        label(i) = "positive";
        boxDist(i) = min(fwd(ahead & abs(ang) <= 10));
    elseif any(ahead & abs(ang) <= halfFov & fwd <= 80)
        label(i) = "side";
    else
        label(i) = "negative";
    end
end
fprintf("%d frames: %d positive (box in path, %.0f-%.0f m), %d side, %d negative | model %s\n", nF, ...
        nnz(label == "positive"), min(boxDist), max(boxDist), nnz(label == "side"), nnz(label == "negative"), model);

%% Run every variant on every frame
nV = numel(variants);
flag = false(nV, nF); lat = zeros(nV, nF); ok = false(nV, nF); what = strings(nV, nF);
try % Load the model before timing
    webwrite("http://localhost:11434/api/generate", struct("model", model, "prompt", "", "stream", false), ...
             weboptions("MediaType", "application/json", "Timeout", 120));
catch ME
    fprintf("Warm-up failed: %s\n", ME.message);
end
for v = 1:nV
    for i = 1:nF
        [sc, lat(v, i)] = describeSceneVLM(imread(frames(i).file), model, variants(v));
        flag(v, i) = sc.flag; ok(v, i) = sc.ok;
        what(v, i) = sc.obstacle + ": " + sc.description;
    end
    fprintf("done: %s\n", variants(v));
end

%% Score
pos = label == "positive"; neg = label == "negative"; side = label == "side";
near = pos & boxDist <= 20;
fprintf("\n%-13s | %8s %12s %11s | %10s %9s | %7s %6s\n", "variant", "recall", "recall<=20m", "side flag", ...
        "false flag", "precision", "latency", "failed");
for v = 1:nV
    f = flag(v, :);
    fprintf("%-13s | %3d/%-3d  %6d/%-3d   %5d/%-3d  | %5d/%-3d  %8.0f%% | %6.2fs %6d\n", variants(v), ...
            nnz(f & pos), nnz(pos), nnz(f & near), nnz(near), nnz(f & side), nnz(side), ...
            nnz(f & neg), nnz(neg), 100 * nnz(f & pos) / max(1, nnz(f & ~side)), mean(lat(v, :)), nnz(~ok(v, :)));
end
fprintf("\nprecision = flags on positive frames / flags on positive + negative frames (side frames excluded)\n");

save(fullfile("vlm_log", "scene_benchmark.mat"), "variants", "model", "frames", "label", "boxDist", ...
     "flag", "lat", "ok", "what");
