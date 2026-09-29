%% Offline Scene-Prompt Benchmark
% Replays logged frames (vlm_log/*) through describeSceneVLM prompt variants and models,
% and scores the "non-map obstacle in path" flag against geometric ground truth: each
% run's unmapped hazards (obstacleCourse.m) projected into the frame's camera using the
% logged position and heading.
%   positive  = a hazard 2-40 m ahead and overlapping the middle +/-10 deg of the view
%   negative  = no hazard visible at all (behind, or outside the +/-35 deg field of view)
%   side      = a hazard visible but off to the side (reported separately; flagging is debatable)
% All positive and side frames are used; negatives are subsampled (fixed seed) to bound runtime.
% Overrides: VLM_SCENE_MODELS, VLM_SCENE_VARIANTS (space-separated), VLM_SCENE_NEG (count)
clear;

models   = ["qwen2.5vl:3b" "qwen2.5vl:7b"];
variants = ["describe" "yesno"];
maxNeg   = 60;
if ~isempty(getenv("VLM_SCENE_MODELS")),   models   = split(string(getenv("VLM_SCENE_MODELS")))'; end
if ~isempty(getenv("VLM_SCENE_VARIANTS")), variants = split(string(getenv("VLM_SCENE_VARIANTS")))'; end
if ~isempty(getenv("VLM_SCENE_NEG")),      maxNeg   = str2double(getenv("VLM_SCENE_NEG")); end

route   = [75 6; 80 110];                            % Route waypoints in the logged runs
halfFov = atand(320 / 450);                          % ~35 deg

%% Gather frames with position + heading + the run's obstacle course
frames = struct("file", {}, "pos", {}, "yaw", {}, "course", {}, "run", {});
runs = dir("vlm_log");
runs = runs([runs.isdir] & ~startsWith({runs.name}, "."));
for r = 1:numel(runs)
    runDir = fullfile(runs(r).folder, runs(r).name);
    courseName = "box";                              % Runs logged before obstacleName existed
    for f = ["scenes.mat" "queries.mat"]
        if isfile(fullfile(runDir, f)) && ismember("obstacleName", who("-file", fullfile(runDir, f)))
            S = load(fullfile(runDir, f), "obstacleName");
            courseName = string(S.obstacleName);
        end
    end
    if isfile(fullfile(runDir, "scenes.mat"))
        S = load(fullfile(runDir, "scenes.mat"), "sceneLog");
        for k = 1:numel(S.sceneLog)
            f = fullfile(runDir, sprintf("s%03d.png", k));
            if isfile(f)
                frames(end+1) = struct("file", f, "pos", S.sceneLog(k).pos(:)', "yaw", S.sceneLog(k).yaw, ...
                                       "course", courseName, "run", string(runs(r).name)); %#ok<SAGROW>
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
                                       "yaw", atan2(d(2), d(1)) - deg2rad(ql.bearing), ...
                                       "course", courseName, "run", string(runs(r).name)); %#ok<SAGROW>
            end
        end
    end
end

%% Ground truth from hazard footprints
courses = containers.Map();
label = strings(1, numel(frames)); hazDist = nan(1, numel(frames)); hazName = strings(1, numel(frames));
for i = 1:numel(frames)
    if ~isKey(courses, frames(i).course), courses(frames(i).course) = obstacleCourse(frames(i).course); end
    hz = courses(frames(i).course).hazards;
    label(i) = "negative";
    for h = 1:numel(hz)
        d   = hz(h).points - frames(i).pos;
        fwd = d * [cos(frames(i).yaw); sin(frames(i).yaw)];
        rgt = d * [-sin(frames(i).yaw); cos(frames(i).yaw)];
        ang = atan2d(rgt, fwd);
        ahead = fwd > 2;
        inPath = ahead & abs(ang) <= 10 & fwd <= 40;
        if any(inPath)
            if label(i) ~= "positive" || min(fwd(inPath)) < hazDist(i)
                hazDist(i) = min(fwd(inPath)); hazName(i) = hz(h).name;
            end
            label(i) = "positive";
        elseif label(i) == "negative" && any(ahead & abs(ang) <= halfFov & fwd <= 80)
            label(i) = "side";
        end
    end
end

% Keep all positive/side frames, subsample negatives with a fixed seed
rng(1);
negIdx = find(label == "negative");
negIdx = negIdx(randperm(numel(negIdx), min(maxNeg, numel(negIdx))));
keep = sort([find(label ~= "negative"), negIdx]);
frames = frames(keep); label = label(keep); hazDist = hazDist(keep); hazName = hazName(keep);
nF = numel(frames);
pos = label == "positive"; neg = label == "negative"; side = label == "side";
fprintf("%d frames: %d positive (%s), %d side, %d negative (sampled)\n", nF, nnz(pos), ...
        strjoin(compose("%s x%d", unique(hazName(pos))', arrayfun(@(n) nnz(hazName(pos) == n), unique(hazName(pos)))'), ", "), ...
        nnz(side), nnz(neg));

%% Run every model x variant on every frame
configs = struct("model", {}, "variant", {});
for m = models
    for v = variants
        configs(end+1) = struct("model", m, "variant", v); %#ok<SAGROW>
    end
end
nC = numel(configs);
flag = false(nC, nF); lat = zeros(nC, nF); ok = false(nC, nF); what = strings(nC, nF);
for c = 1:nC
    try % Load the model before timing
        webwrite("http://localhost:11434/api/generate", struct("model", configs(c).model, "prompt", "", "stream", false), ...
                 weboptions("MediaType", "application/json", "Timeout", 120));
    catch ME
        fprintf("Warm-up failed for %s: %s\n", configs(c).model, ME.message);
    end
    for i = 1:nF
        [sc, lat(c, i)] = describeSceneVLM(imread(frames(i).file), configs(c).model, configs(c).variant);
        flag(c, i) = sc.flag; ok(c, i) = sc.ok;
        what(c, i) = sc.obstacle + ": " + sc.description;
    end
    fprintf("done: %s / %s\n", configs(c).model, configs(c).variant);
end

%% Score (overall, and per hazard type)
near = pos & hazDist <= 20;
hazTypes = unique(hazName(pos));
fprintf("\n%-13s %-13s | %8s %11s %s| %10s %10s | %7s %6s\n", "model", "variant", "recall", "recall<=20m", ...
        sprintf("%-14s", hazTypes + " "), "side flag", "false flag", "latency", "failed");
for c = 1:nC
    f = flag(c, :);
    perType = arrayfun(@(n) sprintf("%2d/%-2d", nnz(f & pos & hazName == n), nnz(pos & hazName == n)), hazTypes);
    fprintf("%-13s %-13s | %3d/%-3d  %5d/%-3d   %s | %5d/%-3d  %5d/%-3d | %6.2fs %6d\n", configs(c).model, configs(c).variant, ...
            nnz(f & pos), nnz(pos), nnz(f & near), nnz(near), sprintf("%-14s", perType), ...
            nnz(f & side), nnz(side), nnz(f & neg), nnz(neg), mean(lat(c, :)), nnz(~ok(c, :)));
end
fprintf("\nper-hazard columns: %s\n", strjoin(hazTypes, " | "));

save(fullfile("vlm_log", "scene_benchmark.mat"), "configs", "frames", "label", "hazDist", "hazName", ...
     "flag", "lat", "ok", "what");
