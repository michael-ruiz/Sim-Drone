%% Offline VLM Sector Benchmark
% Replays every logged query (vlm_log/*/queries.mat + qNNN.png raw frames) through
% several model / overlay configurations and scores each answer against a
% deterministic reference: the open sector (depth >= openDepth) closest to the
% target bearing, or 0 if every sector is blocked. Same frames for every config,
% so the comparison isn't confounded by flights diverging after one decision.
clear;

configs = struct( ...
    "model",   {"qwen2.5vl:3b", "qwen2.5vl:3b", "qwen2.5vl:7b", "qwen2.5vl:7b"}, ...
    "overlay", {false,          true,           false,          true});
openDepth       = 10;                  % A sector counts as open at >= 10 m
sectorAnglesDeg = [-28 -14 0 14 28];

%% Gather logged frames
frames = struct("file", {}, "depths", {}, "bearing", {}, "dist", {}, "label", {}, "run", {});
logs = dir(fullfile("vlm_log", "*", "queries.mat"));
for L = 1:numel(logs)
    S = load(fullfile(logs(L).folder, logs(L).name), "queryLog");
    [~, run] = fileparts(logs(L).folder);
    for q = 1:numel(S.queryLog)
        f = fullfile(logs(L).folder, sprintf("q%03d.png", q));
        if ~isfile(f), continue; end
        ql = S.queryLog(q);
        frames(end+1) = struct("file", f, "depths", ql.depths, "bearing", ql.bearing, ... %#ok<SAGROW>
                               "dist", ql.dist, "label", ql.label, "run", string(run));
    end
end
nF = numel(frames);
if nF == 0
    error("No logged frames found under vlm_log/ - fly drone_vlm_test.m first.");
end

% Reference answer per frame
ref = zeros(1, nF);
for i = 1:nF
    open = frames(i).depths >= openDepth;
    if any(open)
        cand = find(open);
        [~, k] = min(abs(sectorAnglesDeg(cand) - frames(i).bearing));
        ref(i) = cand(k);
    end
end
fprintf("%d frames from %d runs | reference sectors [0..5]: %s\n", nF, numel(logs), ...
        mat2str(histcounts(ref, -0.5:1:5.5)));

%% Query every config on every frame
nC = numel(configs);
picks = zeros(nC, nF);
lat   = zeros(nC, nF);
fellBack = false(nC, nF);
for c = 1:nC
    try % Load the model before timing
        webwrite("http://localhost:11434/api/generate", ...
                 struct("model", configs(c).model, "prompt", "", "stream", false), ...
                 weboptions("MediaType", "application/json", "Timeout", 120));
    catch ME
        fprintf("Warm-up failed for %s: %s\n", configs(c).model, ME.message);
    end
    for i = 1:nF
        img = imread(frames(i).file);
        [picks(c, i), reason, lat(c, i)] = queryVLMNavigator(img, frames(i).depths, frames(i).bearing, ...
                                              frames(i).dist, frames(i).label, configs(c).model, configs(c).overlay);
        fellBack(c, i) = contains(reason, "Local Fallback");
    end
    fprintf("done: %s overlay=%d\n", configs(c).model, configs(c).overlay);
end

%% Score
offCenter = ref ~= 3;
fprintf("\n%-14s %-7s | %7s %11s %9s %8s %9s %8s\n", "model", "overlay", "exact", "off-center", "unsafe", "center%", "latency", "fallback");
for c = 1:nC
    p = picks(c, :);
    chosenDepth = zeros(1, nF);
    chosenDepth(p > 0) = arrayfun(@(i) frames(i).depths(p(i)), find(p > 0));
    unsafe = (p > 0 & chosenDepth < openDepth) | (p == 0 & ref > 0);
    fprintf("%-14s %-7s | %6.0f%% %4.0f%% (n=%2d) %8.0f%% %7.0f%% %8.2fs %8d\n", ...
            configs(c).model, string(configs(c).overlay), ...
            100 * mean(p == ref), 100 * mean(p(offCenter) == ref(offCenter)), nnz(offCenter), ...
            100 * mean(unsafe), 100 * mean(p == 3), mean(lat(c, :)), nnz(fellBack(c, :)));
end
fprintf("\nexact = matches reference | off-center = accuracy when the reference is NOT sector 3\n");
fprintf("unsafe = picked a sector < %d m (or 'blocked' when one was open)\n", openDepth);

save(fullfile("vlm_log", "benchmark.mat"), "configs", "frames", "ref", "picks", "lat", "fellBack", "openDepth");
