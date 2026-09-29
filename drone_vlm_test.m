%% 1. Initialize Live Display (RGB + VLM Reasoning HUD + Depth)
clear; close all;

fig = figure(Name="VLM + Depth Reflex Cross-City UAV Navigator", ...
             Color="w", Position=[50 70 1250 520]);
tLayout = tiledlayout(fig, 1, 2, TileSpacing="compact");

axRGB = nexttile(tLayout, 1);
hRGB = imshow(uint8(zeros(480, 640, 3)), Parent=axRGB);
hold(axRGB, "on");

colEdges = round(linspace(40, 600, 6));
hBoxes = gobjects(1, 5);
hLabels = gobjects(1, 5);
for i = 1:5
    cx = mean([colEdges(i), colEdges(i+1)]);
    hBoxes(i) = rectangle(axRGB, ...
        Position=[colEdges(i), 130, colEdges(i+1)-colEdges(i), 105], ...
        EdgeColor="g", LineWidth=1.5, LineStyle="--");
    hLabels(i) = text(axRGB, cx-12, 182, sprintf("[%d]", i), ...
        Color="y", FontWeight="bold", FontSize=12, BackgroundColor="k");
end

hStatus = text(axRGB, 12, 28, "INITIALIZING...", Color="g", ...
               FontWeight="bold", FontSize=10, BackgroundColor="k");
hVLMText = text(axRGB, 12, 450, "VLM: Waiting for takeoff...", Color="c", ...
                FontWeight="bold", FontSize=10, BackgroundColor="k");
title(axRGB, "Onboard RGB + Sector HUD [1-5] (HUD only, not sent to VLM)");

axDepth = nexttile(tLayout, 2);
hDepth = imagesc(axDepth, zeros(480, 640), [0 45]);
axis(axDepth, "image", "off");
colormap(axDepth, "turbo");
cb = colorbar(axDepth);
ylabel(cb, "Depth (m)");
title(axDepth, "Live Depth Map (m)");

%% 2. Set Up UAV Guidance Model (NED Frame)
model = multirotor;
s = state(model);
u = control(model);
e = environment(model);

s(1:3) = [0; 0; -2]; % Start at (0,0), 2m altitude
hoverThrust = model.Configuration.Mass * e.Gravity;
s(13) = hoverThrust;
u.Thrust = hoverThrust;

%% 3. Mission & Global Route (street-level waypoints from city_map.mat)
startXY  = [0; 0];
goalXY   = [40; 112];  % Street behind a city block; reactive mode turns into a wall at X=40
useRoute = false;      % false = pure reactive VLM toward the goal (no map)

cityMap = [];
if isfile("city_map.mat")
    cityMap = load("city_map.mat"); % Also used as the post-flight plot background
end
route = goalXY';
if useRoute
    if isempty(cityMap)
        error("city_map.mat not found - run map_city_block.m first.");
    end
    % Anything taller than 10 m blocks a 12 m cruise; keep 4 m from walls
    route  = planRoute(cityMap, startXY', goalXY', 10, 4);
    goalXY = route(end, :)'; % Planner snaps the goal to the nearest free street cell
    fprintf("Route: %d waypoints -> %s\n", size(route, 1), ...
            strjoin(compose("(%.0f, %.0f)", route(:,1), route(:,2)), " -> "));
end

%% 4. Create Unreal Engine World & Camera
sampleTime = 1/20; % 20 Hz low-level physics & depth reflex
stopTime   = 90;   % Long enough for the ~190 m street route

world = sim3d.World(Scene="USCityBlock", ...
                    Output=@(w) stepUAVPhysics(w, sampleTime), ...
                    Update=@(w) vlmAndReflexLoop(w, sampleTime, hRGB, hDepth, hStatus, hVLMText, hBoxes, colEdges));

cam = sim3d.sensors.Camera(ActorName="DroneCamera", ...
                           ImageSize=[480 640], ...
                           FocalLength=[450 450], ...
                           OpticalCenter=[320 240], ...
                           EnableDepthOutput=true);
cam.Translation = [startXY', 2];
add(world, cam);

% Shared State & VLM Configuration
world.UserData.Time          = 0;
world.UserData.LastVLMTime   = -10;
world.UserData.VLMInterval   = 2.0;        % Query VLM every 2.0 seconds of sim time
world.UserData.GoalXY        = goalXY;
world.UserData.Route         = route;      % Kx2 waypoints [x y]; last row is the goal
world.UserData.WaypointIdx   = 1;
world.UserData.WaypointRadius = 5.0;       % Advance to the next waypoint within 5 m
world.UserData.UseRoute      = useRoute;
world.UserData.ChosenSector  = 3;          % 0=Blocked, 1=Far Left, 2=Left, 3=Center, 4=Right, 5=Far Right
world.UserData.VLMHeading    = 0;          % World yaw commanded by VLM (radians)
world.UserData.VLMReason     = "Initial climb to cruise altitude";
world.UserData.OllamaModel   = "qwen2.5vl:3b"; % Also tried: "llava" (strong sector-2 bias, ignored depths)
world.UserData.Model         = model;
world.UserData.State         = s;
world.UserData.Control       = u;
world.UserData.Env           = e;
world.UserData.HoverThrust   = hoverThrust;
world.UserData.History       = [];
world.UserData.Arrived       = false;
world.UserData.GoalRadius    = 3.0;        % Hover once within 3m of the goal
world.UserData.Aligning      = false;      % True while rotating to bring the target into the camera FOV
% Route waypoints are line-of-sight, so keep them in view; without a route, only turn
% around when the goal is behind so the drone can follow streets away from it.
world.UserData.AlignEnterDeg = ternary(useRoute, 35, 90);
world.UserData.Escaping      = false;      % Wall-following escape when blocked or stuck
world.UserData.EscapeUntil   = -Inf;
world.UserData.EscapeHeading = 0;
world.UserData.EscapeDuration = 4.0;
world.UserData.EscapeCount   = 0;
world.UserData.StuckWindow   = 6.0;        % Stuck = under 2 m closer to the target in 6 s
world.UserData.StuckMinProgress = 2.0;
world.UserData.ProgressRefT  = 3.5;
world.UserData.ProgressRefDist = Inf;

% Warm up Ollama so the first in-flight query doesn't time out while the model loads
try
    webwrite("http://localhost:11434/api/generate", ...
             struct("model", world.UserData.OllamaModel, "prompt", "", "stream", false), ...
             weboptions("MediaType", "application/json", "Timeout", 120));
    fprintf("Ollama model '%s' loaded.\n", world.UserData.OllamaModel);
catch ME
    fprintf("Ollama warm-up failed (%s) - using local fallback planner.\n", ME.message);
end

%% 5. Run Co-Simulation
try
    run(world, sampleTime, stopTime);
catch ME
    delete(world);
    rethrow(ME);
end

histData    = world.UserData.History;
arrived     = world.UserData.Arrived;
escapeCount = world.UserData.EscapeCount;
wpReached   = world.UserData.WaypointIdx - 1;
delete(world);

%% 6. Post-Flight Map (north/X up, east/Y right)
if ~isempty(histData)
    finalDist = norm(histData(end,1:2)' - goalXY);
    fprintf("\n=== FLIGHT SUMMARY ===\n");
    fprintf("Final position: (%.1f, %.1f), altitude %.1f m\n", histData(end,1), histData(end,2), histData(end,3));
    fprintf("Final distance to goal: %.1f m | Goal reached: %s\n", finalDist, string(arrived));
    fprintf("Waypoints passed: %d of %d | Escapes: %d\n", wpReached, size(route, 1) - 1, escapeCount);
    fprintf("VLM queries: %d | Steps with depth reflex active: %d of %d\n", ...
            nnz(histData(:,4) == 2), nnz(histData(:,4) == 1), size(histData, 1));

    fig2 = figure(Name="Cross-City VLM Navigation Trajectory", Color="w", Position=[100 60 900 800]);
    ax = axes(fig2);
    hold(ax, "on"); grid(ax, "on");
    if ~isempty(cityMap)
        imagesc(ax, cityMap.yCenters, cityMap.xCenters, min(cityMap.heightMap, 60));
        colormap(ax, flipud(gray)); clim(ax, [0 60]);
        cbMap = colorbar(ax); ylabel(cbMap, "Building height (m)");
    end
    hRoute = gobjects(0);
    if useRoute
        hRoute = plot(ax, [startXY(2); route(:,2)], [startXY(1); route(:,1)], "--o", ...
                      Color=[1 0.55 0], LineWidth=1.5, MarkerFaceColor=[1 0.55 0], ...
                      DisplayName="Planned Route");
    end
    hPath  = plot(ax, histData(:,2), histData(:,1), "b-", LineWidth=2);
    repPts = histData(:,4) == 1;
    vlmPts = histData(:,4) == 2;
    escPts = histData(:,4) == 3;
    hRep = scatter(ax, histData(repPts,2), histData(repPts,1), 20, "r", "filled");
    hEsc = scatter(ax, histData(escPts,2), histData(escPts,1), 20, "m", "filled");
    hVLM = scatter(ax, histData(vlmPts,2), histData(vlmPts,1), 50, "g", "filled", MarkerEdgeColor="k");
    hGoal = plot(ax, goalXY(2), goalXY(1), "kp", MarkerSize=18, MarkerFaceColor="y");
    axis(ax, "equal"); set(ax, YDir="normal");
    xlabel(ax, "Y (m, east)"); ylabel(ax, "X (m, north)");
    title(ax, ternary(useRoute, "Top-Down Path (Route-Guided VLM)", "Top-Down Path (Reactive VLM, No Route)"));
    set([hPath, hRep, hEsc, hVLM, hGoal], {"DisplayName"}, ...
        {"Flight Path"; "Depth Reflex Active"; "Escape"; "VLM Query Point"; sprintf("Goal (%g, %g)", goalXY(1), goalXY(2))});
    legend([hRoute, hPath, hRep, hEsc, hVLM, hGoal], Location="bestoutside");

    exportgraphics(fig2, "vlm_run.png", Resolution=150);
    fprintf("Trajectory plot saved to vlm_run.png\n");
end

%% --- Callback 1: Step Multirotor Physics ---
function stepUAVPhysics(world, dt)
    ud = world.UserData;
    ud.Time = ud.Time + dt;

    [~, sHistory] = ode45(@(~, x) derivative(ud.Model, x, ud.Control, ud.Env), ...
                          [0 dt], ud.State);
    s = sHistory(end, :)';
    ud.State = s;
    world.UserData = ud;

    world.Actors.DroneCamera.Translation = [s(1), s(2), max(0.5, -s(3))];
    world.Actors.DroneCamera.Rotation    = [0, 0, s(7)];
end

%% --- Callback 2: High-Level VLM Planner + 20 Hz Depth Safety Reflex ---
function vlmAndReflexLoop(world, ~, hRGB, hDepth, hStatus, hVLMText, hBoxes, colEdges)
    [rgbFrame, depthFrame] = read(world.Actors.DroneCamera);
    depthFrame = min(double(depthFrame), 45); % uint8 meters; clamp sky (255) to the display range

    ud  = world.UserData;
    t   = ud.Time;
    s   = ud.State;
    u   = ud.Control;
    pos = s(1:3);
    vel = s(4:6);
    yaw = s(7);

    % --- 1. LOW-LEVEL PERCEPTION: 5-Sector Depth Scan ---
    sectorDepths = zeros(1, 5);
    for i = 1:5
        strip = depthFrame(130:235, colEdges(i):colEdges(i+1));
        sectorDepths(i) = prctile(strip(:), 2);
    end
    sectorAngles = deg2rad([-28, -14, 0, 14, 28]);
    repulseDist  = 9.0;
    zCruiseNED   = -12; % 12m cruise altitude

    vecToGoal  = ud.GoalXY - pos(1:2);
    distToGoal = norm(vecToGoal);
    if distToGoal < ud.GoalRadius
        ud.Arrived = true;
    end
    active = (t > 3.5) && ~ud.Arrived;

    % Current target = next route waypoint (or the goal itself without a route)
    target = ud.Route(ud.WaypointIdx, :)';
    if ud.WaypointIdx < size(ud.Route, 1) && norm(target - pos(1:2)) < ud.WaypointRadius
        ud.WaypointIdx = ud.WaypointIdx + 1;
        target = ud.Route(ud.WaypointIdx, :)';
        ud.ProgressRefT    = t;
        ud.ProgressRefDist = norm(target - pos(1:2));
        fprintf("t=%5.1fs pos=(%6.1f, %6.1f) waypoint reached -> next waypoint %d at (%.0f, %.0f)\n", ...
                t, pos(1), pos(2), ud.WaypointIdx, target(1), target(2));
    end
    vecToTarget   = target - pos(1:2);
    distToTarget  = norm(vecToTarget);
    targetHeading = atan2(vecToTarget(2), vecToTarget(1));
    targetBearing = rad2deg(atan2(sin(targetHeading - yaw), cos(targetHeading - yaw))); % + is Right

    % --- 2. ESCAPE / STUCK HANDLING ---
    if ud.Escaping && t >= ud.EscapeUntil
        ud.Escaping        = false;
        ud.LastVLMTime     = -Inf; % Fresh VLM decision on the new view
        ud.ProgressRefT    = t;
        ud.ProgressRefDist = distToTarget;
    end
    if active && ~ud.Escaping && (t - ud.ProgressRefT) >= ud.StuckWindow
        if ud.ProgressRefDist - distToTarget < ud.StuckMinProgress
            ud = startEscape(ud, t, pos, yaw, targetBearing, sectorDepths, "stuck (no progress)");
        end
        ud.ProgressRefT    = t;
        ud.ProgressRefDist = distToTarget;
    end

    % Target outside the view -> turn to face it first (hysteresis: resume VLM within 10 deg)
    if active && ~ud.Escaping
        if ~ud.Aligning && abs(targetBearing) > ud.AlignEnterDeg
            ud.Aligning = true;
            fprintf("t=%5.1fs pos=(%6.1f, %6.1f) target @ %+4.0f deg -> turning to face it\n", ...
                    t, pos(1), pos(2), targetBearing);
        elseif ud.Aligning && abs(targetBearing) < 10
            ud.Aligning    = false;
            ud.LastVLMTime = -Inf; % Query the VLM immediately on the new view
        end
    end

    % --- 3. HIGH-LEVEL VLM PLANNER (Runs Every 2.0s After Takeoff) ---
    vlmTriggered = false;
    if active && ~ud.Aligning && ~ud.Escaping && ((t - ud.LastVLMTime) >= ud.VLMInterval)
        ud.LastVLMTime = t;
        vlmTriggered   = true;

        targetLabel = ternary(ud.UseRoute, "next route waypoint", "destination");
        [chosenSector, reason] = queryVLMNavigator(rgbFrame, sectorDepths, targetBearing, ...
                                                   distToTarget, targetLabel, ud.OllamaModel);
        ud.ChosenSector = chosenSector;
        ud.VLMReason    = sprintf("[Sector %d] %s", chosenSector, reason);
        fprintf("t=%5.1fs pos=(%6.1f, %6.1f) target=%5.1fm @ %+4.0f deg | depths=[%s] -> sector %d | %s\n", ...
                t, pos(1), pos(2), distToTarget, targetBearing, ...
                strjoin(compose("%.0f", sectorDepths), " "), chosenSector, reason);
        if chosenSector == 0
            ud = startEscape(ud, t, pos, yaw, targetBearing, sectorDepths, "VLM reports all sectors blocked");
        else
            ud.VLMHeading = yaw + sectorAngles(chosenSector);
        end
    end

    % --- 4. COMBINE HEADING COMMAND WITH 20 Hz DEPTH REPULSION ---
    if t <= 3.5
        % Initial takeoff to 12m street cruise altitude
        vWorldCmd = -0.8 * pos(1:2);
        yawTarget = 0;
        modeFlag  = 0;
        statusStr = "TAKEOFF TO CRUISE ALTITUDE (12 m)";
        statusCol = "g";
    elseif ud.Arrived
        % Hold position over the goal
        vWorldCmd = 0.8 * vecToGoal;
        yawTarget = yaw;
        modeFlag  = 0;
        statusStr = sprintf("GOAL REACHED - HOVERING (%.1f m off)", distToGoal);
        statusCol = "c";
    else
        if ud.Escaping
            % Follow the wall toward the more promising side
            vAttract  = 2.5 * [cos(ud.EscapeHeading); sin(ud.EscapeHeading)];
            yawTarget = ud.EscapeHeading;
        elseif ud.Aligning
            % Rotate in place toward the target (repulsion still active)
            vAttract  = [0; 0];
            yawTarget = targetHeading;
        else
            % Attractive velocity along the VLM's chosen corridor heading
            vAttract  = 3.8 * [cos(ud.VLMHeading); sin(ud.VLMHeading)];
            % Keep the camera on the VLM corridor; repulsion only shifts velocity so hazards stay in view
            yawTarget = ud.VLMHeading;
        end

        % Repulsive safety vectors from 5-sector depth map
        vRepulse = [0; 0];
        modeFlag = 0;
        for i = 1:5
            d = sectorDepths(i);
            if d < repulseDist
                modeFlag = 1;
                obsBearing  = yaw + sectorAngles(i);
                obsDirWorld = [cos(obsBearing); sin(obsBearing)];
                pushMag     = 5.0 * ((repulseDist - d) / repulseDist);
                vRepulse    = vRepulse - pushMag * obsDirWorld;
            end
        end

        vWorldCmd = vAttract + vRepulse;
        if norm(vWorldCmd) > 4.0
            vWorldCmd = vWorldCmd * (4.0 / norm(vWorldCmd));
        end

        statusStr = sprintf("WP %d/%d: %.1f m | GOAL: %.1f m | Min Depth: %.0f m", ...
                            ud.WaypointIdx, size(ud.Route, 1), distToTarget, distToGoal, min(sectorDepths));
        if ud.Escaping
            statusStr = sprintf("ESCAPING (%.1f s) | %s", ud.EscapeUntil - t, statusStr);
        elseif ud.Aligning
            statusStr = sprintf("TURNING TO TARGET (%+.0f deg) | %s", targetBearing, statusStr);
        end
        statusCol = "g";
        if modeFlag == 1, statusCol = "r"; end
        if ud.Escaping, statusCol = "m"; end
    end

    % --- 5. ACTUATE: Multirotor Attitude & Altitude Control ---
    yawErr = atan2(sin(yawTarget - yaw), cos(yawTarget - yaw));
    u.YawRate = max(-1.8, min(1.8, 2.4 * yawErr));

    vForwardCmd  =  cos(yaw) * vWorldCmd(1) + sin(yaw) * vWorldCmd(2);
    vRightCmd    = -sin(yaw) * vWorldCmd(1) + cos(yaw) * vWorldCmd(2);
    vForwardCurr =  cos(yaw) * vel(1) + sin(yaw) * vel(2);
    vRightCurr   = -sin(yaw) * vel(1) + cos(yaw) * vel(2);

    maxTilt = deg2rad(18);
    u.Pitch = max(-maxTilt, min(maxTilt, -0.22 * (vForwardCmd - vForwardCurr)));
    u.Roll  = max(-maxTilt, min(maxTilt,  0.22 * (vRightCmd   - vRightCurr)));

    altErr   = pos(3) - zCruiseNED;
    vzUpCurr = -vel(3);
    u.Thrust = max(0, ud.HoverThrust + ud.Model.Configuration.Mass * (6.0 * altErr - 3.8 * vzUpCurr));

    % Log flags: 0 = normal, 1 = depth reflex, 2 = VLM query, 3 = escape
    logFlag = modeFlag;
    if ud.Escaping, logFlag = 3; end
    if vlmTriggered, logFlag = 2; end
    ud.History = [ud.History; pos(1), pos(2), -pos(3), logFlag];
    ud.Control = u;
    world.UserData = ud;

    % --- 6. UPDATE HUD ---
    if isvalid(hRGB)
        set(hRGB, CData=rgbFrame);
        set(hDepth, CData=depthFrame);
        set(hStatus, String=sprintf("t = %.1f s | %s", t, statusStr), Color=statusCol);
        set(hVLMText, String=sprintf("VLM Decision: %s", ud.VLMReason));

        for i = 1:5
            if i == ud.ChosenSector
                set(hBoxes(i), EdgeColor="c", LineStyle="-", LineWidth=3.0); % Cyan = VLM choice
            elseif sectorDepths(i) < repulseDist
                set(hBoxes(i), EdgeColor="r", LineStyle="-", LineWidth=2.5); % Red = Obstacle
            else
                set(hBoxes(i), EdgeColor="g", LineStyle="--", LineWidth=1.2);
            end
        end
        drawnow limitrate;
    end
end

%% --- Helper: Start a Wall-Following Escape ---
function ud = startEscape(ud, t, pos, yaw, targetBearing, sectorDepths, why)
    % Turn 90 deg toward the side the target is on; if it's dead ahead, take the more open side
    if abs(targetBearing) > 5
        side = sign(targetBearing);
    elseif mean(sectorDepths(4:5)) > mean(sectorDepths(1:2))
        side = 1;
    else
        side = -1;
    end
    ud.Escaping      = true;
    ud.Aligning      = false;
    ud.EscapeHeading = yaw + side * pi/2;
    ud.EscapeUntil   = t + ud.EscapeDuration;
    ud.EscapeCount   = ud.EscapeCount + 1;
    fprintf("t=%5.1fs pos=(%6.1f, %6.1f) ESCAPE %d: %s -> turning %s for %.0f s\n", ...
            t, pos(1), pos(2), ud.EscapeCount, why, ternary(side > 0, "right", "left"), ud.EscapeDuration);
end

%% --- Helper: Query Local Ollama VLM (with Depth + Bearing Fallback) ---
function [sector, reason] = queryVLMNavigator(rgbFrame, sectorDepths, bearingDeg, distToTarget, targetLabel, modelName)
    % Encode current RGB frame as Base64 JPEG for the VLM API
    tmpFile = [tempname, '.jpg'];
    imwrite(imresize(rgbFrame, 0.5), tmpFile, 'Quality', 80);
    fid = fopen(tmpFile, 'rb');
    rawBytes = fread(fid, inf, '*uint8');
    fclose(fid);
    delete(tmpFile);
    b64Image = matlab.net.base64encode(rawBytes);

    prompt = sprintf([ ...
        'You are an autonomous urban drone navigator flying at 12 m altitude. The camera view is divided from left to right ', ...
        'into 5 sectors: 1 (Far Left, -28 deg), 2 (Left, -14 deg), 3 (Center, 0 deg), 4 (Right, +14 deg), 5 (Far Right, +28 deg). ', ...
        'Measured depth clearance in meters for sectors [1..5] is [%.0f, %.0f, %.0f, %.0f, %.0f] (45 means 45 m or more). ', ...
        'Your %s is %.1f meters away at relative bearing %.0f degrees (+ is Right, - is Left). ', ...
        'Pick the open sector (1-5) that avoids buildings and is closest to that bearing. ', ...
        'If every sector is blocked by a building or wall closer than 8 m, answer sector 0. ', ...
        'Respond ONLY with valid JSON: {"sector": <int 0-5>, "reason": "<short 6-word explanation>"}'], ...
        sectorDepths(1), sectorDepths(2), sectorDepths(3), sectorDepths(4), sectorDepths(5), ...
        targetLabel, distToTarget, bearingDeg);

    try
        % Call local Ollama server (http://localhost:11434/api/generate)
        payload = struct("model", modelName, ...
                         "prompt", prompt, ...
                         "images", {{b64Image}}, ...
                         "format", "json", ...
                         "stream", false);
        opts = weboptions("MediaType", "application/json", "Timeout", 8);
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
        % Automatic fallback if Ollama is not running yet: score sectors by depth clearance + bearing alignment
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
end

%% --- Helper: Grid Route Planner over the City Height Map ---
function waypoints = planRoute(map, startXY, goalXY, obstacleHeight, inflateRadius)
    % Shortest 8-connected path over free cells, penalizing cells near walls,
    % then pruned to line-of-sight corner waypoints. Returns Kx2 [x y] (goal last).
    xs = map.xCenters; ys = map.yCenters; cs = map.cellSize;
    occ = isnan(map.heightMap) | map.heightMap > obstacleHeight;
    r = round(inflateRadius / cs);
    blocked  = movmax(movmax(double(occ), 2*r + 1, 1), 2*r + 1, 2) > 0;
    nearWall = movmax(movmax(double(occ), 4*r + 1, 1), 4*r + 1, 2) > 0;
    [nx, ny] = size(blocked);

    startCell = nearestFreeCell(blocked, xyToCell(startXY, xs, ys, cs));
    goalCell  = nearestFreeCell(blocked, xyToCell(goalXY, xs, ys, cs));
    snappedGoal = [xs(goalCell(1)), ys(goalCell(2))];
    if norm(snappedGoal - goalXY) > cs
        fprintf("Goal (%.0f, %.0f) is inside an obstacle - snapped to (%.0f, %.0f)\n", ...
                goalXY(1), goalXY(2), snappedGoal(1), snappedGoal(2));
    end

    idx = reshape(1:nx*ny, nx, ny);
    S = []; T = []; W = [];
    for off = [1 0; 0 1; 1 1; 1 -1]'
        di = off(1); dj = off(2);
        i1 = max(1, 1 - di):min(nx, nx - di);
        j1 = max(1, 1 - dj):min(ny, ny - dj);
        a = idx(i1, j1); b = idx(i1 + di, j1 + dj);
        ok = ~blocked(a) & ~blocked(b);
        cost = hypot(di, dj) * (1 + 2 * (nearWall(a) | nearWall(b)));
        S = [S; a(ok)]; T = [T; b(ok)]; W = [W; cost(ok)]; %#ok<AGROW>
    end
    G = graph(S, T, W, nx*ny);
    path = shortestpath(G, idx(startCell(1), startCell(2)), idx(goalCell(1), goalCell(2)));
    if isempty(path)
        error("No street route found from (%.0f, %.0f) to (%.0f, %.0f).", ...
              startXY(1), startXY(2), goalXY(1), goalXY(2));
    end
    [pi_, pj] = ind2sub([nx ny], path(:));
    pts = [xs(pi_)', ys(pj)'];

    % Greedy line-of-sight pruning to corner waypoints
    waypoints = zeros(0, 2);
    k = 1;
    while k < size(pts, 1)
        j = size(pts, 1);
        while j > k + 1 && ~lineOfSight(blocked, pts(k,:), pts(j,:), xs, ys, cs)
            j = j - 1;
        end
        waypoints(end+1, :) = pts(j, :); %#ok<AGROW>
        k = j;
    end
end

function c = xyToCell(p, xs, ys, cs)
    c = [round((p(1) - xs(1)) / cs) + 1, round((p(2) - ys(1)) / cs) + 1];
    c = min(max(c, [1 1]), [numel(xs) numel(ys)]);
end

function c = nearestFreeCell(blocked, c)
    if ~blocked(c(1), c(2)), return; end
    [fi, fj] = find(~blocked);
    [~, k] = min((fi - c(1)).^2 + (fj - c(2)).^2);
    c = [fi(k), fj(k)];
end

function ok = lineOfSight(blocked, p1, p2, xs, ys, cs)
    n = max(2, ceil(norm(p2 - p1) / (0.5 * cs)));
    px = linspace(p1(1), p2(1), n);
    py = linspace(p1(2), p2(2), n);
    ix = round((px - xs(1)) / cs) + 1;
    iy = round((py - ys(1)) / cs) + 1;
    ok = ~any(blocked(sub2ind(size(blocked), ix, iy)));
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
