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
title(axRGB, "Onboard RGB + Visual Prompting Sectors [1-5]");

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

%% 3. Create Unreal Engine World & Camera
sampleTime = 1/20; % 20 Hz low-level physics & depth reflex
stopTime   = 45;   % 45-second cross-city flight

world = sim3d.World(Scene="USCityBlock", ...
                    Output=@(w) stepUAVPhysics(w, sampleTime), ...
                    Update=@(w) vlmAndReflexLoop(w, sampleTime, hRGB, hDepth, hStatus, hVLMText, hBoxes, colEdges));

cam = sim3d.sensors.Camera(ActorName="DroneCamera", ...
                           ImageSize=[480 640], ...
                           FocalLength=[450 450], ...
                           OpticalCenter=[320 240], ...
                           EnableDepthOutput=true);
cam.Translation = [0 0 2];
add(world, cam);

% Shared State & VLM Configuration
world.UserData.Time          = 0;
world.UserData.LastVLMTime   = -10;
world.UserData.VLMInterval   = 2.0;        % Query VLM every 2.0 seconds of sim time
world.UserData.GoalXY        = [120; 0];   % Destination on the other side of the city
world.UserData.ChosenSector  = 3;          % 1=Far Left, 2=Left, 3=Center, 4=Right, 5=Far Right
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
world.UserData.Aligning      = false;      % True while rotating to bring the goal into the camera FOV
world.UserData.GoalRadius    = 3.0;        % Hover once within 3m of the goal

% Warm up Ollama so the first in-flight query doesn't time out while the model loads
try
    webwrite("http://localhost:11434/api/generate", ...
             struct("model", world.UserData.OllamaModel, "prompt", "", "stream", false), ...
             weboptions("MediaType", "application/json", "Timeout", 120));
    fprintf("Ollama model '%s' loaded.\n", world.UserData.OllamaModel);
catch ME
    fprintf("Ollama warm-up failed (%s) - using local fallback planner.\n", ME.message);
end

%% 4. Run Co-Simulation
try
    run(world, sampleTime, stopTime);
catch ME
    delete(world);
    rethrow(ME);
end

histData = world.UserData.History;
goalXY   = world.UserData.GoalXY;
arrived  = world.UserData.Arrived;
delete(world);

%% 5. Post-Flight Cross-City Map
if ~isempty(histData)
    finalDist = norm(histData(end,1:2)' - goalXY);
    fprintf("\n=== FLIGHT SUMMARY ===\n");
    fprintf("Final position: (%.1f, %.1f), altitude %.1f m\n", histData(end,1), histData(end,2), histData(end,3));
    fprintf("Final distance to goal: %.1f m | Goal reached: %s\n", finalDist, string(arrived));
    fprintf("VLM queries: %d | Steps with depth reflex active: %d of %d\n", ...
            nnz(histData(:,4) == 2), nnz(histData(:,4) == 1), size(histData, 1));

    fig2 = figure(Name="Cross-City VLM Navigation Trajectory", Color="w");
    plot(histData(:,1), histData(:,2), "b-", LineWidth=2);
    hold on; grid on; axis equal;
    vlmPts = histData(:,4) == 2;
    repPts = histData(:,4) == 1;
    scatter(histData(repPts,1), histData(repPts,2), 25, "r", "filled");
    scatter(histData(vlmPts,1), histData(vlmPts,2), 70, "g", "filled", "MarkerEdgeColor", "k");
    plot(goalXY(1), goalXY(2), "kp", MarkerSize=16, MarkerFaceColor="y");
    xlabel("X (m)"); ylabel("Y (m)");
    title("Top-Down Cross-City Path (VLM Waypoints + Depth Safety Reflex)");
    legend("Flight Path", "Depth Reflex Active", "VLM Query Point", ...
           sprintf("City Goal (%g, %g)", goalXY(1), goalXY(2)), Location="best");

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
    depthFrame = min(double(depthFrame), 45); % Clamp sky/Inf returns to the display range

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

    % Relative angle to final city goal (+ is Right, - is Left)
    goalHeading = atan2(vecToGoal(2), vecToGoal(1));
    goalBearing = rad2deg(atan2(sin(goalHeading - yaw), cos(goalHeading - yaw)));

    % Goal outside the camera FOV -> no sector can make progress, so turn to face it first.
    % Hysteresis: start aligning beyond 35 deg (FOV edge), resume VLM once within 10 deg.
    if (t > 3.5) && ~ud.Arrived
        if ~ud.Aligning && abs(goalBearing) > 35
            ud.Aligning = true;
            fprintf("t=%5.1fs pos=(%6.1f, %6.1f) goal @ %+4.0f deg outside FOV -> turning to face goal\n", ...
                    t, pos(1), pos(2), goalBearing);
        elseif ud.Aligning && abs(goalBearing) < 10
            ud.Aligning    = false;
            ud.LastVLMTime = -Inf; % Query the VLM immediately on the new view
        end
    end

    % --- 2. HIGH-LEVEL VLM PLANNER (Runs Every 2.0s After Takeoff) ---
    vlmTriggered = false;
    if (t > 3.5) && ~ud.Arrived && ~ud.Aligning && ((t - ud.LastVLMTime) >= ud.VLMInterval)
        ud.LastVLMTime = t;
        vlmTriggered   = true;

        [chosenSector, reason] = queryVLMNavigator(rgbFrame, sectorDepths, goalBearing, distToGoal, ud.OllamaModel);
        ud.ChosenSector = chosenSector;
        ud.VLMHeading   = yaw + sectorAngles(chosenSector);
        ud.VLMReason    = sprintf("[Sector %d] %s", chosenSector, reason);
        fprintf("t=%5.1fs pos=(%6.1f, %6.1f) goal=%5.1fm @ %+4.0f deg | depths=[%s] -> sector %d | %s\n", ...
                t, pos(1), pos(2), distToGoal, goalBearing, ...
                strjoin(compose("%.1f", sectorDepths), " "), chosenSector, reason);
    end

    % --- 3. COMBINE VLM HEADING WITH 20 Hz DEPTH REPULSION ---
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
        if ud.Aligning
            % Rotate in place toward the goal (repulsion still active)
            vAttract  = [0; 0];
            yawTarget = goalHeading;
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

        statusStr = sprintf("GOAL DIST: %.1f m | Min Depth: %.1f m", distToGoal, min(sectorDepths));
        if ud.Aligning
            statusStr = sprintf("TURNING TO GOAL (%+.0f deg) | %s", goalBearing, statusStr);
        end
        statusCol = "g";
        if modeFlag == 1, statusCol = "r"; end
    end

    % --- 4. ACTUATE: Multirotor Attitude & Altitude Control ---
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

    logFlag = modeFlag;
    if vlmTriggered, logFlag = 2; end
    ud.History = [ud.History; pos(1), pos(2), -pos(3), logFlag];
    ud.Control = u;
    world.UserData = ud;

    % --- 5. UPDATE HUD ---
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

%% --- Helper: Query Local Ollama VLM (with Built-In Vision Heuristic Fallback) ---
function [sector, reason] = queryVLMNavigator(rgbFrame, sectorDepths, goalBearingDeg, distToGoal, modelName)
    % Encode current RGB frame as Base64 JPEG for the VLM API
    tmpFile = [tempname, '.jpg'];
    imwrite(imresize(rgbFrame, 0.5), tmpFile, 'Quality', 80);
    fid = fopen(tmpFile, 'rb');
    rawBytes = fread(fid, inf, '*uint8');
    fclose(fid);
    delete(tmpFile);
    b64Image = matlab.net.base64encode(rawBytes);

    prompt = sprintf([ ...
        'You are an autonomous urban drone navigator. The camera view is divided from left to right ', ...
        'into 5 sectors: 1 (Far Left, -28 deg), 2 (Left, -14 deg), 3 (Center, 0 deg), 4 (Right, +14 deg), 5 (Far Right, +28 deg). ', ...
        'Measured LiDAR/Depth clearance in meters for sectors [1..5] is [%.1f, %.1f, %.1f, %.1f, %.1f]. ', ...
        'The destination on the other side of the city is %.1f meters away at relative bearing %.0f degrees (+ is Right, - is Left). ', ...
        'Pick the best open street sector (1, 2, 3, 4, or 5) that avoids buildings/poles and progresses toward the goal. ', ...
        'Respond ONLY with valid JSON: {"sector": <int 1-5>, "reason": "<short 6-word explanation>"}'], ...
        sectorDepths(1), sectorDepths(2), sectorDepths(3), sectorDepths(4), sectorDepths(5), distToGoal, goalBearingDeg);

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
        sector = max(1, min(5, round(double(sec))));
        reason = string(parsed.reason);
    catch ME
        fprintf("VLM query failed, using fallback: %s\n", ME.message);
        % Automatic fallback if Ollama is not running yet: score sectors by depth clearance + goal alignment
        sectorAnglesDeg = [-28, -14, 0, 14, 28];
        scores = sectorDepths - 0.35 * abs(sectorAnglesDeg - goalBearingDeg);
        [~, sector] = max(scores);
        reason = "Open corridor toward city goal (Local Fallback)";
    end
end