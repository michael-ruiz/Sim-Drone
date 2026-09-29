%% 1. Initialize Live Display (RGB with 5-Sector HUD + Depth)
clear; close all;

fig = figure(Name="Unreal Engine Vector-Field Closed-Loop Orbit", ...
             Color="w", Position=[60 80 1200 500]);
tLayout = tiledlayout(fig, 1, 2, TileSpacing="compact");

axRGB = nexttile(tLayout, 1);
hRGB = imshow(uint8(zeros(480, 640, 3)), Parent=axRGB);
hold(axRGB, "on");

colEdges = round(linspace(40, 600, 6));
hBoxes = gobjects(1, 5);
for i = 1:5
    hBoxes(i) = rectangle(axRGB, ...
        Position=[colEdges(i), 130, colEdges(i+1)-colEdges(i), 105], ...
        EdgeColor="g", LineWidth=1.5, LineStyle="--");
end
hStatus = text(axRGB, 15, 35, "INITIALIZING...", Color="g", ...
               FontWeight="bold", FontSize=11, BackgroundColor="k");
title(axRGB, "Onboard RGB + 5-Sector Repulsive Field HUD");

axDepth = nexttile(tLayout, 2);
hDepth = imagesc(axDepth, zeros(480, 640), [0 40]);
axis(axDepth, "image", "off");
colormap(axDepth, "turbo");
cb = colorbar(axDepth);
ylabel(cb, "Depth (m)");
title(axDepth, "Live Depth Map (m)");

%% 2. Set Up UAV Guidance Model (NED Frame)
model = multirotor;
s = state(model);       % 13x1: [x;y;z; vx;vy;vz; yaw;pitch;roll; p;q;r; thrust]
u = control(model);
e = environment(model);

s(1:3) = [0; 0; -2];
hoverThrust = model.Configuration.Mass * e.Gravity;
s(13) = hoverThrust;
u.Thrust = hoverThrust;

%% 3. Create Unreal Engine World & Camera
sampleTime = 1/20; % 20 Hz control loop
stopTime   = 32;   % 32-second flight

world = sim3d.World(Scene="USCityBlock", ...
                    Output=@(w) stepUAVPhysics(w, sampleTime), ...
                    Update=@(w) processDepthAndControl(w, sampleTime, hRGB, hDepth, hStatus, hBoxes, colEdges));

cam = sim3d.sensors.Camera(ActorName="DroneCamera", ...
                           ImageSize=[480 640], ...
                           FocalLength=[450 450], ...
                           OpticalCenter=[320 240], ...
                           EnableDepthOutput=true);
cam.Translation = [0 0 2];
add(world, cam);

world.UserData.Time        = 0;
world.UserData.Model       = model;
world.UserData.State       = s;
world.UserData.Control     = u;
world.UserData.Env         = e;
world.UserData.HoverThrust = hoverThrust;
world.UserData.History     = [];

%% 4. Run Co-Simulation
try
    run(world, sampleTime, stopTime);
catch ME
    delete(world);
    rethrow(ME);
end

histData = world.UserData.History;
delete(world);

%% 5. Post-Flight Plots: 3D Trajectory + 2D Top-Down XY View
if ~isempty(histData)
    figure(Name="Post-Flight Trajectory Analysis", Color="w", Position=[100 100 1150 480]);
    tiledlayout(1, 2, TileSpacing="compact");

    ax1 = nexttile(1);
    plot3(ax1, histData(:,1), histData(:,2), histData(:,3), "b-", LineWidth=1.8);
    hold(ax1, "on"); grid(ax1, "on"); axis(ax1, "equal");
    avoidPts = histData(:,4) == 1;
    scatter3(ax1, histData(avoidPts,1), histData(avoidPts,2), histData(avoidPts,3), ...
             30, "r", "filled");
    xlabel(ax1, "X (m)"); ylabel(ax1, "Y (m)"); zlabel(ax1, "Altitude (m)");
    title(ax1, "3D Closed-Loop Flight Path");
    legend(ax1, "Normal Mission", "Obstacle Repulsion Active", Location="best");
    view(ax1, [-40 30]);

    ax2 = nexttile(2);
    th = linspace(0, 2*pi, 200);
    plot(ax2, 12*(1 - cos(th)), 12*sin(th), "k--", LineWidth=1.2);
    hold(ax2, "on"); grid(ax2, "on"); axis(ax2, "equal");
    plot(ax2, histData(:,1), histData(:,2), "b-", LineWidth=1.8);
    scatter(ax2, histData(avoidPts,1), histData(avoidPts,2), 30, "r", "filled");
    xlabel(ax2, "X (m)"); ylabel(ax2, "Y (m)");
    title(ax2, "Top-Down XY View (Planned Circle vs. Actual Path)");
    legend(ax2, "Nominal Circle", "Actual Flight Path", "Repulsion Active", Location="best");
end

%% --- Callback 1: Step Multirotor Physics & Move Gimbal-Stabilized Camera ---
function stepUAVPhysics(world, dt)
    ud = world.UserData;
    ud.Time = ud.Time + dt;

    [~, sHistory] = ode45(@(~, x) derivative(ud.Model, x, ud.Control, ud.Env), ...
                          [0 dt], ud.State);
    s = sHistory(end, :)';
    ud.State = s;
    world.UserData = ud;

    posUnreal = [s(1), s(2), max(0.5, -s(3))];
    eulUnreal = [0, 0, s(7)];

    world.Actors.DroneCamera.Translation = posUnreal;
    world.Actors.DroneCamera.Rotation    = eulUnreal;
end

%% --- Callback 2: 5-Sector Repulsive Potential Field Controller ---
function processDepthAndControl(world, ~, hRGB, hDepth, hStatus, hBoxes, colEdges)
    [rgbFrame, depthFrame] = read(world.Actors.DroneCamera);
    depthFrame = double(depthFrame);

    ud  = world.UserData;
    t   = ud.Time;
    s   = ud.State;
    u   = ud.Control;
    pos = s(1:3); % [x; y; z_down] in NED
    vel = s(4:6);
    yaw = s(7);

    % --- 1. PERCEIVE: 2nd-Percentile Depth in 5 Horizon Strips ---
    sectorDepths = zeros(1, 5);
    for i = 1:5
        strip = depthFrame(130:235, colEdges(i):colEdges(i+1));
        sectorDepths(i) = prctile(strip(:), 2);
    end

    % Relative bearing angles of the 5 camera sectors (Left to Right in radians)
    sectorAngles = deg2rad([-26, -13, 0, 13, 26]);
    repulseDist  = 8.5; % Start pushing away when wall/pole is closer than 8.5m

    % --- 2. DECIDE: Vertical Phases vs. Potential-Field Circle ---
    zPeak   = -22; % 22m peak climb
    zCircle = -12; % 12m orbit altitude
    radius  = 12;  % 12m circle radius

    if t <= 4
        % Phase 1: Vertical Climb
        vWorldCmd  = -0.8 * pos(1:2);
        zTargetNED = zPeak;
        yawTarget  = 0;
        modeFlag   = 0;
        statusStr  = "MISSION: VERTICAL CLIMB";
        statusCol  = "g";

    elseif t <= 7
        % Phase 2: Vertical Descent
        vWorldCmd  = -0.8 * pos(1:2);
        zTargetNED = zCircle;
        yawTarget  = 0;
        modeFlag   = 0;
        statusStr  = "MISSION: VERTICAL DESCENT";
        statusCol  = "g";

    else
        % Phase 3: Circle Orbit + 2D Repulsive Vector Field
        theta = min(2*pi, 2 * pi * ((t - 7) / 23));

        % Attractive target point + feedforward tangent velocity along the circle
        targetXY  = [radius * (1 - cos(theta));
                     radius * sin(theta)];
        tangentXY = [sin(theta); cos(theta)] * (2 * pi * radius / 23);

        vAttract = 0.85 * (targetXY - pos(1:2)) + tangentXY;

        % Sum repulsive vectors from any sector closer than repulseDist (8.5m)
        vRepulse = [0; 0];
        modeFlag = 0;
        for i = 1:5
            d = sectorDepths(i);
            if d < repulseDist
                modeFlag = 1;
                % World bearing of the obstacle in sector i
                obsBearing = yaw + sectorAngles(i);
                obsDirWorld = [cos(obsBearing); sin(obsBearing)];

                % Push directly away from obstacle (stronger as d gets smaller)
                pushMag = 4.5 * ((repulseDist - d) / repulseDist);
                vRepulse = vRepulse - pushMag * obsDirWorld;
            end
        end

        % Combine attractive mission pull + obstacle repulsion
        vWorldCmd = vAttract + vRepulse;
        speed = norm(vWorldCmd);
        if speed > 3.8
            vWorldCmd = vWorldCmd * (3.8 / speed);
        end

        % Point camera along the circle tangent / travel direction so it turns around at X = 24m
        if norm(vAttract) > 0.3
            yawTarget = atan2(vAttract(2), vAttract(1));
        else
            yawTarget = yaw;
        end

        % Climb slightly (up to 3m) when actively repulsed by close obstacles
        if modeFlag == 1
            zTargetNED = zCircle - 2.0;
            statusStr  = sprintf("WALL REPULSION ACTIVE (Min Depth: %.1f m)", min(sectorDepths));
            statusCol  = "r";
        else
            zTargetNED = zCircle;
            statusStr  = sprintf("MISSION: CIRCLE (%.0f deg) | Min Depth: %.1f m", ...
                                 rad2deg(theta), min(sectorDepths));
            statusCol  = "g";
        end
    end

    % --- 3. ACTUATE: Convert World Velocity to Body Roll/Pitch/YawRate ---
    yawErr = atan2(sin(yawTarget - yaw), cos(yawTarget - yaw));
    u.YawRate = max(-2.0, min(2.0, 2.8 * yawErr));

    vForwardCmd =  cos(yaw) * vWorldCmd(1) + sin(yaw) * vWorldCmd(2);
    vRightCmd   = -sin(yaw) * vWorldCmd(1) + cos(yaw) * vWorldCmd(2);

    vForwardCurr =  cos(yaw) * vel(1) + sin(yaw) * vel(2);
    vRightCurr   = -sin(yaw) * vel(1) + cos(yaw) * vel(2);

    maxTilt = deg2rad(18);
    u.Pitch = max(-maxTilt, min(maxTilt, -0.22 * (vForwardCmd - vForwardCurr)));
    u.Roll  = max(-maxTilt, min(maxTilt,  0.22 * (vRightCmd   - vRightCurr)));

    altErr   = pos(3) - zTargetNED;
    vzUpCurr = -vel(3);
    u.Thrust = max(0, ud.HoverThrust + ud.Model.Configuration.Mass * (6.0 * altErr - 3.8 * vzUpCurr));

    ud.History = [ud.History; pos(1), pos(2), -pos(3), modeFlag];
    ud.Control = u;
    world.UserData = ud;

    % --- 4. UPDATE HUD & DEPTH DISPLAY ---
    if isvalid(hRGB)
        set(hRGB, CData=rgbFrame);
        set(hDepth, CData=depthFrame);
        set(hStatus, String=sprintf("t = %.1f s | %s", t, statusStr), Color=statusCol);

        for i = 1:5
            if (t > 7) && (sectorDepths(i) < repulseDist)
                set(hBoxes(i), EdgeColor="r", LineStyle="-", LineWidth=2.5);
            else
                set(hBoxes(i), EdgeColor="g", LineStyle="--", LineWidth=1.5);
            end
        end
        drawnow limitrate;
    end
end