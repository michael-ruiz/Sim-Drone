%% Build a 2D height map of USCityBlock from top-down depth shots
% Flies a downward-facing camera over a grid of points at 150 m, projects every
% depth pixel into world X/Y (same frame as the multirotor's NED x/y), and keeps
% the tallest surface per 1 m cell. Saves city_map.mat + city_map.png.
%
% Conventions (measured with a probe): pitch -pi/2 looks straight down; in that
% view image-up = +X and image-right = +Y; depth is planar uint8 meters.
clear; close all;

shotAlt     = 150;                 % Camera altitude (m); must stay < 255 (uint8 depth)
shotX       = -40:60:140;          % Shot centers along X (m)
shotY       = -60:60:120;          % Shot centers along Y (m)
shotRadius  = 35;                  % Only keep points within +/-35 m of each shot center
cellSize    = 1.0;                 % Map resolution (m)
holdSteps   = 6;                   % Sim steps to hold each pose before reading
focal       = 450;
imgSize     = [480 640];
optCenter   = [320 240];

[SX, SY] = ndgrid(shotX, shotY);
shots = [SX(:), SY(:)];
nShots = size(shots, 1);

world = sim3d.World(Scene="USCityBlock", ...
    Output=@(w) placeCamera(w, shots, shotAlt, holdSteps), ...
    Update=@(w) captureShot(w, holdSteps));
cam = sim3d.sensors.Camera(ActorName="MapCamera", ImageSize=imgSize, ...
    FocalLength=[focal focal], OpticalCenter=optCenter, EnableDepthOutput=true);
cam.Translation = [shots(1,:), shotAlt];
cam.Rotation    = [0 -pi/2 0];
add(world, cam);
world.UserData.k     = 0;
world.UserData.Depth = {};
world.UserData.RGB   = {};

try
    run(world, 0.1, 0.1 * holdSteps * nShots);
catch ME
    delete(world);
    rethrow(ME);
end
depthShots = world.UserData.Depth;
rgbShots   = world.UserData.RGB;
delete(world);

%% Project all shots into a shared world grid
xEdges = (min(shotX) - shotRadius):cellSize:(max(shotX) + shotRadius);
yEdges = (min(shotY) - shotRadius):cellSize:(max(shotY) + shotRadius);
nx = numel(xEdges); ny = numel(yEdges);

[uu, vv] = meshgrid(1:imgSize(2), 1:imgSize(1));
allIdx = []; allH = []; allRGB = [];
for k = 1:numel(depthShots)
    d = double(depthShots{k});
    X = shots(k,1) + (optCenter(2) - vv) .* d / focal;   % image up    -> +X
    Y = shots(k,2) + (uu - optCenter(1)) .* d / focal;   % image right -> +Y
    keep = abs(X - shots(k,1)) <= shotRadius & abs(Y - shots(k,2)) <= shotRadius;
    ix = round((X(keep) - xEdges(1)) / cellSize) + 1;
    iy = round((Y(keep) - yEdges(1)) / cellSize) + 1;
    inMap = ix >= 1 & ix <= nx & iy >= 1 & iy <= ny;
    hKeep = shotAlt - d(keep);
    allIdx = [allIdx; sub2ind([nx ny], ix(inMap), iy(inMap))]; %#ok<AGROW>
    allH   = [allH;   hKeep(inMap)];                           %#ok<AGROW>
    rgb = reshape(double(rgbShots{k}), [], 3);
    rgbKeep = rgb(keep(:), :);
    allRGB = [allRGB; rgbKeep(inMap, :)];                      %#ok<AGROW>
end

% Tallest surface per cell (conservative); unobserved cells -> NaN
heightMap = accumarray(allIdx, allH, [nx*ny 1], @max, NaN);
heightMap = reshape(heightMap, nx, ny);
colorMap = zeros(nx*ny, 3);
for c = 1:3
    colorMap(:, c) = accumarray(allIdx, allRGB(:, c), [nx*ny 1], @mean, 0);
end
colorMap = uint8(reshape(colorMap, nx, ny, 3));

% Fill small unobserved holes from their neighbors
holes = isnan(heightMap);
filled = movmax(movmax(heightMap, 3, 1, "omitnan"), 3, 2, "omitnan");
heightMap(holes) = filled(holes);
fprintf("Map %d x %d cells (%.0f%% observed, %.0f%% after hole fill)\n", nx, ny, ...
        100 * mean(~holes(:)), 100 * mean(~isnan(heightMap(:))));

xCenters = xEdges; yCenters = yEdges;
save("city_map.mat", "heightMap", "colorMap", "xCenters", "yCenters", "cellSize");

%% Plot for inspection (X up, Y right, matching the top-down camera view)
fig = figure(Color="w", Position=[100 100 1200 560]);
tiledlayout(fig, 1, 2, TileSpacing="compact");
ax1 = nexttile;
image(ax1, yCenters, xCenters, colorMap); axis(ax1, "xy", "image");
xlabel(ax1, "Y (m)"); ylabel(ax1, "X (m)"); title(ax1, "Top-Down RGB Mosaic");
ax2 = nexttile;
imagesc(ax2, yCenters, xCenters, heightMap); axis(ax2, "xy", "image");
colormap(ax2, "turbo"); cb = colorbar(ax2); ylabel(cb, "Height (m)");
xlabel(ax2, "Y (m)"); ylabel(ax2, "X (m)"); title(ax2, "Surface Height Map");
exportgraphics(fig, "city_map.png", Resolution=120);
fprintf("Saved city_map.mat and city_map.png\n");

%% --- Callbacks ---
function placeCamera(w, shots, alt, holdSteps)
    k = w.UserData.k + 1;
    w.UserData.k = k;
    s = min(size(shots, 1), ceil(k / holdSteps));
    w.Actors.MapCamera.Translation = [shots(s,:), alt];
    w.Actors.MapCamera.Rotation    = [0 -pi/2 0];
end

function captureShot(w, holdSteps)
    if mod(w.UserData.k, holdSteps) ~= 0, return; end
    [rgb, depth] = read(w.Actors.MapCamera);
    w.UserData.Depth{end+1} = depth;
    w.UserData.RGB{end+1}   = rgb;
end
