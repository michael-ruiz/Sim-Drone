function course = obstacleCourse(name)
%OBSTACLECOURSE Unmapped obstacles spawned in Unreal but NOT in city_map.mat.
%   course.parts:     struct array of primitives: shape ("box"|"cylinder"|"sphere"),
%                     center [x y z] (m), size [sx sy sz] (m), color [r g b]
%   course.hazards:   struct array of 2D hazard footprints for plotting and ground truth:
%                     name, points (Nx2 [x y] samples), zRange [zmin zmax] (m)
%
%   "box"        20 m orange box blocking the road on the first route leg (legacy test)
%   "crane_tree" tower crane on leg 1 with its hoist cable and hook hanging into the
%                flight path, and a street tree whose canopy overhangs the leg 2 lane
    arguments
        name (1,1) string {mustBeMember(name, ["none" "box" "crane_tree"])} = "crane_tree"
    end

    parts   = struct("shape", {}, "center", {}, "size", {}, "color", {});
    hazards = struct("name", {}, "points", {}, "zRange", {});

    switch name
        case "box"
            parts(end+1) = part("box", [40 2 10], [4 8 20], [1 0.4 0]);
            hazards(end+1) = hazard("box", boxPoints([40 2], [4 8]), [0 20]);

        case "crane_tree"
            yellow = [1 0.78 0];
            % Tower crane: mast on the east sidewalk, jib across the street above flight
            % altitude, hoist cable + hook hanging down to just below 12 m in the lane
            parts(end+1) = part("box",      [40  9 13],   [2 2 26],     yellow);        % mast
            parts(end+1) = part("box",      [40 -4 25],   [1.6 30 1.4], yellow);        % jib (y -19..11)
            parts(end+1) = part("box",      [40 12 23.5], [3 5 3],      [0.4 0.4 0.4]); % counterweight
            parts(end+1) = part("cylinder", [40 3 17.5], [0.2 0.2 14], [0.1 0.1 0.1]); % hoist cable
            parts(end+1) = part("box",      [40 3 9.8],  [1.2 1.2 1.5], yellow);       % hook block
            hazards(end+1) = hazard("crane mast", boxPoints([40 9], [2 2]), [0 26]);
            hazards(end+1) = hazard("crane cable", boxPoints([40 3], [1.2 1.2]), [9 24.5]);

            % Street tree on the south sidewalk of leg 2; canopy overhangs the lane
            parts(end+1) = part("cylinder", [69 60 5],  [1 1 10], [0.35 0.22 0.1]);     % trunk
            parts(end+1) = part("sphere",   [69 60 13], [11 11 9], [0.15 0.45 0.12]);   % canopy
            hazards(end+1) = hazard("tree canopy", discPoints([69 60], 5.5), [8.5 17.5]);
    end
    course = struct("name", name, "parts", parts, "hazards", hazards);
end

function p = part(shape, center, sz, color)
    p = struct("shape", string(shape), "center", center, "size", sz, "color", color);
end

function h = hazard(name, points, zRange)
    h = struct("name", string(name), "points", points, "zRange", zRange);
end

function pts = boxPoints(c, sz)
    [gx, gy] = ndgrid(c(1) + (-sz(1)/2:0.5:sz(1)/2), c(2) + (-sz(2)/2:0.5:sz(2)/2));
    pts = [gx(:), gy(:)];
end

function pts = discPoints(c, r)
    [gx, gy] = ndgrid(-r:0.5:r, -r:0.5:r);
    in = gx.^2 + gy.^2 <= r^2;
    pts = c + [gx(in), gy(in)];
end
