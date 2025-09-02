function robotShooter(v,theta,pencolor)
%physical constants and inputs
xmax = 20;
ymax = 10;

ay = -9.8;  % m/s/s

vdisturbance = v * (rand()-0.5)*2  * 0.075;
v = v + vdisturbance;

thetadisturbance = theta * (rand()-0.5) * 2 * 0.075;
theta = theta+thetadisturbance;
vx = v * cosd(theta);
vy = v * sind(theta);
x0 = 0;  % m
y0 = 1.5; % m

% define postion and height of target

xtarget = 15;       % m   ... location of front face of target
ytarget = 1.0;      % m   ... height of target
depthTarget = 1.0;  % m  .... depth of target

% draw target
hold on
targetlocationx = [xtarget, xtarget, xtarget + depthTarget, xtarget + depthTarget];
targetlocationy = [0, ytarget, ytarget, 0];

plot(targetlocationx, targetlocationy)
axis([0, xmax, 0 , ymax])


% define trajectory path

t = 0:.01:10;

x = x0 + vx *t; 
y = y0 + vy *t + 0.5 * ay * t.*t;



for n=1:length(y)
    plot(x(1:n),y(1:n),'Color',pencolor)
    pause(0.05)
    if(y(n) < 0 || x(n) > xmax)
        break
    end
end
hold off









