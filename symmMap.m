%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Josh Trzasko
% Center for Advanced Imaging Research
% Mayo Clinic 
% 4/7/2007
%   Modified for horizontal and vertical symmetry.
%   -Jon Dattorro July 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SM2 = symmMap(N0,P)
N = ceil(N0*sqrt(2)/2)*2;
sampleMap = zeros(N,N);
slice     = zeros(N/2+1,2);
rot_slice = zeros(N/2+1,2);
dTheta = pi/P;
for l=1:N/2+1
    x = N/2+1;
    y = l;
    slice(l,1) = x;
    slice(l,2) = y;    
end
for p = 0:2*P
   theta = p*dTheta;
   for n = 1:N/2
       x = slice(n,1)-N/2-1;
       y = slice(n,2)-N/2-1;
       x2 = round( x*cos(theta)+y*sin(theta)+N/2+1);
       y2 = round(-x*sin(theta)+y*cos(theta)+N/2+1);
       sampleMap(y2,x2) = 1;
    end
end
SM = sampleMap(1:N,1:N);
SM(N/2+1,N/2+1) = 1;

Nc  = N/2;
N0c = N0/2;

SM2 = SM(Nc-N0c+1:Nc+N0c, Nc-N0c+1:Nc+N0c);

% make vertically and horizontally symmetric
[N1,N1] = size(SM2);
Xi = fliplr(eye(N1-1));
SM2 = round((SM2 + [   SM2(1,1)        SM2(1,2:N1)*Xi;
                    Xi*SM2(2:N1,1)  Xi*SM2(2:N1,2:N1)*Xi])/2);
