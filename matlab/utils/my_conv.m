function Smooth = my_conv(S1, sig)
% fast gaussian smoothing, ensures that the filter on the edges is still unit norm
% works on GPU as well
% works on second axis by default

NN = size(S1,1);
NT = size(S1,2);

dt = -4*sig:1:4*sig;
gaus = exp( - dt.^2/(2*sig^2));
gaus = gaus'/sum(gaus);

% Norms = conv(ones(NT,1), gaus, 'same');
%Smooth = zeros(NN, NT);
%for n = 1:NN
%    Smooth(n,:) = (conv(S1(n,:)', gaus, 'same')./Norms)';
%end

Smooth = filter(gaus, 1, [S1' ones(NT,1); zeros(ceil(4*sig), NN+1)]);
Smooth = Smooth(1+ceil(4*sig):end, :);
Smooth = Smooth(:,1:NN) ./ (Smooth(:, NN+1) * ones(1,NN));

Smooth = Smooth';
