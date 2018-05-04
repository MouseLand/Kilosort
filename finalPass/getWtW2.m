function wtw0 =  getWtW2(Params, W1, W2, utu0);

nt0            = size(W1,1);
W1(2*nt0-1, 1,1) = 0;
W2(2*nt0-1, 1,1) = 0;

fW1 = fft(W1, [], 1);
fW2 = conj(fft(W2, [], 1));

fW2 = permute(fW2, [1 3 2]);

fW = bsxfun(@times, fW1, fW2); 

%
wtw0 = real(ifft(fW, [], 1));

wtw0 = fftshift(wtw0, 1);

utu0 = permute(utu0, [3 1 2]);
wtw0 = bsxfun(@times, wtw0, utu0);


end