function A=dctbasis(N,D)
%A = dctbasis(N,D)
%Computes the elementary basis functions for 1D and 2D DCT
%
%Input: 
% N - basis size
% D - basis dimension, 1 for 1D, 2 for 2D 
%
%Output: 
% A - array containing the elementary basis functions
%
%Note: 
% For 1D the output array is of size NxN with columns containing the 
% elementary functions.
% For 2D the output array is of size NxNxNxN, and the elementary function
% corresponding to (i,j) DCT coefficient can be obtained as A(:,:,i,j)   
%
%Example:
% A=dctbasis(8,2); %8x8 DCT basic functions, display it with e.g.
%                  %image_show(128*(1+A(:,:,2,7)),256,16,'DCT-2-7');
% A=dctbasis(4,1); %4x1 DCT basic functions
% A=dctbasis(4,2);


i=0:N-1;
j=0:N-1;
[I,J]=meshgrid(i,j);
A=sqrt(2/N)*cos(((2.*I+1).*J*pi)/(N*2));
A(1,:)=A(1,:)./sqrt(2);
A=A';   
if D==1
elseif D==2
 B=zeros(N,N,N,N);
 for i=1:N
  for j=1:N
   B(:,:,i,j)=A(:,i)*A(:,j)';
   %max(max(B(:,:,i,j)))-min(min(B(:,:,i,j)))   
  end;
 end;    
 A=B;
end;