% mexGPUall. For these to complete succesfully, you need to configure the
% Matlab GPU library first (see README files for platform-specific
% information)

    mexcuda -largeArrayDims mexThSpkPC.cu
    mexcuda -largeArrayDims mexGetSpikes2.cu
    mexcuda -largeArrayDims mexMPnu8.cu

    mexcuda -largeArrayDims mexSVDsmall2.cu
    mexcuda -largeArrayDims mexWtW2.cu
    mexcuda -largeArrayDims mexFilterPCs.cu
    mexcuda -largeArrayDims mexClustering2.cu
    mexcuda -largeArrayDims mexDistances2.cu


%    mex -largeArrayDims mexMPmuFEAT.cu
%    mex -largeArrayDims mexMPregMU.cu
%    mex -largeArrayDims mexWtW2.cu

% If you get uninterpretable errors, run again with verbose option -v, i.e. mexcuda -v largeArrayDims mexGetSpikes2.cu


