function data = shift_data(data, dy, ycoords, xcoords, iCovChans, sigDrift, Wrot)

shiftM = shift_matrix(dy, ycoords, xcoords, iCovChans, sigDrift, Wrot);
data   = shiftM * data;



