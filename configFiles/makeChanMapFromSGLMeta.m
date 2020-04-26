

function cm = makeChanMapFromSGLMeta(m)

shankMap = m.snsShankMap; 

% nCh = str2num(m.acqApLfSy);
nCh = m.nChans-1;

chanMap = [1:nCh(1)]'; 
chanMap0ind = chanMap-1;
connected = true(size(chanMap)); 

shankSep = 250; 
rowSep = 15; 
colSep = 32;

openParen = find(shankMap=='('); 
closeParen = find(shankMap==')'); 
for c = 1:nCh(1)
    thisMap = shankMap(openParen(c+1)+1: closeParen(c+1)-1); 
    thisMap(thisMap==':') = ',';
    n = str2num(thisMap); 
    xcoords(c) = (n(1)-1)*shankSep + (n(2)-1)*colSep; 
    ycoords(c) = (n(3)-1)*rowSep; 
end

cm = struct();
cm.chanMap = chanMap; 
cm.chanMap0ind = chanMap0ind;
cm.xcoords = xcoords'; 
cm.ycoords = ycoords'; 
cm.connected = connected;
[~,name] = fileparts(m.imRoFile); 
cm.name = name;