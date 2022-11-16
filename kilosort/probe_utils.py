import numpy as np
import torch

dev = torch.device('cuda')
def template_centers(x_chan, y_chan):
    xmin, xmax, ymin, ymax = x_chan.min(), x_chan.max(), y_chan.min(), y_chan.max()
    dmin = np.median(np.diff(np.unique(y_chan)))
    yup = np.arange(ymin, ymax+.00001, dmin//2)

    yunq = np.unique(y_chan)
    mxc = np.NaN * np.ones(len(yunq))
    for j in range(len(yunq)):
        xc = x_chan[y_chan==yunq[j]]
        if len(xc)>1:
            mxc[j] = np.median(np.diff(np.sort(xc)))
    dminx = np.nanmedian(mxc)
    nx = np.round((xmax - xmin) / (dminx/2)) + 1
    xup = np.linspace(xmin, xmax, int(nx))
    return yup, xup, dmin, dminx

def nearest_chans(ys, y_chan, xs, x_chan, nC):
    ds = (ys - y_chan[:,np.newaxis])**2 + (xs - x_chan[:,np.newaxis])**2
    iC = np.argsort(ds, 0)[:nC]
    iC = torch.from_numpy(iC).to(dev)
    ds = np.sort(ds, 0)[:nC]
    return iC, ds

def chan_weights(x_chan, y_chan, nC=10, nC2=100):
    sig = 10
    nsizes = 5
    
    yup, xup, dmin, dminx = template_centers(x_chan, y_chan)
    [ys, xs] = np.meshgrid(yup, xup)
    ycup, xcup = ys.flatten(), xs.flatten()
    iC, ds = nearest_chans(ycup, y_chan, xcup, x_chan, nC)
    iC2, ds2 = nearest_chans(ycup, ycup, xcup, xcup, nC2)
    ds_torch = torch.from_numpy(ds).to(dev).float()
    weigh = torch.exp(- ds_torch.unsqueeze(-1) / (sig * (1+torch.arange(nsizes, device = dev)))**2)
    weigh = torch.permute(weigh, (2, 0, 1)).contiguous()
    weigh = weigh / (weigh**2).sum(1).unsqueeze(1)**.5
    return xcup, ycup, dmin, dminx, iC, iC2, weigh