def sstows_conversion(pt, ws_lim):
    """ Convert from [0,1] search space limits to franka robot workspace limits"""
    lim = ws_lim[:,1]-ws_lim[:,0]
    new_pt = [pt[i]*lim[i]+ws_lim[i,0] for i in range(len(lim))]
    return new_pt

def wstoss_conversion(pt, ws_lim):
    """ Convert from franka robot workspace limits to  [0,1] search space limits"""
    lim = ws_lim[:,1]-ws_lim[:,0]
    new_pt =[(pt[i]-ws_lim[i,0])/lim[i] for i in range(len(lim))]   
    return new_pt

def ws_conversion(pt, in_dim, out_dim):
    """ Convert from [0,1] search space limits to franka robot workspace limits"""
    ilim= in_dim[:,1]-in_dim[:,0]
    olim = out_dim[:,1]-out_dim[:,0]
    new_pt = [(pt[i]-in_dim[i,0])/ilim[i]*olim[i]+out_dim[i,0] for i in range(len(ilim))]
    return new_pt
