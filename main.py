from mpi4py import MPI
import numpy as np
from tqdm import tqdm

def _read_input_file(filename):
    with open(filename, 'r') as f:
        T0, T, biasf = map(float, f.readline().split())
        tmin, tmax = map(int, f.readline().split())
        gridmin1, gridmax1, griddiff1 = map(float, f.readline().split())
        gridmin2, gridmax2, griddiff2 = map(float, f.readline().split())
        gridmin3, gridmax3, griddiff3 = map(float, f.readline().split())
        gridmin4, gridmax4, griddiff4 = map(float, f.readline().split())
        w_cv, w_hill, sigma = map(float, f.readline().split())
    return T0, T, biasf, tmin, tmax, w_cv, w_hill, sigma

def _read_colvar(filename):
    data = np.loadtxt(filename)
    return data[:,1], data[:,3], data[:,5], data[:,7]

def _read_hills(filename):
    data = np.loadtxt(filename)
    hill1 = data[:,1]
    hill2 = data[:,3]
    hill3 = data[:,5]
    hill4 = data[:,7]
    height = data[:,9] * 0.239006  # convert kJ/mol to kcal/mol
    return hill1, hill2, hill3, hill4, height

def _compute_vbias_segment(s1_seg, s2_seg, s3_seg, s4_seg, hill1, hill2, hill3, hill4, height, w_cv, w_hill, sigma, alpha, rank):
    nsteps = len(s1_seg)
    vbias = np.zeros(nsteps)
    dsq = sigma * sigma

    iterator = tqdm(enumerate(range(nsteps)), total=nsteps, desc=f"[Rank {rank}] calculating vbias", disable=(rank != 0))

    for i_local, i in iterator:
        i_global = i
        mtd_max = int((i_global+1) * w_cv / w_hill)
        if mtd_max == 0:
            vbias[i_local] = 0.0
            continue

        diff1 = s1_seg[i_local] - hill1[:mtd_max]
        diff2 = s2_seg[i_local] - hill2[:mtd_max]
        diff3 = s3_seg[i_local] - hill3[:mtd_max]
        diff4 = s4_seg[i_local] - hill4[:mtd_max]

        # Periodic wrapping
        diff1 = np.where(diff1 > 3.14, diff1 - 6.28, diff1)
        diff1 = np.where(diff1 < -3.14, diff1 + 6.28, diff1)
        diff2 = np.where(diff2 > 3.14, diff2 - 6.28, diff2)
        diff2 = np.where(diff2 < -3.14, diff2 + 6.28, diff2)
        diff3 = np.where(diff3 > 3.14, diff3 - 6.28, diff3)
        diff3 = np.where(diff3 < -3.14, diff3 + 6.28, diff3)
        diff4 = np.where(diff4 > 3.14, diff4 - 6.28, diff4)
        diff4 = np.where(diff4 < -3.14, diff4 + 6.28, diff4)

        g1 = 0.5 * (diff1 ** 2)
        g2 = 0.5 * (diff2 ** 2)
        g3 = 0.5 * (diff3 ** 2)
        g4 = 0.5 * (diff4 ** 2)
        e = np.exp(-(g1 + g2+ g3 +g4) / dsq)
        vbias[i_local] = np.sum((height[:mtd_max] / alpha) * e)

    return vbias

def calculate_vbias(input_file, colvar_file, hills_file, output_file="vbias.dat"):
    """
    Calculates the bias potential from simulation data using MPI.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        T0, T, biasf, tmin, tmax, w_cv, w_hill, sigma = _read_input_file(input_file)
        s1_all, s2_all, s3_all, s4_all = _read_colvar(colvar_file)
        hill1, hill2, hill3, hill4, height = _read_hills(hills_file)
    else:
        T0 = T = biasf = tmin = tmax = w_cv = w_hill = sigma = None
        s1_all = s2_all =s3_all=s4_all= hill1 = hill2 = hill3 = hill4 = height = None

    # Broadcast data to all processes
    T0 = comm.bcast(T0, root=0)
    T = comm.bcast(T, root=0)
    biasf = comm.bcast(biasf, root=0)
    w_cv = comm.bcast(w_cv, root=0)
    w_hill = comm.bcast(w_hill, root=0)
    sigma = comm.bcast(sigma, root=0)
    s1_all = comm.bcast(s1_all, root=0)
    s2_all = comm.bcast(s2_all, root=0)
    s3_all = comm.bcast(s3_all, root=0)
    s4_all = comm.bcast(s4_all, root=0)
    hill1 = comm.bcast(hill1, root=0)
    hill2 = comm.bcast(hill2, root=0)
    hill3 = comm.bcast(hill3, root=0)
    hill4 = comm.bcast(hill4, root=0)
    height = comm.bcast(height, root=0)

    alpha = (T + (biasf - 1) * T) / ((biasf - 1) * T)
    nsteps = len(s1_all)
    indices = np.arange(nsteps)
    local_indices = np.array_split(indices, size)[rank]
    
    s1_seg = s1_all[local_indices]
    s2_seg = s2_all[local_indices]
    s3_seg = s3_all[local_indices]
    s4_seg = s4_all[local_indices]

    vbias_local = _compute_vbias_segment(s1_seg, s2_seg, s3_seg, s4_seg, hill1, hill2, hill3, hill4, height, w_cv, w_hill, sigma, alpha, rank)

    all_vbias = comm.gather(vbias_local, root=0)
    all_indices = comm.gather(local_indices, root=0)

    if rank == 0:
        full_vbias = np.zeros(nsteps)
        for inds, vals in zip(all_indices, all_vbias):
            full_vbias[inds] = vals
        with open(output_file, 'w') as f:
            for i, val in enumerate(full_vbias, 1):
                f.write(f"{i:10d} {val:16.8f}\n")
        print(f"[Rank 0] vbias written to {output_file}")
