import torch_fidelity
out = torch_fidelity.calculate_metrics(
            input1='samples',
            input2=None,
            fid_statistics_file='../JiT/fid_stats/jit_in256_stats.npz',
            cuda=False,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,

)
print(out)