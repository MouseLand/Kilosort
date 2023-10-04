from kilosort.spikedetect import extract_wPCA_wTEMP


def test_wpca_wtemp(bfile, saved_ops, torch_device):
    # Make sure extracting templates from data works, and with
    # differnt values than the default for n_templates, n_pcs
    ops = saved_ops.copy()
    ops['n_templates'] = 3
    ops['n_pcs'] = 5

    wPCA, wTEMP = extract_wPCA_wTEMP(ops, bfile, device=torch_device)
