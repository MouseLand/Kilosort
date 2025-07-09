Hardware recommendations
========================
We start with a list of recommendations for a standard Neuropixels recording (384 channels, <3 hours), and then we continue with a list of modifications depending on how your recordings differ from this standard one. Note it is possible to run Kilosort4 on the CPU, but this should only be done for testing purposes as it is much slower than even a cheap GPU (i.e. RTX 3060 12GB). 

Recommendations for Neuropixels/large recordings
------------------------------------------------

GPU
^^^
We only provide support for Nvidia GPUs since those have the most robust support with PyTorch. We recommend at least 12GB of RAM, which is available on relatively cheap GPUs. Speed increases with a newer generation and/or larger GPU. The Nvidia numbering scheme is N0X0, where N is the generation and X is the tier level which increases with GPU size. For example, GTX 1080 Ti is generation "1" and tier "8", and the Ti suffix indicates a slightly bigger/better card. Current cards are generation "5". An RTX 4070 is a good option with 12GB, 16GB, and "Ti" versions available. Note that the "professional" GPUs are not much faster for Kilosort processing despite being many times more expensive.

SSD
^^^
Check for read/write speed. A good SATA SSD is around 500MB/s, and PCI-based (NVMe) SSDs are usually a lot better. You shouldn't need anything faster than 500MB/s, but you may want to have generous capacity for the purpose of batch processing many datasets. A typical workflow is to copy datasets you want processed to the SSD, then run Kilosort on them, then run Phy on the local copy of the data with the Kilosort results. Once you are happy with the sorting, you would typically free up the local copy of the data to make space for the next round of spike sorting. **Using a HDD is not recommended** as it will slow down sorting substantially. In our experience, transfering data to and from a SSD adds less time than sorting on a HDD.

Other
^^^^^
At least 32GB of RAM is recommended, and an 8-core CPU. A faster CPU is not likely to noticeably affect sorting time. Note, however, that the amount of memory needed will depend on the number of spikes present in the data, so 64GB of RAM is preferable. For very long recordings, like 6 or more hours of Neuropixels data, more RAM will likely be needed.


Additional recommendations
-----------------------------------------------------

Longer recordings (more than 6 hours)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Assuming an equal number of channels, a longer dataset may require more RAM (128GB or more). Kilosort splits the data into small batches (default = 2s) and sorts each separately. However, a few steps (like spike extraction and clustering) require keeping some information from each batch in RAM before it is used or saved at the end of the run through the entire dataset. For example, two ~13 hour Neuropixels recordings from different labs each required around 300 GB of RAM.

High channel count
^^^^^^^^^^^^^^^^^^
This will require more GPU RAM. If you are running into GPU limitations for this kind of data, try reducing the batch size (to 1s or possibly 0.5s). For densely spaced channels, you can also reduce the memory requirements by keeping dminx set at the default (32um) or a similar-valued multiple of the contact spacing. Denser spacing will oversample the recorded space, which is typically not necessary.

Very few channels
^^^^^^^^^^^^^^^^^
Kilosort4 was designed with Neuropixels-like probes in mind and may fail for single-channel data, or low-channel data like a single tetrode. Typically this happens when there are not enough spikes detected in a given batch to generate universal templates. If you run into this issue, try increasing batch size (to around 10s to start, or larger if hardware allows) to include more data. You can also try stacking channels from multiple recordings, as long as they are the same length (with a small amount of padding if needed) and you expect the spiking activity across recordings to be similar.
