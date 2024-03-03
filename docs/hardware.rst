Hardware recommendations
==================================
We start with a list of recommendations for a standard Neuropixels recording (384 channels, <3 hours), and then we continue with a list of modifications depending on how your recordings differ from this standard one. Note it is possible to run Kilosort4 on the CPU, but this should only be done for testing purposes as it is much slower than even a cheap GPU (i.e. RTX 3060 12GB). 

Recommendations for Neuropixels/large recordings
-------------------------------------------------------------

#. GPU: only Nvidia GPUs are supported via pytorch. We recommend at least 8GB of RAM, which can also be had in relatively cheap GPU. Speed increases with a newer generation and larger GPU. The Nvidia numbering scheme is N0X0, where N is the generation and X is the tier level which increases with GPU size. For example, GTX 1080 Ti is generation "1" and tier "8", and the Ti suffix indicates a slightly bigger/better card. Current cards are generation "4" and RTX 4070 is probably a sweet spot. Note that the "professional" GPUs are not much faster for Kilosort processing despite being many times more expensive. 
#. SSD: check for read/write speed. Any good SATA SSD is around 500MB/s, and PCI-based SSD are usually a lot better. You shouldn't need anything faster than 500MB/s, but you might want to have generous capacity for the purpose of batch processing of many datasets. A typical workflow is to copy datasets you want processed to the SSD, then run Kilosort on them, then run Phy on the local copy of the data with the Kilosort results. Once you are happy with the sorting, you would typically free up the local copy of the data to make space for the next round of spike sorting. 
#. rest of the system: 32GB of RAM are recommended and an 8-core CPU. There is little performance to be gained by increasing these.

Additional recommendations
-----------------------------------------------------

#. Same number of channels, but longer recordings, like 6h or more. Right now, this situation typically requires more RAM, like 32 or 64 GB. Kilosort splits the data into small batches (default = 2s) and sorts each separately. However, a few steps (spike extraction, clustering) require keeping some information from each batch in RAM, before it is used or saved at the end of the run through the entire data. This situation is also likely to arise with fewer channels, and correspondingly longer recordings. 
#. More channels. This has not been tested but is likely to require more GPU RAM. Reduce the batch size if running into memory limitations. 
#. Very few channels. Kilosort4 might fail for 1 channel. 