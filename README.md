# Kilosort2: automated spike sorting with drift tracking and template matching on GPUs #

Welcome to Kilosort2, a Matlab package for spike sorting electrophysiological data of up to thousands of channels. To aid in setting up a Kilosort2 run on your own probe configuration, we have developed a [graphical user interface](https://github.com/MouseLand/Kilosort2/wiki) where filepaths can be set and data loaded and visually inspected, to make sure Kilosort2 sees it correctly. The final output of Kilosort2 can be visualized in the [Phy GUI](https://github.com/kwikteam/phy), which must be installed separately (we recommend the development version).

Required toolboxes: parallel computing toolbox, signal processing toolbox.

There is currently no preprint or paper for Kilosort2, so please read below to find out how it works. Kilosort2 improves on Kilosort primarily by employing drift tracking, which changes the templates continuously as a function of drift. Drift tracking does not depend on a particular probe geometry, but denser spacing of sites should generally help to better track neurons, especially if the probe movement is large. Kilosort2 has been primarily developed on Neuropixels 1.0 data, but has also been tested on a few other configurations.

### Installation ###
You must run and complete successfully "mexGPUall" in the CUDA folder. This requires mexcuda support, which comes with the parallel computing toolbox. To set up mexcuda compilation, install the exact version of the CUDA toolkit compatible with your Matlab version (see [here](https://www.mathworks.com/help/distcomp/gpu-support-by-release.html)). On Windows, you must also install a CPU compiler, for example the freely available [Visual Studio Community 2013](https://www.visualstudio.com/vs/older-downloads/). Note that the most recent editions of Visual Studio are usually not compatible with CUDA. If you had previously used a different CPU compiler in Matlab, you must switch to the CUDA-compatible compiler using mex -setup C++. For more about mexcuda installation, see these [instructions](http://uk.mathworks.com/help/distcomp/mexcuda.html).

### General instructions for running Kilosort ###

#### Option 1: Using the GUI

Navigate to the kilosort directory and run 'kilosort':
```
>> cd \my\kilosort2\directory\
>> kilosort
```
See the [GUI documentation](https://github.com/MouseLand/Kilosort2/wiki) for more details.

#### Option 2: Using scripts (classic method)

1. Make a copy of master_file_example_MOVEME.m and \configFiles\StandardConfig_MOVEME.m and put them in the directory with your data.
2. Generate a channel map file for your probe using \configFiles\createChannelMap.m as a starting point.
3. Edit the config file with desired parameters. You should at least set the file paths (ops.fbinary, ops.fproc (this file will not exist yet - kilosort will create it), and ops.root), the sampling frequency (ops.fs), the number of channels in the file (ops.NchanTOT) and the location of your channel map file (ops.chanMap).
4. Edit master_file so that the paths at the top (lines 3-4) point to your local copies of those github repositories, and so that the configuration file is correctly specified (lines 6-7).

### Parameters ###

If you are unhappy with the quality of the automated sorting, try changing one of the main parameters:

`ops.Th = [10 4]` (default). Thresholds on spike detection used during the optimization (Th(1)) or during the final pass (Th(2)). These thresholds are applied to the template projections, not to the voltage. Typically, Th(1) is high enough that the algorithm only picks up sortable units, while Th(2) is low enough that it can pick all of the spikes of these units. It doesn't matter if the final pass also collects noise: an additional per neuron threshold is set afterwards, and a splitting step ensures clusters with multiple units get split.

`ops.splitAUC = 0.9` (default). Threshold on the area under the curve (AUC) criterion for performing a split in the final step. If the AUC of the split is higher than this, that split is considered valid. If the cross-correlation of the split units does not contain a big dip at time 0, the split goes through.

`ops.lambda = 10` (default).  The individual spike amplitudes are biased towards the mean of the cluster by this factor. 50 is a lot, 0 is no bias.

A list of all the adjustable parameters is in the example configuration file.

### Integration with Phy GUI ###
Kilosort2 provides a results file called "rez", where the first column of rez.st are the spike times and the second column are the cluster identities. It also provides a field rez.good which is 1 if the algorithm classified that cluster as a good single unit. To visualize the results of Kilosort2, you can use [Phy](https://github.com/kwikteam/phy), which also provides a manual clustering interface for refining the results of the algorithm. Unlike Kilosort1, Kilosort2 will automatically set the "good" units in Phy based on a <20% estimated contamination rate with spikes from other neurons (computed from the refractory period violations relative to expected).

Because Phy is written in Python, you also need to install [npy-matlab](https://github.com/kwikteam/npy-matlab), to provide read/write functions from Matlab to Python.

Detailed instructions for interpreting results are provided [here](https://github.com/kwikteam/phy-contrib/blob/master/docs/template-gui.md). This documentation was developed for Kilosort1, so things will look a little different with Kilosort2.

### Matlab output structures ###

Kilosort is best used in conjunction with Phy. The .npy and .csv output files can then be loaded back into Matlab, following these general instructions: https://github.com/kwikteam/phy-contrib/blob/master/docs/template-gui.md. For a full example, see the tutorial with Neuropixels results data available here: http://data.cortexlab.net/singlePhase3/ and here: http://data.cortexlab.net/dualPhase3/.

However, in some situations you might need to use the Matlab results structures. Here is an explanation of these variables, available inside the struct called "rez"

xc, yc: x and y coordinates of each channel on the probe, in the order of channels provided in the channel map (default is linear, 1:1:nChannels).

connected: whether a channel in the original binary dat is "connected", or "active". Inactive channels are ignored.

Wrot: cross-channel whitening matrix. Wrot * high_pass_filtered_data = post_data, where post_data is the postprocessed data on which the Kilosort algorithm is applied.

WrotInv: is the matrix inverse of Wrot. WrotInv * post_data = high_pass_filtered_data

ops: keeps all the configuration settings provided by the user, and cumulative information added throghout the Kilosort steps

ccb: dissimilarity matrix between each batch and every other batch

ccbsort: ccb reordered by the drift estimation algorithm

st: first column is the spike time in samples, second column is the spike template, third column is the extracted amplitude, and fifth column is the post auto-merge cluster (if you run the auto-merger).

mu: mean amplitude for each template

U: low-rank components of the spatial masks for each template

W: low-rank components of the temporal masks for each template

dWU: average of a subset of spikes corresponding to each template. The low-rank decomposition of this matrix results in W and U.

Wraw: the spike template, un-whitened by the operation Wraw(:,:,n) = Wrotinv' * (U(:,n,:) * W(:,n,:)'), for each template n.

simScore: correlation between all pairs of templates.

cProj: projections of each detected spike onto the principal components of the channels corresponding to the spike's assigned template. The channel order for each template is available in iNeigh.

iNeigh: for each template, the channels with largest amplitudes are indexed in order (default 12). This indexing is used to sort coefficients in cProj. Notice this is a fundamentally sparse scheme: only the top channels for each template are stored.

cProjPC: projections of each detected spike onto the top templates most similar to the spike's assigned template. The nearest-template order for each template is available in iNeighPC.

iNeighPC: for each template, the other templates with largest similarity are indexed in order (default 12). This indexing is used to sort coefficients in cProjPC. Notice this is a fundamentally sparse scheme: only the top closest template for each template are stored.

### How Kilosort2 works ###

The pipeline in Kilosort2 is divided into four steps, which we explain separately.

#### 1) Data preprocessing and channel selection.

First we determine which channels contain significant numbers of negative threshold crossings (default 0.1Hz). In our experience, channels with very few threshold crossings are not useful and can only confuse the sorting. We high-pass filter the data (default 150Hz), and "whiten" the channels. Whitening is the process of removing the correlations between a channel and its neighboring channels by subtracting a weighted sum of the neighbors. This increases the discriminability of spikes that are localized on the probe and thus sortable into single units, as contrasted with faraway spikes that are seen over a large number of channels and are typically not sortable.

#### 2) Drift calculation and batch reordering.

Kilosort2 tracks waveforms as they change as a function of time, or as a function of drift. Because changes as a function of time are mostly due to probe drift, it is advantageous to "re-order time" so that Kilosort2 can traverse the data approximately in the order of vertical probe location. The preprocessed data is already broken up into small batches (default is 2s). The goal of this step is to determine the dissimilarity matrix between all batches, and re-order the batches so that the dissimilarity matrix becomes more diagonal, or at least the dissimilarities along the diagonal are smallest. To achieve this, spikes are extracted via threshold crossings from each batch, and then clustered via scaled k-means. To compare two batches, we compute the average minimum distance from each cluster in one batch to the clusters in the other batch, then symmetrize this matrix and z-score it.

Re-ordering the dissimilarity matrix is related to the traveling salesman problem. We implemented two efficient approximate solutions. The default solution assigns a phase variable to each batch, and optimizes these phases by gradient descent so that the similarity between all batches x and y is approximated by cos(phase_x - phase_y). The alternative solution uses the [rastermap](https://github.com/MouseLand/rastermap) algorithm. If the re-ordering is successful, the re-ordered matrix should clearly contain most high-similarity values along the diagonal.

#### 3) The main optimization.

To cluster neurons, Kilosort2 traverses the data twice, in the order computed at the previous step. On the first pass, a template of the waveforms is determined for each cluster. This template is allowed to change from batch to batch, retaining a memory of the previous ~N spikes (default N starts at 20 and increases to 400 during optimization). Because the batches have been sorted in order of electrode depth, as the depth changes slowly, the template will similarly change to track the moving waveforms. The iterative template matching procedure is similar to Kilosort1: spikes are detected from the raw data based on peak correlation with the set of templates, and a scaled version of this template is subtracted at that location, thus "peeling away" that spike. This allows other nearby spikes to be extracted even if they have a large overlap with the spikes that were peeled away.

Unlike Kilosort, Kilosort2 automatically determines the right number of templates. The algorithm starts with no templates, and continually scans the residual voltage data after template matching to find any spikes that were not matched to existing templates. Such spikes are introduced as new templates. Conversely, templates which do not capture spikes (default minFR rate is 1/50 Hz) are dropped during the optimization. This results in a continual turnover of spikes which Kilosort2 "tries on" to see if they can account for any of the residual voltage data. This procedure results in automated determination of the number of templates, and more importantly controls the degree of over-splitting. Because templates are introduced gradually, local minima are avoided in which two templates can co-exist that account for the same neuron, unless that neuron has a very large degree of amplitude variability. These relatively few cases are handled by periodically merging clusters which have very correlated waveforms. If any oversplits survive through the entire, they are handled in the final step below.

Note that both the first "optimization" pass, and the final "extraction" pass use the exact same template matching and subtraction algorithm. In contrast, Kilosort1 did not do template subtraction in the optimization phase, and was thus unable to determine whether the residual voltage contains waveforms unaccounted for by existing templates.  

#### 4) Final merges and splits.

These steps are meant to mimic the actions which a user would take refining the automated output of Kilosort in the Phy GUI. They result in a set of "good" neurons that are similar to that produced by a human curator in the Phy GUI.

**Merges**: on rare occasions (~ 5 per 300 neurons), the amplitude variability is so high that the neuron is oversplit and returned as a few different clusters. In these cases, we perform an additional merging step after the main optimization, where a merge goes through if the waveforms are correlated >0.5 AND the cross-correlogram has very few refractory violations, as would be expected if the clusters belonged to the same neuron.

**Splits**: by design, Kilosort2 over-merges units, which has several advantageous consequences. For example, similar units also "drift together". Therefore, changes to the template for one neuron will generalize to track the template from another neuron. Since sparsely-firing neurons are hard to track, grouping them together with a nearby unit can be beneficial. Another advantageous consequence is that the post-optimization steps are more restricted and independent: each cluster is split into multiple units completely independently of the other clusters, and the pieces are never re-merged. To determine directions of splits for each cluster, we developed a pursuit algorithm similar to independent components analysis (ICA), where we pursue bimodality rather than sparseness. The optimized projection contains a local maximum of bimodality, such that the one-dimensional distribution of spike projections on that dimension are well modelled by a mixture of Gaussians. The inferred overlap between the Gaussians in the mixture is taken as a measure of the contamination of the two putative clusters, and the AUC is used as a score. Only splits with an AUC score above splitAUC (default 0.9) can be made. The splits are veto-ed by the shape of the spike times cross-correlogram (CCG) after the putative split. If the CCG appears to be refractory, the split is veto-ed and the clusters are merged back together.

**Splits (1/2 and 2/2)**: two sets of splits are performed by default to account for local minima of the pursuit procedure. The first pass is initialized with the top principal component of the cluster. The second pass is initialized with the mean template as the putative bimodal projection axis. The spikes are high-pass filtered at this step, to account for the slow changes induced by drift.

**Threshold detection**: finally, we found that some neurons at the noise floor can benefit from adaptive setting of the threshold for each neuron. We start this procedure with ops.Th(1), and independently for each neuron lower the threshold in 0.5 decrements, as long as the number of refractory violations in the autocorrelogram does not increase beyond the acceptable quality threshold (of 20% estimated contamination). For this step to work well, ops.Th(2) used in the final pass should be relatively small, so that all potential spikes are collected.


### Release history

v0.3 (February 2019): first public release; includes more automation.
1. Removed a number of optimization steps that were judged ineffective. The spike amplitude variance is no longer empirically calculated, and is instead fixed and incorporated into the ops.lam parameter the way it is in Kilosort1. One of the two types of merging during the optimization was removed, because it was never used.
2. Default ops.lam value changed to 10, due to the above. The ops.lam parameter is now equivalent to the corresponding parameter in Kilosort1.
3. Several of the variables that accumulate during optimization (dWU, W, nsp) were switched from singles to doubles to lower the run-to-run variability. Some variability still exists, which we believe is due to numerical errors from using single variables on the GPU. This is unavoidable, since most GPUs have much faster processing of single-typed data, but should have minimal impact on the final results.  
4. A new merging step was added at the end of the optimization. It is similar to the merges during optimization, in that waveforms that are correlated get merged. Unlike the main optimization, no restriction is put on the amplitudes of the waveforms to be merged, so that a unit with high amplitude variability can be consolidated. However, this step also requires that the cross-correlogram (CCG) have a very clear refractory period.  
5. A bug was fixed in the splitting step, where the spikes were not re-ordered according to drift before high-pass filtering. This resulted in much less effective splits.
6. As a consequence of 5, the ccsplit threshold (now called splitAUC) can now be lowered significantly before false positives are found. The default changes to 0.9.
7. A veto step was added to the splits: if a split would result in a refractory CCG, it is not performed.
8. Units with clean auto-correlograms (few refractory violations) are labelled as good, in rez.good, and in Phy. The default threshold is 20% estimated contamination.
9. A step was added before data preprocessing where bad channels (no spikes) are detected and removed.
10. The Kilosort GUI was updated. The data views now use the actual time-varying templates used by Kilosort2. The adjustable parameters were restricted to Th, splitAUC and lam. The GUI now allows the creation of new probe geometries. A wiki was added to guide the GUI user.
11. A new batch re-ordering algorithm was added (rastermap), to be tried if the default fails in some noticeable way.

v0.2 (June 2018): first fully functional version, which fuses drift tracking with template matching.
1. Used in original Stringer, Pachitariu et al, 2018b preprint.
2. Ongoing testing by collaborators.

v0.1 (April 2018): initial batch re-ordering algorithm and drift correction.
1. This original version did not contain template matching, and the spikes were instead detected by threshold crossing.
2. Used in original Stringer, Pachitariu et al, 2018a preprint.


### Questions ###

Please create an issue for bugs / installation problems.

### Licence ###

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
