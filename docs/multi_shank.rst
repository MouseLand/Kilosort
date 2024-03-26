.. _multi_shank:

Using probes with multiple shanks
==================================
Currently, Kilosort4 does not process shanks separately. Until this is added, we recommend changing your probe layout to artificially stack the shanks in a single column with a bit of vertical space (~100um) between each shank.

If that option isn't feasible for some reason, you can also try adjusting the `min_template_size` and/or `dminx` parameters in the GUI, or in the settings argument for `run_kilosort` if you're using the API. Setting `dminx` to around half the total width of the probe seems to be a good starting point, and you can adjust from there.

See `issue 606 <https://github.com/MouseLand/Kilosort/issues/606>`_ and `issue 617 <https://github.com/MouseLand/Kilosort/issues/617>`_ for additional context.