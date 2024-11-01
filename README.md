# METEOR
----------

**Map Enhancement Tools for Ephemeral Occupancy Refinement**

[![Pytest](https://github.com/alisiafadini/meteor/actions/workflows/tests.yml/badge.svg)](https://github.com/your_username/your_repo/actions/workflows/tests.yml)
[![Mypy](https://github.com/alisiafadini/meteor/actions/workflows/mypy.yml/badge.svg)](https://github.com/your_username/your_repo/actions/workflows/mypy.yml)
[![Ruff](https://github.com/alisiafadini/meteor/actions/workflows/lint.yml/badge.svg)](https://github.com/your_username/your_repo/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/alisiafadini/meteor/graph/badge.svg?token=NCYMP06MNS)](https://codecov.io/gh/alisiafadini/meteor)
![GitHub Tag](https://img.shields.io/github/v/tag/rs-station/meteor)


`meteor` is a tool for computing crystallographic difference maps. 

`meteor` specializes the robust identification of weak signals arising from minor but scientifically interesting populations, such as bound ligands or changes that occur in time-resolved experiments. That said, if you need an difference map, `meteor` can do it!


## quickstart

❗ `meteor` is currently **in beta**. We re-wrote everything recently, moving from a research code to something that can be robustly used as a tool. If you are willing to put up with a few sharp edges, it would be great if you give it a spin and then send us feedback: on how easy/not it was to use and what kinds of scientific results you obtain.

Finally: a word of caution. Expect changes in the coming weeks as we stress test the code. You might want to consider this before publishing any results with `meteor` until we exit `beta`. 

First, meteor needs a `python3.11` environment. We're working hard to extend this to as many versions as possible. To be sure things work, we recommend [installing conda](https://docs.anaconda.com/miniconda/) and creating a fresh environment,
```
conda create --name meteor python==3.11 --yes
conda activate meteor
```

Then install `meteor` using pip
```
pip install meteor-maps
```

Once installed, you will have two command-line scripts. Ask for more info using `-h`:
```
meteor.diffmap -h
meteor.phaseboost -h
```
these scripts compute denoised difference maps using the constant-phase approximation _vs._ iterative phase retrieval, respectively. See below for additional detail

## the science behind `meteor`

### philosophy: better science through automation

`meteor` aims to:

1. maximize signal to noise
2. be objective and reproducible (minimize user choice & bias)
3. be easy to use

Aim 1 is met using structure factor amplitude weighting (e.g. k-weighting, existing art) and [TV denoising](https://en.wikipedia.org/wiki/Total_variation_denoising) (new in the context of crystallography). Aims 2 and 3 are met through automatically setting parameters using negentropy maximization (as in [ICA](https://en.wikipedia.org/wiki/Independent_component_analysis)). For all the details, see our paper (coming soon to a preprint server near you).


### isomorphous data, please

METEOR is only for isomorphous difference maps, meaning the lattices/symmetries of the _native_ and _derivative_ datasets are comparable. If you need to compare non-isomorphous lattices, check out [matchmaps](https://github.com/rs-station/matchmaps).


## command-line details

`meteor` provides two command-line scripts that most users will want. If you prefer working in Jupyter notebooks or want to develop against `meteor`'s library, refer to the [API documentation](https://rs-station.github.io/meteor/).

Both of `meteor`'s scripts generate difference maps (MTZs). We recommend starting with `meteor.diffmap`. This script applies k-weighting then TV-denoising, and picks parameters for both by maximizing negentropy. It's relatively fast to run (expect about a minute) and has fewer nobs to turn. After you've tried `meteor.diffmap`, you can give `meteor.phaseboost` a try. This script iteratively applys TV denoising, and adjusts the phases of the _derivative_ data to try and produce a denoised map. At the end, it applies a k-weighting and TV denoising pass. It often results in slightly better maps, at the cost of additional compute (many minutes). We also have noticed the algorithm can become unstable in rare circumstances. If you think that might be happening to you, please get in touch.

Note that individual steps in both of these scripts can be turned off or modified using command-line flags, as described below.

One note: in the lingo we adope, we compute _derivative_ minus _native_ maps. Initial phases typically from a model of the _native_ data, computed from a CIF/PDB model. Usually the derivative data are ligand bound, time-resolved-activated, or similar... but in the end, the use case is defined by you!

### meteor.diffmap

Compute difference maps, including k-weighting and TV-denoising options. To see the full help, just run `meteor.diffmap -h`. Here are some examples that highlight common use cases:

Computing a k-weighted, TV-denoised diffmap, with `meteor` making some smart default choices for me:
```
meteor.diffmap derivative.mtz native.mtz -s native.pdb
```
note that the order of `derivative.mtz` and `native.mtz` matters!

Suppose I have some non-standard column names. Using `gemmi mtz <my.mtz>` might be smart to find out what they should be; then,
```
meteor.diffmap derivative.mtz --derivative-amplitude-column F_ON --derivative-uncertainty-column SIGF_ON native.mtz -s native.pdb
```
or, equivalently,
```
meteor.diffmap derivative.mtz --da F_ON --du SIGF_ON native.mtz -s native.pdb
```

what if I want to compute a k-weighted map, with k-parameter of `0.05`, and no TV?
```
meteor.diffmap derivative.mtz native.mtz -s native.pdb --kweight-mode fixed --kweight-parameter 0.05 --tv-denoise-mode none 
```

To note and remember:

  - the k-parameter sets how strongly outlier difference structure factor amplitudes are suppressed
  - k-weighting with k-parameter of `0.0` is NOT the same as with k-weighting totally turned off!
  - the TV-weight trades off the smoothness (lack of noise) in the final map _vs._ fidelity to the original map. A higher TV weight means more denoising, and a greater departure from the original data


### meteor.phaseboost

Compute iterative-TV denoised difference maps. The usage is very similar to `meteor.diffmap`, so let's focus on the three new flags:

  - `--tv-weights-to-scan` sets what TV weights are assessed at every iteration -- you may want to try and play with this. Increasing the number of scanned points will probably result in more stable runs, and perhaps slightly better results, at the cost of computation
  - `--convergence-tolerance` dictactes the phase change at which the algorithm stops. The default is pretty good, but if you notice the negentropy is still increasing at the end of your run, try lowering this. If instead, you have a lot of maps to denoise and you notice things converge early, you can lower this
  - `--max-iterations` is mostly just there to force the algorithm to bail out just in case it starts to thrash; so far, we haven't seen cases where it's necessary to adjust this


### advanced options

If you feel adventerous, check out `meteor/settings.py`. Default values are collected there, and you could ducktype to your heart's content. Not recommended for 99% of users!


### what the heck are these `meteor_metadata.json` files?

These are flat text files that contain information about how `meteor` ran. The primary intent is for debugging purposes, but you can also [read them with `meteor`](https://github.com/rs-station/meteor/blob/64f96ca0a293520cbd0163267768ddbfd68c7b0b/meteor/scripts/common.py#L331) or any standard JSON parser, and make some plots to better understand `meteor`'s runtime behavior.
