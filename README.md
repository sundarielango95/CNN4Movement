# Bilateral neural network model of stroke recovery

Code for the paper:

> Elango S, Francis AJA, Chakravarthy VS (2022). **Interaction of network and rehabilitation therapy parameters in defining recovery after stroke in a Bilateral Neural Network.** *Journal of NeuroEngineering and Rehabilitation* 19:142. [doi:10.1186/s12984-022-01106-3](https://doi.org/10.1186/s12984-022-01106-3) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/36536385/)

## What the model does

A bilateral convolutional network sees a target through two eyes (a stereo image pair), receives a cue saying which hand or hands to move, and outputs the joint angles that bring one or both arms to the target in 3D space. Each hemisphere has its own channel, with connections between them.

To simulate a stroke, a chosen percentage of units in one hemisphere is silenced. The lesioned network is then retrained under different simulated therapies, so recovery can be compared across:

- lesion size and time since stroke onset,
- movement complexity during therapy (exploratory vs stereotypic),
- hand-selection mode (affected hand alone, or both hands),
- local vs global plasticity,
- structural integrity of the connections between hemispheres.

Main findings:

- Exploratory training beat stereotypic training at every lesion size.
- Moving the affected hand mattered; whether the other hand moved as well made little difference.
- Larger lesions, or a longer time since onset, needed global plasticity to recover.
- The integrity of the inter-hemispheric connections changed which kind of plasticity worked best.

## Scripts

| Script | What it does |
|---|---|
| `05_a_Healthy_Model_Creator_CCM.py` | Trains the healthy (unlesioned) bilateral network and saves the model and the lesion index files |
| `05_b_Healthy_Model_Tester_CCM.py` | Lesions the healthy model at a chosen percentage and measures reaching error |
| `05_c_Rehabilitation_Centre_CCM.py` | Retrains the lesioned model under a simulated therapy |
| `05_c_Rehabilitation_Centre_CCM_3bit.py` | The same, with a three-bit task cue instead of two |
| `05_CCM_l1_reg.py` | Trains the healthy model with L1 activity regularisation |
| `05_CCM_OD_Testing.py` | Loads the L1-regularised model and saves its layer activations |
| `05_d_…_Han_et_al_creator.py`, `05_e_…_Han_et_al_tester.py`, `05_e_…_Han_et_al_rehab.py` | A variant of the model following Han et al.: create, test and rehabilitate |

Run them in the order **creator → tester → rehabilitation centre**.

## Before you run it

This is the research code as it was run for the paper (2020). Three things to know:

1. **Paths.** The scripts read and write absolute Windows paths (for example `D:/V2_from_PC/...`). Change these to your own folders.
2. **Data.** The stereo target images, rendered in Blender, are not included in this repository. Contact the author for access.
3. **Environment.** The code uses standalone Keras 2.3 on TensorFlow 2.x (2020 releases). `keras.layers.normalization` no longer exists in newer Keras, so use the versions in `requirements.txt`. It also needs OpenCV, NumPy and pandas.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python 05_a_Healthy_Model_Creator_CCM.py
```

## Citation

```bibtex
@article{elango2022bilateral,
  title   = {Interaction of network and rehabilitation therapy parameters in defining recovery after stroke in a Bilateral Neural Network},
  author  = {Elango, Sundari and Francis, Amal Jude Ashwin and Chakravarthy, V. Srinivasa},
  journal = {Journal of NeuroEngineering and Rehabilitation},
  volume  = {19},
  pages   = {142},
  year    = {2022},
  doi     = {10.1186/s12984-022-01106-3}
}
```

## Contact

Sundari Elango, sundarielango95@gmail.com
