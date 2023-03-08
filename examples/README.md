# Targeted BEHRT
Repository for publication: Targeted-BEHRT: Deep Learning for Observational Causal Inference on Longitudinal Electronic Health Records<br/>
IEEE Transactions on Neural Networks and Learning Systems; Special Issue on Causality<br/>
https://ieeexplore.ieee.org/document/9804397/<br/>
DOI: 10.1109/TNNLS.2022.3183864.<br/>

![Screenshot](screenshot.png)

How to use:<br/>
In "examples" folder, run the "run_TBEHRT.ipynb" file. A test.csv file is provided to test/play and demonstrate how the vocabulary/year/age/etc function (please read full paper linked above for further methodological details). <br/>
The files in the "src" folder contain model and data handling packages in addition to other necessary VAE relevant files and helper functions.

Requirements:<br/>
torch >1.6.0<br/>
numpy 1.19.2<br/>
sklearn 0.23.2<br/>
pandas 1.1.3<br/>
<br/>
