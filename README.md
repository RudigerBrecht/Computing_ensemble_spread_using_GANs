
# Computing the ensemble spread from deterministic weather predictions using conditional generative adversarial networks

Ensemble prediction systems are an invaluable tool for weather forecasting. Practically, ensemble predictions are obtained by running several perturbation of the deterministic control forecast. However, ensemble prediction is associated with a high computational cost and often involves statistical post-processing steps to improve its quality.
Here we propose to use deep-learning-based algorithms to learn the statistical properties of an ensemble prediction system, the ensemble spread, given only the deterministic control forecast. Thus, once trained, the costly ensemble prediction system will not be needed anymore to obtain future ensemble forecasts, and the statistical properties of the ensemble can be derived from a single deterministic forecast. We adapt the classical \texttt{pix2pix} architecture to a three-dimensional model and also experiment with a shared latent space encoder--decoder model, and train them against several years of operational (ensemble) weather forecasts for the 500 hPa geopotential height. The results demonstrate that the trained models indeed allow obtaining a highly accurate ensemble spread from the control forecast only.

# About this repository 

Once the manuscript (link) is accapted, we will upload the code to train the neural networks.
For now, the code in this repository reproduces the plots from the manuscript. 
