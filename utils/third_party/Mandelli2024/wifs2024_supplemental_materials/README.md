## Supplemental materials

In `ROC_results_fully-synth_vs_laundered.jpg` we report the ROCs of the binary classification problems tackled in Table I and II of the paper _When Synthetic Traces Hide Real Content: Analysis of Stable Diffusion Image Laundering_, accepted at IEEE WIFS 2024.

We also trained a ternary classifier based on the same backbone architecture (EfficientNet-B4) to separate real images from fully-synthetic and laundered images. 
At test phase, we considered the same testing set used in our paper.
The achieved balanced classification accuracy on the test set is 98.88%.
In `ternary_cm.jpg` we report the confusion matrix achieved by the ternary classifier on the test set.
