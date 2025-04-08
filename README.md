# ML4SCI EXXA Tests

## Contact Information
- **Name**    : Gokul M K
- **Email**   : mkgokul2003@gmail.com
- **Github**  : [gokulmk-12](https://github.com/gokulmk-12)
- **Website** : [Portfolio](https://gokulmk-12.github.io/)
- **LinkedIn**: [gokul-m-k](https://www.linkedin.com/in/gokul-m-k-886a93263/)
- **Location**: Chennai, Tamil Nadu, India
- **Timezone**: IST (UTC +5:30)

## Education Details
- **University**: Indian Institute of Technology Madras
- **Degree**: Bachelors in Engineering Design (B.Tech), Dual Degree in Robotics (IDDD)
- **Major**: Robotics and Artificial Intelligence
- **Expected Graduation**: May, 2026

## Background
Hi, I’m Gokul, a Dual Degree student in Engineering Design with a specialization in Robotics and a minor in AI at the Indian Institute of Technology, Madras. I’m interested in developing Learning and Control based solutions for Robotic Challenges. To achieve this, I have gathered knowledge in Reinforcement Learning, Control Engineering, Deep Learning and Generative AI. I am proficient in **C**, **C++**, **Python**, and **MATLAB**, along with frameworks such as **PyTorch**, **Torch Lightning**, **Keras**, **TensorFlow**, and **JAX**. Lately, I have been delving into astrophysics, exploring topics like analysing exoplanet atmospheres, strong gravitational lensing. My strong foundation in Deep Learning, combined with my interest in astrophysics, has led me to the EXXA Project.

# Test Details
## 1) General Test: Unsupervised Clustering of Protoplanetary disks
- **Goal**: Machine learning model capable of unsupervised clustering of the disks
- **Evaluation**: Visual Inspection
- **Library**: torch, torchvision, sklearn
- **Specific Reference**: [An evaluation of pre-trained models for feature extraction in image classification](https://arxiv.org/abs/2310.02037)

### Plan & Results
- To cluster the images, the initial approach was to extract features. Based on the paper, "An evaluation of pre-trained models for feature extraction in image classification", CLIP-ResNet50, ViT-H-14, and ConvNeXt demonstrated superior performance. I chose **ConvNeXt** since the other two models required resizing images from (600, 600) to (224, 224), which could result in the loss of crucial clustering features.
- After extracting features, each image was represented by a 1000-dimensional feature vector. To reduce dimensionality, I applied UMAP, selecting the three most relevant components for clustering with the Gaussian Mixture Model (to deal with cluster overlaps). The optimal number of clusters was determined using the Silhouette Score and Davies-Bouldin Score. From the plot below, 2 had the best, but was neglected after visual inspection.

<img src="General Test/results/score.png" style="width:100%; height:auto;"/>

- A cluster count of **6** provided the best balance, yielding the **highest Silhouette Score** and **lowest DB Score** across a tested range of 2 to 20. The upper limit was chosen after visually inspecting the data and noting the presence of 73 unique planets. 

### Clustering Analysis
- The model aimed to capture fine-grained details on the planet's disk. For instance, in one of the clusters, it identified three high-intensity points around the protoplanetary disk as key features for clustering the planets. This can be confirmed by examining the activation maps. (FM - Feature Map)

<img src="General Test/results/features1.png" style="width:100%; height:10%;"/>
<img src="General Test/results/cluster4.png" style="width:100%; height:10%;"/>

- The model accurately identifies the specific regions where the disks are located in the image and utilizes this information, along with intensity levels, to cluster planets, as illustrated below.

<img src="General Test/results/features2.png" style="width:100%; height:10%;"/>
<img src="General Test/results/cluster3.png" style="width:100%; height:10%;"/>

**NOTE**: The DB and Silhouette scores are highly variable and change with each run. I used them to estimate an approximate cluster number. Please disregard the plot and instead refer to the interactive cluster plot in the last cell.

## 2) Image-Based Test: Reconstruction of Disk Images
- **Goal**: Autoencoder to output the images resembling the inputs
- **Evaluation**: MSE, SSIM 
- **Library**: torch, torchvision, sklearn
- **Dataset**: https://drive.google.com/file/d/1zUYoQs7VxxtYNgBueL-53zh9-JClfQ5U/view?usp=sharing

### Plan & Results
I have implemented a Convolutional Autoencoder to reconstruct input images. The figure below gives an overview of the architecture used. The images were resized to (128, 128) to enable homogenity with the autoencoder architecture. 

**Experiments**
- Applied a standard MSE loss between the input and output images, but the reconstructions were poor.
- Used a weighted combination of MSE and SSIM loss, resulting in decent reconstructions but no significant improvement.
- Experimented with a weighted blend of L1 loss and SSIM loss, which produced much better results than the previous approaches.
- Introduced a perceptual loss component using a pretrained VGG-16 network, comparing the original and reconstructed images. This led to clean, sharp reconstructions that effectively preserved the original features.

<img src="Image-Based Test/result/flowchart.png" style="width:100%;"/>

**Hyperparamters**

<table>
  <tr>
    <td>
      <table>
        <tr><th>Hyperparameter</th><th>Value</th></tr>
        <tr><td>L1 Loss Weight</td><td>0.45</td></tr>
        <tr><td>SSIM Loss Weight</td><td>0.1</td></tr>
        <tr><td>Perceptual Loss Weight</td><td>0.45</td></tr>
        <tr><td>Train-Test Split</td><td>90:10</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><th>Hyperparameter</th><th>Value</th></tr>
        <tr><td>Learning Rate</td><td>0.0001</td></tr>
        <tr><td>Optimizer</td><td>Adam</td></tr>
        <tr><td>Batch Size</td><td>16</td></tr>
        <tr><td>Iterations</td><td>500</td></tr>
      </table>
    </td>
  </tr>
</table>

Below are the MSE, SSIM plot distribution over the test dataset (15 images), and few example reconstructions from the test dataset. Please do take care of the scale below in the MSE plot.

<img src="Image-Based Test/result/resultPlot.png" style="width:100%;"/>

<img src="Image-Based Test/result/resultAE.png" style="width:100%;"/>

## 3) Sequential Test: Supervised classifier for light curves
- **Goal**: Machine learning model capable of determining the presence of planets from light (transit) curves.
- **Evaluation**: ROC, AUC, Real Kepler Observation
- **Library**: pytransit, torch, sklearn, astropy 
- **Specific Reference**: [Deep learning for time series classification: a review](https://arxiv.org/abs/1809.04356)
- **Dataset**: Located in the same directory as the notebook

### Plan & Results
I generated a dataset of 1,000 light curves using the [PyTransit](https://pytransit.readthedocs.io/en/latest/index.html) package. Half of them (500) simulate planetary transits with visible transit depths, while the remaining 500 are flat curves representing the absence of a planet. To mimic real-world observations, Gaussian noise was added. The table below outlines the parameter ranges used for simulating the transits.

<table>
  <tr>
    <td>
      <table>
        <tr><th>Transit Parameters</th><th>Range</th></tr>
        <tr><td>Planet-Star Radius Ratio (k)</td><td>Normal(0.10, 0.002)</td></tr>
        <tr><td>Orbital Period (p)</td><td>Normal(1.0, 0.01)</td></tr>
        <tr><td>Limb Darkening Coefficient (ldc)</td><td>Uniform(0.0, 0.6, (1, 2))</td></tr>
        <tr><td>Scaled Semi-Major Axis (a)</td><td>Normal(4.2, 0.1)</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><th>Transit Parameters</th><th>Range</th></tr>
        <tr><td>Zero Epoch (t0)</td><td>Normal(0.0, 0.001)</td></tr>
        <tr><td>Orbital Inclination (i)</td><td>Uniform(0.48 &pi;, 0.5 &pi;)</td></tr>
        <tr><td>Orbital Eccentricity (e)</td><td>Uniform(0.0, 0.25)</td></tr>
        <tr><td>Gaussian Noise</td><td>300 PPM</td></tr>
      </table>
    </td>
  </tr>
</table>

<img src="Sequential Test/results/sampleplot.png" style="width:100%;"/>

For the classification task, I employed a ResNet-1D architecture. According to the paper [Deep learning for time series classification: a review](https://arxiv.org/abs/1809.04356), ResNet-1D consistently outperforms other models in univariate time series classification. The model trained rapidly, within just 10 seconds, and successfully learned the classification in under 10 iterations. Below are the key hyperparameters used, followed by the results.

<table>
  <tr><th>Hyperparameters</th><th>Value</th></tr>
  <tr><td>Learning Rate</td><td>0.001</td></tr>
  <tr><td>Optimizer</td><td>Adam</td></tr>
  <tr><td>Batch Size</td><td>16</td></tr>
  <tr><td>Train-Test Split</td><td>90:10</td></tr>
</table>

<img src="Sequential Test/results/rocauc.png" style="width:100%;"/>
<img src="Sequential Test/results/modelpred.png" style="width:100%;"/>

The model was then evaluated on real light curve observations from the Kepler mission. I utilized a curated [Kepler example file](https://docs.astropy.org/en/stable/timeseries/index.html#using-timeseries) provided by Astropy, along with raw Kepler light curve data from the [MAST archive](https://archive.stsci.edu/kepler/download_options.html). Both datasets are in FITS format. The model's predictions on these files are presented below.

<img src="Sequential Test/results/planet1.png" style="width:100%;"/>
<img src="Sequential Test/results/planet2.png" style="width:100%;"/>

## References
- https://learn.astropy.org/tutorials/FITS-images.html
- https://avanderburg.github.io/tutorial/tutorial.html
- https://pytransit.readthedocs.io/en/latest/index.html
- https://lightkurve.github.io/lightkurve/index.html
- https://docs.astropy.org/en/stable/timeseries/index.html#using-timeseries
- https://archive.stsci.edu/kepler/download_options.html
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- https://www.kungfu.ai/blog-post/convnext-a-transformer-inspired-cnn-architecture
