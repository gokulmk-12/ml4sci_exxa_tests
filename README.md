# ML4SCI EXXA Tests

## Contact Information
- **Name**    : Gokul M K
- **Email**   : ed21b026@smail.iitm.ac.in
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
