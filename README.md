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
Hi, I’m Gokul, a Dual Degree student in Engineering Design with a specialization in Robotics at the Indian Institute of Technology, Madras. I’m interested in developing Learning and Control based solutions for Robotic Challenges. To achieve this, I have gathered knowledge in Reinforcement Learning, Control Engineering, Deep Learning and Generative AI. I am proficient in **C**, **C++**, **Python**, and **MATLAB**, along with frameworks such as **PyTorch**, **Torch Lightning**, **Keras**, **TensorFlow**, and **JAX**. Lately, I have been delving into astrophysics, exploring topics like gravitational lensing of massive stars and galaxies. My strong foundation in Deep Learning, combined with my interest in astrophysics, has led me to the DeepLense Project.

# Test Details
## 1) General Test: Unsupervised Clustering of Protoplanetary disks
- **Goal**: Machine learning model capable of unsupervised clustering of the disks
- **Evaluation**: Visual Inspection
- **Library**: torch, torchvision, sklearn

## 2) Image-Based Test: Reconstruction of Disk Images
- **Goal**: Autoencoder to output the images resembling the inputs
- **Evaluation**: MSE, SSIM 
- **Library**: torch, torchvision, sklearn

### Plan & Results
I have implemented a Convolutional Autoencoder to reconstruct input images. The figure below gives an overview of the architecture used. The images were resized to (128, 128) to enable homogenity with the autoencoder architecture. 

**Experiments**
- Applied a standard MSE loss between the input and output images, but the reconstructions were poor.
- Used a weighted combination of MSE and SSIM loss, resulting in decent reconstructions but no significant improvement.
- Experimented with a weighted blend of L1 loss and SSIM loss, which produced much better results than the previous approaches.
- Introduced a perceptual loss component using a pretrained VGG-16 network, comparing the original and reconstructed images. This led to clean, sharp reconstructions that effectively preserved the original features.

![flowchart](https://github.com/user-attachments/assets/f34bd624-4c8b-466c-892c-20be26d1f2f4)

**Hyperparamters**

<table>
  <tr>
    <td>
      <table>
        <tr><th>Hyperparameter</th><th>Value</th></tr>
        <tr><td>MSE Loss Weight</td><td>0.45</td></tr>
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

Below are the MSE, SSIM plot distribution over the test dataset (15 images), and few example reconstructions from the test dataset.

![resultPlot](https://github.com/user-attachments/assets/f235a658-d589-445d-8d27-9bc52a935867)

![resultAE](https://github.com/user-attachments/assets/4909fe11-7a11-4de5-ab4d-23e1bc3af2e2)
