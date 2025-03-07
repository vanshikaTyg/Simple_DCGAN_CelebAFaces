# Simple_DCGAN_CelebAFaces
This repository contains an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) in PyTorch to generate synthetic images. The model is trained on an image dataset, applying various preprocessing steps before training both the generator and discriminator models.

## Aim  
The aim of this assignment is to implement a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic images from a given dataset. DCGANs leverage convolutional neural networks (CNNs) to improve the stability and performance of traditional GANs.
<hr>

## Dataset  
The model is trained on the *CelebA Faces* dataset, which consists of large-scale high-resolution images. These datasets are commonly used for generative modeling tasks.

<hr>

## Dataset Preprocessing

### Steps:
1. **Dataset Location:** Place the dataset inside the `datasets/extracted/` folder.
2. **Image Transformations:**
   - Resize images to `64x64` pixels.
   - Apply center cropping.
   - Convert images to tensors.
   - Normalize pixel values to the range `[-1, 1]`.
3. **DataLoader Configuration:**
   - `Batch size:` 128
   - `Shuffle:` Enabled
   - `Workers:` 4
<hr>

## Training the model

### Generator Architecture:
- Input: `100-dimensional` random noise vector.
- Series of transposed convolutional layers with batch normalization and ReLU activations.
- Final layer uses `Tanh` activation to output images in range `[-1,1]`.

### Discriminator Architecture:
- Input: 64x64 image (real or generated).
- Convolutional layers with `LeakyReLU` activation for classification.
- Final layer applies `Sigmoid` activation to output a probability score.

### Training Process:
1. The generator produces synthetic images from noise.
2. The discriminator classifies real and generated images.
3. Both models update through **Binary Cross Entropy (BCE) Loss** and the **Adam optimizer**.
4. The generator aims to fool the discriminator, while the discriminator tries to improve its classification.
5. Training runs for **50-100 epochs** for better-quality results.

<hr>

### Training Loop*  
1.⁠ ⁠*Discriminator Training:*  
   - Forward pass with real images and compute loss.  
   - Generate fake images and compute loss.  
   - Update the Discriminator.  
<hr>

2.⁠ ⁠*Generator Training:*  
   - Generate fake images.  
   - Compute loss (fake images should be classified as real).  
   - Update the Generator.  
<hr>

3.⁠ ⁠*Saving Generated Images:*  
   - At each epoch, sample and save generated images to visualize progress.

<hr>

## Testing the Model

1. Once trained, the generator can create new images from random noise.
2. Generate images by sampling noise and passing it through the trained generator.
3. Save or visualize generated images using:
   ```python
   torchvision.utils.save_image(generated_images, 'output.png')

