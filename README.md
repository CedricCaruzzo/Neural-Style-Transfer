# Neural Style Transfer

![[Main Image](https://github.com/CedricCaruzzo/Neural-Style-Transfer/blob/main/img/results/dancing_picasso.jpg)](https://github.com/CedricCaruzzo/Neural-Style-Transfer/blob/main/img/results/dancing_picasso.jpg)

## Introduction

Neural Style Transfer (NST) was introduced by **Leon Gatys et al. in 2015**. It consists of applying the **style** of a reference image to a target image while conserving the **content**, as exemplified:

**Style**: Textures, colors, visual patterns across various spatial scales.  
**Content**: Higher-level macrostructure of the image.

The idea of style transfer is related to **texture generation**, which has a long history in image processing before the development of its neural counterpart.

The key notion behind the implementation of NST is the same idea fundamental to all Deep Learning algorithms: definition of a loss function.

In high-level terms, the loss function is defined as:


```math
\mathcal{L}_{\text{style\_transfer}} = \underbrace{\text{distance}(\text{style}(\text{reference\_image}), \text{style}(\text{combination\_image}))}_{\text{Style loss}} + \underbrace{\text{distance}(\text{content}(\text{original\_image}), \text{content}(\text{combination\_image}))}_{\text{Content loss}}
```


Here:
- $\text{distance}$: A norm function such as $\text{L}\_2$ norm.  
- $\text{content}$: A function computing a representation of the image content.  
- $\text{style}$: A function computing the representation of the image style.

Minimizing the loss function ensures:
- $\text{style}(\text{combination}\_{image}) \approx \text{style}(\text{reference}\_{image})$,  
- $\text{content}(\text{combination}\_{image}) \approx \text{content}(\text{original}\_{image})$.

Gatys et al. found that convolutional neural networks (CNNs) offer a way to mathematically define the $\text{style}$ and $\text{content}$ functions.

---

## The Content Loss

The activations from earlier layers in a network contain *local* information, while activations from higher layers capture increasingly global and abstract information. Thus, the **content** of an image, which is more global, is found in the upper-layer representations of a CNN.

Let $F^l \in \mathbb{R}^{C_l \times H_l \times W_l}$ denote the feature map of the generated image at layer $l$, and $P^l$ denote the feature map of the content image at the same layer. Then the content loss is defined as:

```math
\mathcal{L}_{\text{content}} = \frac{1}{2} \sum_{i,j} \left( F_{ij}^l - P_{ij}^l \right)^2
```

This guarantees that the generated image will maintain high-level structural similarity to the target content image. It assumes that upper layers in a CNN effectively "see" the content of the input images.

---

## The Style Loss

Unlike content loss, which uses a single upper layer, style loss uses **multiple layers** of a CNN to capture texture patterns across spatial scales.

To model the style of an image, we compute the **Gram matrix** of a layer's activations. The Gram matrix captures the correlations between feature maps at layer $l$, reflecting the texture statistics at that scale.

For a feature map $F^l \in \mathbb{R}^{C_l \times H_l \times W_l}$, the Gram matrix is:

```math
G^l_{ij} = \sum_{k=1}^{H_l W_l} F_{ik}^l F_{jk}^l
```

where $G^l \in \mathbb{R}^{C_l \times C_l}$. The style loss for layer $l$ is the Frobenius norm between the Gram matrices of the generated image ($G^l$) and the style reference image ($A^l$):

```math
\mathcal{L}_{\text{style}}^l = \frac{1}{4 N_l^2 M_l^2} \sum_{i,j} \left( G_{ij}^l - A_{ij}^l \right)^2
```

where:
- $N_l = C_l$: Number of feature maps (channels).  
- $M_l = H_l \times W_l$: Number of spatial locations.  

The total style loss aggregates contributions across multiple layers:

```math
\mathcal{L}_{\text{style}} = \sum_{l} w_l \mathcal{L}_{\text{style}}^l
```

Here, $w_l$ is a weight factor controlling the contribution of layer $l$.

---

## Total Loss

The total loss combines content and style losses:

```math
\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{content}} + \beta \mathcal{L}_{\text{style}}
```

where:
- $\alpha$: Weight for content preservation.  
- $\beta$: Weight for style transfer.

---

## Additional Notes

- **Choice of Layers:**  
   - Content loss typically uses higher layers to capture global structure.  
   - Style loss uses multiple layers (low and high) to capture textures at various scales.

- **Customization of Style Scales:**  
   By adjusting $w_l$, specific spatial scales of style can be emphasized or suppressed.

- **Optimization Process:**  
   - The generated image is initialized (e.g., as noise or the content image).  
   - Gradient descent is applied to iteratively minimize $\mathcal{L}_{\text{total}}$.

- **Summary:**  
   - Preserve content by aligning high-level activations of the generated and content images.  
   - Preserve style by aligning feature correlations (Gram matrices) of the generated and style images.

---
 
# Resources

\[1\] Gatys, L. A. (2015). A neural algorithm of artistic style. _arXiv preprint arXiv:1508.06576_.

\[2\] Chollet, Francois. Deep learning with Python. Simon and Schuster, 2021.

---

# Neural Style Transfer Notebook Usage

This project provides a simple interface to experiment with Neural Style Transfer, including style transfer, content visualization, and style visualization.

## Usage Examples

### 1. Full Style Transfer
Use this to apply the style of an image to a content image.

```python
main(content_path='img/content/dancing.jpg', style_path='img/style/picasso.jpg', mode='style_transfer')
```

### 2. Visualizing Content Reconstruction from Random Noise
Use this to observe how the content image emerges from random noise during optimization.

```python
main(content_path='noise', style_path='img/content/dancing.jpg', mode='content')
```

### 3. Visualizing Style Representation
Use this to visualize how the network interprets the style of the image.

```python
main(content_path='noise', style_path='img/style/picasso.jpg', mode='style')
```

## Parameters
- content_path: Path to the content image or 'noise' for random noise initialization.
- style_path: Path to the style image.
- mode: One of the following:
  - 'style_transfer': Performs full style transfer.
  - 'content': Reconstructs the content image from noise.
  - 'style': Visualizes the style as seen by the network.

## Example Directory Structure
```
img/
├── content/
│   └── dancing.jpg
│
├── results/
│
├── style/
│   └── picasso.jpg
```

Happy experimenting!



## Results

![(https://github.com/CedricCaruzzo/Neural-Style-Transfer/blob/main/img/results/dancing_vg_starry_night.jpg)](https://github.com/CedricCaruzzo/Neural-Style-Transfer/blob/main/img/results/dancing_vg_starry_night.jpg)

![(https://github.com/CedricCaruzzo/Neural-Style-Transfer/blob/main/img/results/village_paper_texture.jpg)](https://github.com/CedricCaruzzo/Neural-Style-Transfer/blob/main/img/results/village_paper_texture.jpg)

![(https://github.com/CedricCaruzzo/Neural-Style-Transfer/blob/main/img/results/dancing_picasso.jpg)](https://github.com/CedricCaruzzo/Neural-Style-Transfer/blob/main/img/results/village_night.png)
