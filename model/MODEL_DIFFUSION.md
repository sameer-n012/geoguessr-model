# GeoGuessr Diffusion Model Architecture

## Goal
Build a diffusion-based vision model that predicts where one or more street-level images were taken by learning a **geographic probability distribution** rather than only a single best point.

The model should support:

- `country`
- `region / admin area`
- `geographic density over Earth`
- `latitude, longitude`
- `uncertainty / confidence`
- multiple plausible modes when the image is ambiguous

The model should work for both:

- a single image
- a set of images from the same location or immediate surroundings

## Recommended Diffusion Formulation
Use a **conditional diffusion model over a geographic heatmap**.

This is the cleanest diffusion formulation for this task because the true target is often multimodal:

- one road scene may plausibly match several countries
- one suburban view may match many distant regions
- one partial image may only constrain climate, road rules, or hemisphere

Instead of directly regressing one coordinate, the model learns:

- `p(geography | image)`

and then extracts:

- top modes
- expected coordinate
- uncertainty radius

## Why Heatmap Diffusion Instead of Coordinate Diffusion
A naive diffusion model over `(lat, lon)` is possible, but it is not a good first design.

Problems with direct coordinate diffusion:

- coordinates are too low-dimensional for diffusion to be especially natural
- Earth is spherical and wraps at longitude boundaries
- multimodal outputs are awkward unless many samples are drawn
- training targets become brittle around poles and boundaries

A heatmap representation avoids most of that.

Recommendation:

- represent the Earth as a 2D rasterized geographic density map
- diffuse over that map conditioned on image features
- decode the final map into coordinates and probabilities

## Prediction Target

### Primary target: geographic density map
Define a global Earth map in an equal-area or near-equal-area projection.

Practical v1 choice:

- `256 x 128` heatmap
- each pixel represents a geographic bin
- ground truth is a **soft target**, not a single hot pixel

Construct the training target by placing a Gaussian-like kernel around the true location.

Why a soft target matters:

- it stabilizes training
- it reflects label uncertainty
- it makes nearby predictions less harshly wrong
- it aligns well with diffusion on spatial distributions

### Secondary target: derived cell distribution
To compare against the non-diffusion model, decode the heatmap into:

- coarse geographic cells
- fine geographic cells
- country scores

This gives a direct way to benchmark the two systems on common metrics.

### Final coordinate prediction
Derive coordinates from the predicted heatmap using one of these methods:

- expected coordinate under the density map
- mode of the heatmap
- top-k local maxima with weights

Recommended output bundle:

- global expected coordinate
- top-k modes with scores
- uncertainty radius from entropy or mass concentration

## Model Architecture

### 1. Image encoder
Use a strong pretrained vision backbone.

Best practical options:

- `ViT-B/16` or `ViT-L/14` pretrained with DINOv2 / SigLIP / CLIP-style objectives
- ConvNeXt if compute is tighter

Recommendation:

- start with `ViT-B/16`
- image size `336x336` or `448x448`

Encoder output:

- per-image embedding `z_i`
- optional patch tokens for richer conditioning

The backbone requirements are the same as the non-diffusion model. The hard part is still visual representation quality.

### 2. Multi-image fusion module
For one or more images from the same location, encode each image independently and fuse them into one conditioning representation.

Recommended v1 fusion:

- shared image encoder for all images
- learned quality gate `g_i`
- weighted average of image embeddings
- small attention or MLP fusion block

Pseudo-structure:

```text
for each image i:
  z_i = Encoder(image_i)
  g_i = sigmoid(MLP_gate(z_i))

z_pool = sum(g_i * z_i) / sum(g_i)
z_cond = FusionBlock({z_i}, z_pool)
```

This `z_cond` becomes the conditioning vector for the diffusion model.

### 3. Geographic latent representation
Represent the target as a heatmap `H` over Earth.

Possible parameterizations:

- probability heatmap over a projected Earth raster
- multi-scale heatmap pyramid
- hierarchical cell raster projected into image-like form

Recommended v1:

- one single-channel heatmap
- resolution `256 x 128`
- values normalized to sum to 1 after decoding

### 4. Diffusion backbone
Use a small conditional `U-Net` operating on the geographic heatmap.

Input to the denoiser at timestep `t`:

- noisy target heatmap `x_t`
- timestep embedding `emb(t)`
- image conditioning vector `z_cond`

Architecture:

- sinusoidal timestep embedding
- FiLM or cross-attention conditioning using `z_cond`
- 2D U-Net with residual blocks
- predict either:
  - added noise `epsilon`
  - denoised target `x_0`
  - velocity `v`

Recommendation:

- use `v`-prediction or `epsilon`-prediction
- keep the U-Net modest in size because the output map is small

Practical structure:

```text
noisy_geo_heatmap + timestep + image_condition
    -> conditional U-Net
    -> denoised heatmap prediction
```

### 5. Multi-scale refinement head
Geographic uncertainty is naturally coarse-to-fine. Reflect that in the diffusion decoder.

Recommended enhancement:

- predict heatmaps at `64x32`, `128x64`, and `256x128`
- supervise all scales
- use coarse-to-fine skip refinement in the U-Net

This gives the model a way to represent:

- continental uncertainty at low resolution
- local refinement at high resolution

### 6. Optional retrieval branch
A diffusion model will still benefit from retrieval.

Purpose:

- correct visually specific cases
- sharpen ambiguous geographic distributions
- improve landmark-like performance

Architecture:

- add a retrieval embedding head from `z_cond`
- retrieve top-k similar geotagged examples
- convert retrieved coordinates into a retrieval density map
- fuse retrieval density with the diffusion output

This is optional in v1, but likely useful if the diffusion model underperforms on memorization-heavy scenes.

## Full System Diagram

```text
single image or image set
        |
   shared image encoder
        |
   per-image embeddings
        |
   set fusion / pooling
        |
   conditioning vector z_cond ---------------------> retrieval embedding head -> ANN search
        |
noisy geographic heatmap + timestep
        |
   conditional geographic U-Net
        |
   denoised heatmap / density prediction
        |
   normalized geographic posterior
        |
   top modes + expected coordinate + uncertainty
        |
 optionally fused with retrieval-based density map
```

## Forward Process and Targets

### Ground-truth density construction
For each sample with true coordinate `(lat, lon)`:

1. project the coordinate into the heatmap grid
2. place a Gaussian or von Mises-Fisher-like blob around that point
3. normalize the target map

Possible kernel widths:

- narrow kernel for precise labels
- broader kernel for noisy labels or sparse imagery

Recommendation:

- start with a kernel corresponding to roughly `25-100 km` depending on map resolution
- tune this carefully, because it changes how sharp the posterior becomes

### Diffusion target choices
Three common prediction targets:

- noise prediction `epsilon`
- clean sample prediction `x_0`
- velocity prediction `v`

Recommendation:

- use `v`-prediction or `epsilon`-prediction
- keep the implementation aligned with standard DDPM / DDIM practice

## Loss Functions
Use a weighted sum of losses.

### 1. Diffusion reconstruction loss
Standard denoising loss:

- `L_diff = ||pred - target||^2`

where the target depends on whether the model predicts `epsilon`, `x_0`, or `v`.

### 2. Final heatmap consistency loss
After reconstructing the clean geographic density map, add a direct loss on the predicted heatmap.

Recommended options:

- KL divergence between predicted and target density
- cross-entropy on normalized maps
- Earth Mover's Distance if feasible

Recommendation:

- use KL divergence or cross-entropy first

### 3. Coordinate consistency loss
Decode the predicted heatmap to an expected coordinate and apply a geographic loss.

Options:

- Haversine loss
- ECEF L2 loss

This should be a small auxiliary term, not the primary objective.

### 4. Auxiliary classification losses
Decode the density map into:

- country probabilities
- coarse cell probabilities
- fine cell probabilities

Then supervise these with auxiliary losses.

Why add these:

- they stabilize learning
- they make the diffusion model easier to compare against the non-diffusion baseline
- they improve broad geographic calibration

### 5. Retrieval loss
If retrieval is enabled:

- `L_retr = InfoNCE / contrastive loss`

### Total loss

```text
L = w1 * L_diff
  + w2 * L_heatmap
  + w3 * L_geo
  + w4 * L_aux
  + w5 * L_retr
```

Initial weighting suggestion:

- make `L_diff` dominant
- keep `L_heatmap` moderate
- keep geographic and auxiliary losses secondary

## Inference Pipeline

### Single image
1. Encode image into `z_cond`
2. Start from Gaussian noise in geographic heatmap space
3. Run reverse diffusion steps to generate one or more heatmaps
4. Average or ensemble samples
5. Decode the posterior into:
   - expected coordinate
   - top-k modes
   - country and cell probabilities
   - uncertainty radius
6. Optionally fuse with retrieval density

### Multi-image
1. Encode each image with shared encoder
2. Fuse to obtain `z_cond`
3. Run the same diffusion pipeline
4. Decode final geographic posterior

## Output Format
The model should return:

- expected `lat, lon`
- top-k modes with coordinates and scores
- optional country top-k
- optional cell top-k
- uncertainty radius in kilometers
- optional predicted heatmap for visualization

Example:

```json
{
  "lat": 45.421,
  "lon": -75.690,
  "modes": [
    {"lat": 45.4, "lon": -75.7, "score": 0.43},
    {"lat": 44.9, "lon": -73.2, "score": 0.16}
  ],
  "country_topk": [["CA", 0.58], ["US", 0.24]],
  "uncertainty_km": 71.4
}
```

## Evaluation
Use the same metrics as the non-diffusion model so the comparison is fair.

### Core metrics
- median haversine error
- mean haversine error
- percentage within `1 km`, `10 km`, `25 km`, `100 km`, `500 km`
- country accuracy
- region accuracy

### Distribution quality metrics
These matter more for diffusion than for the baseline model.

- negative log-likelihood of the true region under the predicted heatmap
- calibration of posterior mass
- entropy vs actual error correlation
- top-k mode recall

### Slice metrics
Report by:

- country
- continent
- urban vs rural
- single-image vs multi-image
- camera source
- low-ambiguity vs high-ambiguity scenes

## Training Strategy

### Stage 1: warm-start image encoder
Start from a pretrained vision backbone.

### Stage 2: supervised heatmap predictor before full diffusion
This is recommended even if the final model is diffusion-based.

Train a simple deterministic head first:

- image embedding -> geographic heatmap

Why:

- verifies that the representation works
- gives a strong initialization signal
- reduces the risk of diffusion training hiding basic failures

### Stage 3: train conditional diffusion on geographic heatmaps
Enable the denoising objective.

Recommendation:

- begin with single-image training
- then move to multi-image fusion

### Stage 4: multi-sample posterior decoding
At inference time, draw multiple samples and evaluate:

- posterior sharpness
- calibration
- mode collapse risk
- runtime vs accuracy tradeoff

### Stage 5: add retrieval if needed
If the diffusion model struggles on highly specific roads or landmarks, add the retrieval branch.

## Strengths of This Architecture

- models multimodal geographic uncertainty naturally
- produces a full posterior rather than just a point estimate
- gives interpretable heatmaps for debugging
- can express ambiguity across regions or countries
- supports one-image and multi-image inputs cleanly

## Weaknesses of This Architecture

- more expensive at inference than a classifier
- more complex to train and tune
- likely slower to iterate on
- may still underperform retrieval-heavy baselines on memorization cases
- output resolution is limited by the heatmap grid unless extra refinement is added

## Comparison With the Non-Diffusion Model

### Where diffusion may win
- ambiguous scenes with multiple plausible regions
- uncertainty calibration
- top-k geographic posterior quality
- interpretability via density maps

### Where diffusion may lose
- raw point-estimate accuracy per unit of compute
- training stability
- inference speed
- engineering simplicity

### Practical expectation
If both are implemented competently:

- the hierarchical classifier/regressor is more likely to win on first-pass median distance error
- the diffusion model may produce better calibrated uncertainty and richer multi-modal predictions

## Minimum Viable Version
If building the first diffusion version quickly, implement this subset:

1. pretrained `ViT-B/16` image encoder
2. weighted multi-image fusion
3. `256x128` geographic heatmap target
4. conditional diffusion U-Net over heatmaps
5. expected-coordinate decoder
6. no retrieval in v1

That is the simplest testable diffusion baseline.

## Recommended V1 / V2 Roadmap

### V1
- conditional geographic heatmap diffusion
- single-image and multi-image support
- expected coordinate output
- uncertainty from posterior entropy or concentration

### V2
- multi-scale heatmap supervision
- country and cell auxiliary heads
- DDIM or fast sampler for cheaper inference
- retrieval branch

### V3
- hierarchical diffusion over coarse-to-fine cells
- latent diffusion instead of pixel-space heatmap diffusion
- mixture of diffusion and retrieval density fusion

## Implementation Notes

### Map projection
Projection choice matters.

You want a projection that avoids extreme training distortion.

Reasonable options:

- equirectangular for simplicity
- equal-area projection for cleaner density semantics

Recommendation:

- use equirectangular in v1 for engineering simplicity
- mask oceans if the dataset is exclusively land-based road imagery

### Heatmap normalization
The predicted map should represent a probability distribution.

Recommended decoding:

- apply nonnegative output transform
- normalize to sum to 1 over valid geographic bins
- compute all downstream outputs from this normalized density

### Land mask
A land mask is useful.

Benefits:

- prevents wasting density on impossible locations
- improves calibration
- sharpens inference for road-image datasets

### Coordinate decoding
Compute:

- expected coordinate from the normalized map
- top-k local maxima for modal predictions

Be careful with longitude wraparound near `-180 / 180`.

### Sampling budget
Diffusion inference cost depends on the number of denoising steps.

Recommendation:

- start with a modest sampler such as DDIM
- test `10-50` steps instead of using a large default blindly

## Concrete Recommendation
If you want a diffusion model that is directly comparable to the baseline classifier architecture, build it around the following stack:

- **Backbone:** pretrained `ViT-B/16`
- **Input:** one image or `1-6` perspective crops
- **Fusion:** gated weighted average + small attention block
- **Target:** `256x128` soft geographic heatmap
- **Diffusion core:** conditional 2D U-Net with timestep embedding and image conditioning
- **Output:** posterior heatmap, expected coordinate, top-k modes, confidence
- **Later add:** retrieval density fusion

This is the most defensible diffusion formulation for the project.

## Pseudocode Sketch

```text
images -> shared_encoder -> per_image_embeddings

if num_images == 1:
    z_cond = embedding_1
else:
    weights = sigmoid(gate_mlp(per_image_embeddings))
    z_pool = weighted_pool(per_image_embeddings, weights)
    z_cond = attention_fusion(z_pool, per_image_embeddings)

x_T = gaussian_noise_heatmap()

for t in reverse_timesteps:
    pred = geographic_unet(x_t, t, z_cond)
    x_{t-1} = diffusion_step(x_t, pred, t)

heatmap = normalize(decode(x_0))
expected_coord = expected_latlon(heatmap)
modes = topk_local_maxima(heatmap)

retr_embed = normalize(retrieval_head(z_cond))
neighbors = ann_search(retr_embed)
final_heatmap = fuse_heatmap_with_retrieval_density(heatmap, neighbors)
```

## Final Recommendation
The diffusion alternative should be built as a **conditional geographic heatmap diffusion model**, not as direct diffusion over raw coordinates.

That gives you a realistic way to test whether richer geographic posteriors are worth the added complexity relative to the hierarchical classifier/regressor baseline.
