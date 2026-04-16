# GeoGuessr Model Architecture

## Goal
Build a vision model that predicts where one or more street-level images were taken, with useful outputs at multiple granularities:

- `country`
- `region / admin area`
- `geographic cell`
- `latitude, longitude`
- `uncertainty / confidence`

The model should work for both:

- a single image
- a set of images from the same location or immediate surroundings

## Recommended System
Use a **coarse-to-fine geolocation model with retrieval augmentation**.

This is the most practical architecture because the task has two very different failure modes:

- Some images are solved by broad geographic cues: road markings, vegetation, architecture, driving side, utility poles, language.
- Some images require memorization-like matching: a specific road, skyline, mountain profile, sign style, or camera artifact.

A single end-to-end regressor is usually worse than a system that combines:

1. **global semantic understanding**
2. **discrete geographic classification**
3. **metric coordinate regression or expectation over cells**
4. **image retrieval against a large geotagged index**

## Prediction Formulation

### Primary target: hierarchical geographic cells
Discretize the Earth into cells and predict a distribution over them.

Recommended hierarchy:

- Level 1: `country / macro-region`
- Level 2: `S2 or H3 cells` at a coarse level
- Level 3: `S2 or H3 cells` at a finer level

A good starting point:

- coarse grid: cells with typical diameter around `250-500 km`
- fine grid: cells with typical diameter around `25-75 km`

Why classification first:

- classification is easier to optimize than raw lat/lon regression
- it naturally supports uncertainty
- it handles multimodal predictions better
- it enables geographically aware losses

### Secondary target: coordinate prediction
Produce final coordinates from the fine-cell distribution using one of these approaches:

- **Expected coordinate over top-k fine cells** using each cell centroid and its probability
- **Residual regression inside the predicted fine cell**

Recommended first version:

- predict fine-cell probabilities
- predict a small `(dx, dy)` residual relative to the winning or expected cell centroid

### Auxiliary targets
Add auxiliary heads to improve feature learning:

- `country classification`
- `climate / biome zone`
- `driving side`
- `road type / urban-vs-rural`
- `hemisphere` or `broad latitude band`

These are optional at inference time, but useful during training.

## Model Architecture

### 1. Image encoder
Use a strong pretrained vision backbone.

Best practical options:

- `ViT-L/14` or `ViT-B/16` pretrained with DINOv2 / SigLIP / CLIP-style objectives
- a modern ConvNeXt if training resources are more limited

Recommendation:

- Start with a `ViT-B/16`-class encoder for iteration speed
- Move to `ViT-L` after the pipeline is stable

Input resolution:

- `336x336` or `448x448`

Reasoning:

- Geo-localization depends on global layout and small regional cues
- ViTs tend to work well when both scene semantics and texture cues matter

Encoder output:

- one global embedding vector `z_img`
- optionally a small set of patch tokens for attention pooling or retrieval refinement

### 2. Single-image prediction head
Feed the encoder embedding into a multi-task prediction block.

Structure:

- `LayerNorm`
- `MLP projection`
- separate heads for:
  - coarse-cell logits
  - fine-cell logits
  - country logits
  - coordinate residual
  - uncertainty scalar

This is a standard multi-head architecture:

```text
image -> encoder -> z_img
                 -> coarse head
                 -> fine head
                 -> country head
                 -> residual head
                 -> uncertainty head
```

### 3. Multi-image fusion module
For one or more images from the same guess location, encode each image independently, then fuse.

Let image embeddings be:

- `z_1, z_2, ..., z_n`

Use a **set encoder** rather than concatenation so the system supports variable numbers of images.

Recommended fusion stack:

- per-image encoder shared across all images
- learned image-quality gate `g_i`
- transformer or attention pooling over `{z_i}`
- pooled embedding `z_set`

Practical first version:

- score each image with a small MLP gate
- compute weighted average of embeddings
- run one self-attention block or one MLP on the pooled vector

Why this works:

- some views contain stronger location signals than others
- one image may show vegetation while another shows road signs
- weighted fusion helps ignore low-value or redundant images

Pseudo-structure:

```text
for each image i:
  z_i = Encoder(image_i)
  g_i = sigmoid(MLP_gate(z_i))

z_pool = sum(g_i * z_i) / sum(g_i)
z_set  = FusionBlock({z_i}, z_pool)

z_set -> prediction heads
```

### 4. Retrieval branch
Add a parallel embedding head for nearest-neighbor retrieval.

Purpose:

- retrieve visually similar geotagged training images or panoramas
- improve results for landmark-like or highly specific road scenes
- provide interpretable evidence at inference time

Architecture:

- project `z_img` or `z_set` into normalized retrieval embedding `e`
- train with contrastive learning using geographic positives

Positive pair definition:

- same panorama or nearby images within a small radius
- semantically similar nearby scenes if available

Negative pair definition:

- geographically distant images
- visually similar but far-away hard negatives

Inference:

- search ANN index over all training embeddings
- collect top-k retrieved images and their coordinates
- merge retrieval evidence with classifier probabilities

Fusion of retrieval + classifier:

- convert retrieved neighbors into a kernel density estimate over coordinates or cells
- linearly combine with model logits / probabilities

This gives two complementary signals:

- classifier: robust global prior
- retrieval: instance-level correction

### 5. Geographic prior head
Add an optional lightweight prior model that depends on metadata if allowed.

Possible metadata:

- sun elevation estimate
- EXIF timestamp
- sequence ordering
- camera heading

If no metadata is available, skip this entirely.

If metadata exists, encode it with a small MLP and merge with `z_img` or `z_set` before prediction.

## Full System Diagram

```text
single image or image set
        |
   shared image encoder
        |
   per-image embeddings
        |
   set fusion / pooling  ----> retrieval embedding head ----> ANN search
        |
   shared fused embedding
        |
   +-------------------+-------------------+------------------+
   |                   |                   |                  |
coarse cell head   fine cell head     aux heads         residual head
   |                   |                   |                  |
   +-------------------+-------------------+------------------+
                           |
                calibrated geographic distribution
                           |
             coordinate estimate + uncertainty
                           |
       optionally fused with retrieval-based coordinate density
```

## Loss Functions
Use a weighted sum of losses.

### 1. Coarse-cell classification loss
- `L_coarse = CrossEntropy(coarse_logits, coarse_target)`

### 2. Fine-cell classification loss
- `L_fine = CrossEntropy(fine_logits, fine_target)`

Improvement:
- use label smoothing or neighbor-aware smoothing so nearby cells are penalized less than far-away cells

### 3. Coordinate residual loss
- `L_coord = Huber(pred_residual, true_residual)`

Compute residual relative to the target fine-cell centroid.

### 4. Geographic distance loss
Add a loss directly on the final predicted coordinates.

Options:

- Haversine distance
- ECEF-space L2 loss

Recommendation:

- use a small-weight Haversine loss as a consistency term

### 5. Retrieval / metric learning loss
- `L_retr = InfoNCE / contrastive loss / triplet loss`

### 6. Auxiliary classification losses
- country / biome / driving-side losses

### Total loss

```text
L = w1 * L_coarse
  + w2 * L_fine
  + w3 * L_coord
  + w4 * L_geo
  + w5 * L_retr
  + w6 * L_aux
```

Initial weighting suggestion:

- make `L_fine` dominant
- keep regression and geographic losses secondary
- keep retrieval loss moderate

## Data Representation

### Training example format
Each sample should store:

- image path or panorama crop paths
- `latitude`
- `longitude`
- country
- optional admin region
- optional metadata
- split assignment
- cell ids at each hierarchy level

For multi-image samples:

- `sample_id`
- list of image paths from the same location / panorama / nearby headings

### Image sampling
Geo models can overfit to country frequency and urban density. Balance the dataset.

Recommended sampling rules:

- cap oversampled countries or cities
- oversample underrepresented regions
- mix urban and rural scenes
- include both easy and ambiguous examples

### Panorama handling
If using panoramas or 360 sources:

- do not feed the full equirectangular pano first
- extract 4 to 8 perspective crops at fixed headings and pitch
- treat them as a multi-image set

This is simpler and usually more effective than training directly on raw panoramas in v1.

## Training Strategy

### Stage 1: representation warm start
Start from a pretrained encoder. Do not train from scratch.

### Stage 2: single-image geolocation training
Train on single images first.

Objective:

- stabilize coarse/fine geographic classification
- validate dataset quality and evaluation code

### Stage 3: multi-image fusion training
Enable image sets and train the fusion module.

Strategy:

- sample 1 image some of the time
- sample 2 to 6 images other times
- use image dropout so the model does not require every viewpoint

### Stage 4: retrieval training and indexing
Train the retrieval embedding head, build ANN index, and calibrate fusion with classifier outputs.

### Stage 5: fine-tuning with hard negatives
Mine mistakes such as:

- Spain vs. Portugal
- Balkans confusion
- US vs. Canada border regions
- Argentina vs. Uruguay
- Australia vs. South Africa dry regions

Hard-negative training matters a lot for this task.

## Inference Pipeline

### Single image
1. Encode image
2. Predict coarse and fine cell distributions
3. Predict residual coordinate offset
4. Retrieve top-k nearest neighbors from ANN index
5. Fuse classifier and retrieval outputs
6. Return best coordinate, top countries, uncertainty

### Multi-image
1. Encode each image with shared encoder
2. Compute image-quality weights
3. Fuse into a set embedding
4. Run the same prediction and retrieval stack
5. Return a single final prediction with confidence

## Output Format
The model should return:

- predicted `lat, lon`
- top-k fine cells with probabilities
- top-k countries with probabilities
- uncertainty radius in kilometers
- optional nearest retrieved examples

Example:

```json
{
  "lat": 45.421,
  "lon": -75.690,
  "country_topk": [["CA", 0.61], ["US", 0.21], ["FI", 0.04]],
  "cell_topk": [["h3_8_xxx", 0.33], ["h3_8_yyy", 0.27]],
  "uncertainty_km": 48.2
}
```

## Evaluation
Use metrics that reflect actual GeoGuessr quality.

### Core metrics
- median haversine error
- mean haversine error
- percentage within `1 km`, `10 km`, `25 km`, `100 km`, `500 km`
- country accuracy
- region accuracy

### Calibration metrics
- expected calibration error for country and cell probabilities
- coverage of predicted uncertainty radius

### Slice metrics
Report by:

- country
- continent
- urban vs rural
- daytime vs low-light if relevant
- single-image vs multi-image
- seen-camera-source vs unseen-camera-source

## Why This Architecture
This design is recommended because it matches the structure of the problem.

- Pure coordinate regression is too unstable.
- Pure classification does not give precise coordinates.
- Retrieval alone does not generalize well to unseen places.
- Multi-image fusion is necessary when the user supplies several views.
- Auxiliary tasks force the encoder to learn geographically meaningful cues.

The combination is strong because:

- classification gives robustness
- residual regression gives precision
- retrieval gives memorization capacity
- set fusion gives support for multiple images

## Minimum Viable Version
If building the first version quickly, implement this subset:

1. pretrained `ViT-B/16` image encoder
2. coarse + fine geographic cell heads
3. coordinate residual head
4. weighted-average multi-image fusion
5. no retrieval in v1

That version is simpler and still strong enough to validate the dataset and training loop.

## Recommended V1 / V2 Roadmap

### V1
- single-image + multi-image classifier/regressor
- hierarchical cells
- residual coordinate prediction
- uncertainty head

### V2
- retrieval branch with ANN index
- neighbor-aware label smoothing
- hard-negative mining
- probability calibration

### V3
- sequence-aware fusion for nearby frames
- pano-aware pretraining
- mixture density output for multi-modal predictions

## Implementation Notes

### Cell system
Prefer `H3` or `S2`.

- `H3` is convenient and widely used for spatial ML
- `S2` is also strong and maps well to hierarchical Earth partitioning

Either is fine. The important point is:

- hierarchical cells
- stable cell IDs
- easy neighbor lookup

### Coordinate parameterization
Avoid directly regressing raw latitude and longitude as the only target.

If a direct coordinate head is used, prefer:

- residuals from cell centroids
- or Earth-centered coordinates `(x, y, z)` for stability

### Uncertainty
Use either:

- a scalar predicted radius
- or entropy of the fine-cell distribution

First version:

- return entropy-derived uncertainty from the fine-cell softmax
- optionally learn a scalar calibration head later

## Concrete Recommendation
If this repository is starting from scratch, build the model around the following stack:

- **Backbone:** pretrained `ViT-B/16`
- **Input:** one image or `1-6` perspective crops
- **Fusion:** gated weighted average + small attention block
- **Targets:** coarse cell, fine cell, country, residual coordinate
- **Loss:** cross-entropy + Huber + small Haversine term
- **Output:** top-k cells, top-k countries, final coordinate, confidence
- **Later add:** retrieval embedding + ANN search

This is the best balance of:

- accuracy
- engineering complexity
- scalability
- support for multiple images

## Pseudocode Sketch

```text
images -> shared_encoder -> per_image_embeddings

if num_images == 1:
    z = embedding_1
else:
    weights = sigmoid(gate_mlp(per_image_embeddings))
    z = weighted_pool(per_image_embeddings, weights)
    z = attention_fusion(z, per_image_embeddings)

coarse_logits = coarse_head(z)
fine_logits   = fine_head(z)
country_logits = country_head(z)
residual      = residual_head(z)
retr_embed    = normalize(retrieval_head(z))

fine_probs = softmax(fine_logits)
pred_cell  = decode_topk(fine_probs)
pred_coord = centroid(pred_cell) + residual

neighbors = ann_search(retr_embed)
final_coord = fuse_classifier_and_retrieval(pred_coord, neighbors)
```

## Final Recommendation
The model should be built as a **hierarchical geoclassification system with coordinate refinement and optional retrieval**, not as a plain coordinate regressor.

That architecture is the most defensible choice for a GeoGuessr-style task and scales cleanly from one image to multiple images.
