Phase A: Baseline & Controls
                                                                                                                     
  ID: A1                                                                                                          
  Approach: Clean baseline reproduction. Run original OpenVPRLab MixVPR+ResNet50 with config/mixvpr_resnet50.yaml 
    unchanged. Target: MSLS R@1 >= 87.5. This is the anchor for all comparisons.                                  
  Prompt for Code Modification: No code change needed. Run as-is.                                                 
  ────────────────────────────────────────                                                                        
  ID: A2                                                                                                             
  Approach: Innovation code audit. Run your modified codebase with distillation fully disabled. Verify the gap vs A1
    is < 0.3%. If not, the framework has side effects to fix before any distillation work.                           
  Prompt for Code Modification: In vpr_framework.py, when distillation is disabled: do NOT instantiate or register 
  the                                                                                                                
     DistillationModule at all. Remove it from __init__. Ensure configure_optimizers only  collects parameters from 
    self.backbone and self.aggregator. The training_step should  have zero branching related to distillation when    
    disabled. The forward path must be  identical to the original OpenVPRLab code.

---
  Phase B: Global Semantic Distillation
                                                                                                                     
  ID: B1                                           
  Approach: CLIP global feature alignment (cosine). Frozen CLIP ViT-B-16 image encoder as teacher. Extract CLIP's CLS
                                                        
    token (768-dim) as the global semantic representation. Add a lightweight projector (Linear 2048->768) after the
    student's MixVPR output. Distillation loss = 1 - cosine_similarity(proj(student_global),  clip_cls).
    lambda_global=0.05, no warmup delay.                                                                             
  Prompt for Code Modification: Add a DistillationModule containing: (1) a frozen CLIP ViT-B-16 image encoder loaded 
    via open_clip, (2) a trainable Linear projection layer mapping student descriptor dim  to CLIP dim (768). In     
    training_step, after computing student descriptors via  backbone+aggregator, also run the batch through the 
  frozen
     CLIP encoder with  torch.no_grad() to get CLS tokens. Compute cosine distillation loss between projected
  student
     descriptors and CLIP CLS tokens. Total loss = metric_loss + lambda_global *  distill_loss. The projection
  layer's
     parameters must be included in  configure_optimizers. Config params: lambda_global (float), distillation.enabled

     (bool). At validation/inference time, discard the projector and CLIP entirely -- only  backbone+aggregator are
    used.
  ────────────────────────────────────────
  ID: B2                                                                                                             
  Approach: B1 with lambda sweep. Same as B1 but test lambda_global in {0.01, 0.05, 0.1, 0.2}. Pick the value that
    maximizes MSLS R@1 without degrading Pitts30k.                                                                   
  Prompt for Code Modification: Same code as B1. Sweep lambda_global via config.
  ────────────────────────────────────────
  ID: B3                                                                                                             
  Approach: B1 with scheduled lambda ramp-up. Instead of constant lambda, linearly ramp lambda_global from 0 to the
    best value (from B2) over the first 5 epochs, then hold constant. This avoids distillation interfering with early
                                                        
    metric learning.
  Prompt for Code Modification: Modify the distillation loss weighting: compute effective_lambda = min(1.0,
    current_step / ramp_steps) * lambda_global. Add config param distillation.ramp_epochs  (int). Convert to steps
    internally as ramp_steps = ramp_epochs * steps_per_epoch.

---
  Phase C: Local Semantic Attention Distillation
                                                                                                                     
  ID: C1                                           
  Approach: CLIP attention map extraction. Extract the CLS-to-patch attention weights from CLIP ViT-B-16's last      
    transformer layer. These form a spatial heatmap (1x196 for 14x14 patches) showing which image regions CLIP
    considers semantically salient. Resize to 20x20 to match student feature map. Normalize to sum=1 (probability
    distribution). This is the local semantic contribution distribution -- the teacher's spatial importance signal.
  Prompt for Code Modification: In the DistillationModule, modify the CLIP forward pass to also return the           
    CLS-to-patch attention from the last transformer block. Specifically, hook into the  last MultiheadAttention 
  layer                                                                                                              
     and extract attn_weights[:, :, 0, 1:] (CLS token  attending to all patch tokens), average across heads to get 
    shape [B, num_patches].  Reshape to [B, 1, 14, 14], bilinearly interpolate to [B, 1, 20, 20], then apply  softmax

    over the spatial dimension to get a normalized attention distribution. Return  this alongside the CLS token.
  ────────────────────────────────────────
  ID: C2                                                                                                             
  Approach: Student spatial attention head + KL divergence loss. Add a lightweight 1x1 Conv + Sigmoid head on the
    student's 20x20 backbone features to predict a spatial attention map. Supervise this with CLIP's attention       
    distribution via KL divergence. The student attention map is applied as multiplicative spatial weights on
  backbone
     features before MixVPR aggregation.
  Prompt for Code Modification: Add a SpatialAttentionHead module: nn.Sequential(nn.Conv2d(1024, 1, kernel_size=1),
    nn.Flatten(start_dim=2), nn.Softmax(dim=-1)). This takes the student backbone output  [B, 1024, 20, 20] and
    produces a spatial distribution [B, 400]. In training_step: (1)  compute student attention distribution, (2)
    compute KL divergence loss vs CLIP  attention distribution from C1, (3) reshape student attention to [B, 1, 20,
    20] and  multiply elementwise with backbone features, (4) pass weighted features to MixVPR.  Total loss =
    metric_loss + lambda_global * global_distill + lambda_local * kl_loss.  Start with lambda_local=0.1. The
    SpatialAttentionHead parameters go into  configure_optimizers. At inference: the SpatialAttentionHead is retained

    (it's  lightweight), CLIP is discarded.
  ────────────────────────────────────────
  ID: C3                                                                                                             
  Approach: C2 lambda_local sweep. Test lambda_local in {0.01, 0.05, 0.1, 0.2} with best lambda_global from B2/B3.
  Prompt for Code Modification: Same code as C2. Sweep via config.                                                   
                                                        
---
  Phase D: VPR-Oriented Spatial Weight Learning

  ID: D1                                           
  Approach: Cross-image attention consistency loss. For images from the same place (positive pairs within the PxK
    batch), their student spatial attention maps should be similar (same regions matter for the same place). Add a
    consistency regularization: for all positive pairs (i, j), compute L_consist = 1 -  cosine_similarity(attn_i, 
    attn_j), averaged over all positive pairs. This teaches the student that VPR-relevant regions should be stable
    across views of the same place.                                                                                  
  Prompt for Code Modification: In training_step, after computing student spatial attention maps [B, 400]: group 
    images by place (K=4 images per place, P places per batch). For each place, compute  pairwise cosine similarity  
    among its K attention maps, forming a K*K similarity  matrix. The consistency loss = 1 - mean(off-diagonal 
    similarities). Average across all  P places. Add lambda_consist * L_consist to total loss. Config param:
    lambda_consist  (start with 0.05).
  ────────────────────────────────────────
  ID: D2                                                                                                             
  Approach: Negative attention divergence loss. In addition to D1, add a loss that pushes attention maps of different
                                                                                                                     
    places to be dissimilar. For hard negative pairs (identified by the MultiSimilarityMiner), compute L_diverge = 
    max(0, cosine_similarity(attn_i, attn_neg) - margin). This teaches the model that discriminative regions should
    differ between places.
  Prompt for Code Modification: Extend D1: after the miner produces hard negative pairs (indices from
    MultiSimilarityMiner), compute cosine similarity between their attention maps.  L_diverge = mean(relu(cos_sim -
    margin)). margin=0.5. Total loss += lambda_diverge *  L_diverge. Config param: lambda_diverge (start with 0.02).
    This forces the student to  attend to place-specific discriminative regions, not generic salient regions.
  ────────────────────────────────────────
  ID: D3                                                                                                             
  Approach: Multi-head spatial attention (4 heads). Replace the single-channel attention head with 4 independent
    attention heads, each producing a 20x20 weight map. Concatenate the 4 weighted feature maps and adjust MixVPR's  
    input accordingly. Each head can specialize on different semantic aspects (structures, textures, layout, etc.).
  Prompt for Code Modification: Replace SpatialAttentionHead with MultiHeadSpatialAttention: nn.Conv2d(1024,
    num_heads, kernel_size=1) producing [B, 4, 20, 20]. Apply softmax per head. Multiply  backbone features by each
    head's weights to get 4 weighted feature maps, each [B,  1024, 20, 20]. Concatenate along spatial dimension to
  [B,
     1024, 20, 80] OR average the  4 weighted maps. For CLIP supervision: the CLIP attention from C1 supervises the
    mean  of the 4 student heads. Config: num_attention_heads=4. Adjust MixVPR in_w if  concatenating.

---
  Phase E: Full Framework Integration & Tuning
                                                                                                                     
  ID: E1                                                                                                          
  Approach: Best combination from B-D. Assemble the best lambda values and components: global cosine distill (B2/B3) 
  +                                                                                                               
    local KL attention distill (C2/C3) + attention consistency (D1) + optional divergence (D2). Train for full 40 
    epochs.                                                                                                       
    Prompt for Code Modification: Combine configs from best B/C/D experiments. No new code.                            
    ────────────────────────────────────────                                                                        
    ID: E2                                                                                                             
    Approach: Distillation annealing. After epoch 20, linearly decay all distillation lambdas to 0 by epoch 30. The 
    last
    10 epochs train with pure metric learning loss, allowing the model to fully optimize for VPR without distillation
   
    interference.                                                                                                    
    Prompt for Code Modification: Add distillation.anneal_start_epoch and distillation.anneal_end_epoch to config. In 
    training_step, compute anneal_factor = 1.0 if epoch < start, 0.0 if epoch >= end,  linear interpolation
    otherwise.
     Multiply all distillation/consistency lambdas by  anneal_factor.
    ────────────────────────────────────────
    ID: E3                                                                                                             
    Approach: Backbone upgrade: ResNet101. Replace ResNet50 with ResNet101 (same crop_last_block=True). The deeper
    backbone may better absorb the distilled semantic knowledge while maintaining VPR discrimination.                
    Prompt for Code Modification: In config, change backbone_name from resnet50 to resnet101. in_channels stays 1024 
    (layer3 output is same). No code change needed if ResNet class already supports  resnet101.

---
  Phase F: Benchmark Validation
                                                                                                                     
  ID: F1                                           
  Approach: Add Tokyo24/7 evaluation. This is the key benchmark for condition invariance (day/sunset/night queries vs
                                                        
    daytime database). If your semantic distillation works, this is where the gain should appear.
  Prompt for Code Modification: Add a Tokyo247 dataset class in src/dataloaders/valid/. It should load 315 query 
    images and 75,984 database images. Ground truth: each query matches a specific  database place within 25m GPS    
    threshold. Follow the same pattern as pittsburgh.py.  Register "tokyo247" as a valid val_set_name in 
    vpr_datamodule.py.                                                                                               
  ────────────────────────────────────────              
  ID: F2                                                                                                             
  Approach: Add Nordland evaluation. Four-season traversal benchmark. The most extreme appearance-change test.
  Prompt for Code Modification: Add a Nordland dataset class in src/dataloaders/valid/. Load query images from one   
    season (e.g., winter) and database images from another (e.g., summer). Frame-level  correspondence (query i 
    matches database i). Follow pittsburgh.py pattern.
  ────────────────────────────────────────
  ID: F3                                                                                                             
  Approach: Final comparison table. Run A1 baseline and best E-series model on all benchmarks: MSLS-val,
  Pitts30k-val,                                                                                                      
    Tokyo24/7, Nordland. Report R@1/5/10. The semantic distillation should show clear improvement on Tokyo24/7 and
    Nordland while being neutral or positive on MSLS/Pitts30k.
  Prompt for Code Modification: No code change. Run inference with saved checkpoints.

---
  Expected Outcome Targets
                          
  ┌───────────────┬───────────────┬──────────────────────────┐
  │   Benchmark   │ Baseline (A1) │ Target with Distillation │                                                       
  ├───────────────┼───────────────┼──────────────────────────┤
  │ MSLS-val R@1  │ 87.7          │ >= 87.7 (no regression)  │                                                       
  ├───────────────┼───────────────┼──────────────────────────┤
  │ Pitts30k R@1  │ 93.4          │ >= 93.4 (no regression)  │                                                       
  ├───────────────┼───────────────┼──────────────────────────┤                                                       
  │ Tokyo24/7 R@1 │ ~70-75 (est.) │ >= 78 (+3-5%)            │                                                       
  ├───────────────┼───────────────┼──────────────────────────┤                                                       
  │ Nordland R@1  │ ~30-40 (est.) │ >= 40-50 (+10%)          │
  └───────────────┴───────────────┴──────────────────────────┘                                                       
                                                        
  The core principle: semantic distillation should help where appearance changes are extreme, and should not hurt    
  where the baseline already performs well. If gains only appear on MSLS/Pitts30k, the approach is overfitting to
  easy benchmarks. If gains appear on Tokyo24/7/Nordland, the semantic prior is genuinely improving condition        
  invariance.                               