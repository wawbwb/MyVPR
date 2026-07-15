# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import numpy as np
import torch
import lightning as L
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import v2 as T2
import src.utils as utils
import yaml

class VPRFramework(L.LightningModule):
    def __init__(
        self,
        backbone,
        aggregator,
        loss_function,
        lr=1e-4,
        optimizer="adamw",
        weight_decay=1e-3,
        warmup_steps=1500,
        milestones=[5, 10, 15],
        lr_mult=0.25,
        verbose=True,
        config_dict=None,  # configuation to be saved with logs
    ):
        """
        Initializes the VPRFramework class.

        Args:
            backbone: The backbone model.
            aggregator: The aggregator model.
            loss_function: The loss function.
            lr (float, optional): The learning rate. Defaults to 1e-4.
            optimizer (str, optional): The optimizer algorithm. Defaults to "adamw".
            weight_decay (float, optional): The weight decay. Defaults to 1e-3.
            warmup_steps (int, optional): The number of warmup steps. Defaults to 1500.
            milestones (list, optional): The milestones for learning rate scheduling. Defaults to [5, 10, 15].
            lr_mult (float, optional): The learning rate multiplier. Defaults to 0.25.
            verbose (bool, optional): Whether to print verbose information. Defaults to True.
        """
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.loss_function = loss_function
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult
        self.verbose = verbose
        
        # save the hyperparameters except the classes
        # self.save_hyperparameters(ignore=["loss_function", "backbone", "aggregator", "verbose"])
        self.save_hyperparameters(config_dict)
        
    def forward(self, x):
        """
        Forward pass through the backbone then the aggregator.

        Args:
            x: Input tensor.

        Returns:
            Tensor (or list of tensors) after passing through the backbone and aggregator.
        """
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    def _optimizer_param_groups(self):
        return [
            {"params": self.backbone.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.aggregator.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.

        Returns:
            List of optimizers and schedulers that will be used by the Lightning trainer.
        """
        optimizer_params = self._optimizer_param_groups()
        
        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_params,
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(optimizer_params)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Define how a single optimization step is executed.

        Args:
            epoch: Current epoch.
            batch_idx: Current batch index.
            optimizer: Optimizer instance.
            optimizer_closure: Closure for the optimizer.
        """
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * pg["initial_lr"]

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
    
    @torch.compiler.disable() # do not run the compiler on this function
    def compute_loss(self, descriptors, labels):
        """
        Compute the loss.

        Args:
            descriptors: Descriptor tensors.
            labels: Corresponding labels.

        Returns:
            Loss value and batch accuracy.
        """
        # NOTE: in this framework, the loss also returns a batch_accuracy value 
        # which represents the fraction of valid positve pairs in the batch (after mining)
        # this is useful for debugging and monitoring the training process
        # but it is not used in the loss computation nor for comparing models.
        loss, batch_accuracy = self.loss_function(descriptors, labels)
        return loss, batch_accuracy
    
    
    
    def on_train_start(self):
        """
        Actions to perform at the start of training.
        """
        # you can do something here before the training starts
        # let's save the configuration to the log
        # if self.config_dict is not None:
        #     with open(f"{self.logger.log_dir}/config_args.yaml", 'w') as file:
        #         yaml.dump(self.config_dict, file)
    
    ########################################################
    ################ Training loop starts here #############
    ########################################################
    def on_train_epoch_start(self):
        """
        Actions to perform at the start of each training epoch.
        """
        pass
    
    # This is the main training loop
    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Loss value for the batch.
        """
        images, labels = batch
        P, K, c, h, w = images.shape # P: number of places, K: number of views
        images = images.view(P * K, c, h, w) # so B = P * K 
        labels = labels.view(-1)
        
        model_output = self(images)
        
        # sometimes the model returns a list, sometimes a single tensor
        # for example, BoQ returns (descriptors, attentions)
        # but netvlad, mixvpr and many others return only descriptors
        # so we check if the model output is a list or a single tensor
        if isinstance(model_output, tuple) or isinstance(model_output, list):
            descriptors = model_output[0]
        else:
            descriptors = model_output
        
        loss, batch_accuracy = self.compute_loss(descriptors, labels)

        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        """
        Actions to perform at the end of each training epoch.
        """
        pass
    
    ########################################################
    ################ Validation loop starts here ###########
    ########################################################
    def on_validation_epoch_start(self):
        """
        Actions to perform at the start of each validation epoch.
        """
        # we init an empty dictionary to store the descriptors for each dataloader
        self.validation_step_outputs = {}

    # At each iteration, we compute the output descriptors
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step for each batch.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
            dataloader_idx: Index of the dataloader.

        Returns:
            None
        """
        images, labels = batch
        model_output = self(images)
        
        # sometimes the model returns a list, sometimes a single tensor
        # for example, BoQ returns [descriptors, attentions]
        # but netvlad, mixvpr and many others return only descriptors
        # so we check if the model output is a list or a single tensor
        if isinstance(model_output, tuple) or isinstance(model_output, list):
            descriptors = model_output[0]
        else:
            descriptors = model_output
            
        descriptors = descriptors.detach().cpu().numpy()

        if dataloader_idx not in self.validation_step_outputs:
            # initialize the list of descriptors for this dataloader
            self.validation_step_outputs[dataloader_idx] = []
        # save the descriptors to compute the recall@k at the end of the validation epoch
        self.validation_step_outputs[dataloader_idx].append(descriptors)

    # At the end of the validation epoch, we compute the recall@k
    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of each validation epoch.
        """
        dm = self.trainer.datamodule
        list_of_recalls = [] # one list for each validation set
        for dataloader_idx, descriptors_list in self.validation_step_outputs.items():
            descriptors = np.concatenate(descriptors_list, axis=0)
            dataset = dm.val_datasets[dataloader_idx]

            if self.trainer.fast_dev_run:
                # skip the recall computation for fast dev runs
                if dataloader_idx == 0:
                    print("\nFast dev run: skipping recall@k computation\n")
            else:
                # we will use the descriptors, the number of references, number of queries, and the ground truth
                # NOTE: make sure these are available in the dataset object and ARE IN THE RIGHT ORDER.
                # meaning that the first `num_references` descriptors are reference images and the rest are query images
                recalls_dict = utils.compute_recall_performance(
                        descriptors, 
                        dataset.num_references,
                        dataset.num_queries,
                        dataset.ground_truth,
                        k_values=[1, 5, 10, 15]
                )
                recalls_log = {
                    f"{dm.val_set_names[dataloader_idx]}/R1": recalls_dict[1],
                    f"{dm.val_set_names[dataloader_idx]}/R5": recalls_dict[5],
                }
                self.log_dict(recalls_log, prog_bar=False, logger=True)
                list_of_recalls.append(recalls_dict)

        if self.verbose:
            utils.display_recall_performance(list_of_recalls, dm.val_set_names)
        self.validation_step_outputs.clear()


class VPRFrameworkDistill(VPRFramework):
    def __init__(
        self,
        backbone,
        aggregator,
        loss_function,
        lr=1e-4,
        optimizer="adamw",
        weight_decay=1e-3,
        warmup_steps=1500,
        milestones=[5, 10, 15],
        lr_mult=0.25,
        verbose=True,
        config_dict=None,  # configuation to be saved with logs
        distill_module=None,
        spatial_attn_head=None,
        lambda_global=0.1,
        lambda_region=0.05,
        lambda_attn=0.0,
        distill_warmup_steps=1500,
    ):
        super().__init__(
            backbone=backbone,
            aggregator=aggregator,
            loss_function=loss_function,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            milestones=milestones,
            lr_mult=lr_mult,
            verbose=verbose,
            config_dict=config_dict,
        )
        for name, value in (
            ("lambda_global", lambda_global),
            ("lambda_region", lambda_region),
            ("lambda_attn", lambda_attn),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if distill_module is None and any(
            value > 0 for value in (lambda_global, lambda_region, lambda_attn)
        ):
            raise ValueError(
                "distill_module is required when a distillation weight is non-zero"
            )
        if lambda_attn > 0 and spatial_attn_head is None:
            raise ValueError(
                "spatial_attn_head is required when lambda_attn is non-zero"
            )

        self.distill_module = distill_module
        self.spatial_attn_head = spatial_attn_head
        self.lambda_global = lambda_global
        self.lambda_region = lambda_region
        self.lambda_attn = lambda_attn
        self.distill_warmup_steps = distill_warmup_steps

    def _student_forward(self, images):
        """Run the exact student path shared by train, validation and inference."""
        featmap = self.backbone(images)
        student_attn = None
        if self.spatial_attn_head is not None:
            featmap, student_attn = self.spatial_attn_head(featmap)
        model_output = self.aggregator(featmap)
        return model_output, featmap, student_attn

    def forward(self, x):
        model_output, _, _ = self._student_forward(x)
        return model_output

    def _optimizer_param_groups(self):
        optimizer_params = super()._optimizer_param_groups()
        if self.distill_module is not None:
            distill_trainable = [
                p for p in self.distill_module.parameters() if p.requires_grad
            ]
        else:
            distill_trainable = []
        if distill_trainable:
            optimizer_params.append(
                {"params": distill_trainable, "lr": self.lr, "weight_decay": self.weight_decay}
            )
        if self.spatial_attn_head is not None:
            spatial_trainable = [
                p for p in self.spatial_attn_head.parameters() if p.requires_grad
            ]
            if spatial_trainable:
                optimizer_params.append(
                    {
                        "params": spatial_trainable,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
        return optimizer_params

    def training_step(self, batch, batch_idx):
        """Training step with CLIP teacher distillation."""
        # Unpack: batch may contain an augmented view
        if len(batch) == 3:
            images, images_aug, labels = batch
        else:
            images, labels = batch
            images_aug = None

        P, K, c, h, w = images.shape
        images = images.view(P * K, c, h, w)
        if images_aug is not None:
            images_aug = images_aug.view(P * K, c, h, w)
        labels = labels.view(-1)

        # Student forward: backbone -> optional spatial gate -> aggregator.
        # ``forward`` uses this same helper, so validation/checkpoint inference
        # cannot silently bypass the phase-C module.
        model_output, featmap, student_attn = self._student_forward(images)

        if isinstance(model_output, (tuple, list)):
            descriptors = model_output[0]
        else:
            descriptors = model_output

        # VPR loss
        loss_vpr, batch_accuracy = self.compute_loss(descriptors, labels)

        # Distillation losses. The lambda=0 architecture control skips CLIP
        # entirely while retaining the exact same student inference path.
        distillation_active = self.distill_module is not None and any(
            value > 0
            for value in (self.lambda_global, self.lambda_region, self.lambda_attn)
        )
        if distillation_active:
            distill_out = self.distill_module(
                images,
                images_aug,
                featmap,
                descriptors,
                student_attn=(student_attn if self.lambda_attn > 0 else None),
                compute_global=self.lambda_global > 0,
                compute_region=self.lambda_region > 0,
            )
        else:
            zero = loss_vpr.new_zeros(())
            distill_out = {
                "loss_global": zero,
                "loss_region": zero,
                "loss_attn": zero,
            }

        # Linear warmup for distillation weights
        if self.distill_warmup_steps > 0:
            warmup_scale = min(1.0, float(self.trainer.global_step) / self.distill_warmup_steps)
        else:
            warmup_scale = 1.0

        loss = (
            loss_vpr
            + warmup_scale * self.lambda_global * distill_out["loss_global"]
            + warmup_scale * self.lambda_region * distill_out["loss_region"]
            + warmup_scale * self.lambda_attn * distill_out["loss_attn"]
        )

        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("loss_vpr", loss_vpr, prog_bar=False, logger=True)
        self.log("loss_global_distill", distill_out["loss_global"], prog_bar=False, logger=True)
        self.log("loss_region_distill", distill_out["loss_region"], prog_bar=False, logger=True)
        self.log("loss_attn_distill", distill_out["loss_attn"], prog_bar=False, logger=True)
        self.log("distill_warmup_scale", warmup_scale, prog_bar=False, logger=True)
        if student_attn is not None:
            attn_fp32 = student_attn.float().clamp_min(1e-8)
            attn_entropy = -(attn_fp32 * attn_fp32.log()).sum(dim=-1).mean()
            self.log("student_attn_entropy", attn_entropy, prog_bar=False, logger=True)
            self.log(
                "student_attn_peak",
                attn_fp32.amax(dim=-1).mean(),
                prog_bar=False,
                logger=True,
            )
        self.log("batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss
