# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import math

import numpy as np
import torch
import lightning as L
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import v2 as T2
import src.utils as utils
import yaml
from collections.abc import Mapping

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
        groups = []
        for module in (self.backbone, self.aggregator):
            parameters = [
                parameter
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
        if not groups:
            raise ValueError("model has no trainable backbone/aggregator parameters")
        return groups
    
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
        lambda_alias=0.0,
        lambda_positive=0.0,
        distill_warmup_steps=1500,
        detach_backbone_for_attn=False,
        semantic_region_gate=None,
        semantic_region_target=None,
        lambda_semantic_region=0.0,
        query_semantic_target=None,
        lambda_query_semantic=0.0,
        crop_semantic_target=None,
        lambda_crop_semantic=0.0,
        crop_semantic_diagnostic_interval=100,
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
            ("lambda_alias", lambda_alias),
            ("lambda_positive", lambda_positive),
            ("lambda_semantic_region", lambda_semantic_region),
            ("lambda_query_semantic", lambda_query_semantic),
            ("lambda_crop_semantic", lambda_crop_semantic),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value}"
                )
        if distill_module is None and any(
            value > 0
            for value in (
                lambda_global,
                lambda_region,
                lambda_attn,
                lambda_alias,
                lambda_positive,
            )
        ):
            raise ValueError(
                "distill_module is required when a distillation weight is non-zero"
            )
        if lambda_attn > 0 and spatial_attn_head is None:
            raise ValueError(
                "spatial_attn_head is required when lambda_attn is non-zero"
            )
        if detach_backbone_for_attn and spatial_attn_head is None:
            raise ValueError(
                "detach_backbone_for_attn requires spatial_attn_head"
            )

        self.distill_module = distill_module
        self.spatial_attn_head = spatial_attn_head
        self.lambda_global = lambda_global
        self.lambda_region = lambda_region
        self.lambda_attn = lambda_attn
        self.lambda_alias = lambda_alias
        self.lambda_positive = lambda_positive
        self.distill_warmup_steps = distill_warmup_steps
        self.detach_backbone_for_attn = bool(detach_backbone_for_attn)
        self.semantic_region_gate = semantic_region_gate
        self.semantic_region_target = semantic_region_target
        self.lambda_semantic_region = float(lambda_semantic_region)
        if self.lambda_semantic_region > 0 and (
            self.semantic_region_gate is None or self.semantic_region_target is None
        ):
            raise ValueError(
                "semantic region gate and target are required when its lambda is non-zero"
            )
        self.query_semantic_target = query_semantic_target
        self.lambda_query_semantic = float(lambda_query_semantic)
        if self.lambda_query_semantic > 0 and self.query_semantic_target is None:
            raise ValueError(
                "query semantic target is required when its lambda is non-zero"
            )
        if self.query_semantic_target is not None and not hasattr(
            self.aggregator, "predict_semantics"
        ):
            raise ValueError(
                "query semantic supervision requires an aggregator semantic head"
            )
        self.crop_semantic_target = crop_semantic_target
        self.lambda_crop_semantic = float(lambda_crop_semantic)
        self.crop_semantic_enabled = (
            getattr(self.backbone, "crop_semantic_film", None) is not None
        )
        if self.lambda_crop_semantic > 0 and self.crop_semantic_target is None:
            raise ValueError(
                "crop semantic target is required when its lambda is non-zero"
            )
        if self.crop_semantic_target is not None and not (
            self.crop_semantic_enabled
        ):
            raise ValueError(
                "crop semantic target requires backbone.crop_semantic_film"
            )
        if self.crop_semantic_enabled and self.lambda_crop_semantic <= 0:
            # This 128->512 projection exists only to match the training-time
            # crop teacher. It is outside the retrieval path and would be an
            # unused optimizer parameter in the architecture-only control.
            self.backbone.crop_semantic_film.semantic_projection.requires_grad_(
                False
            )
        if (
            isinstance(crop_semantic_diagnostic_interval, bool)
            or int(crop_semantic_diagnostic_interval) < 1
        ):
            raise ValueError(
                "crop_semantic_diagnostic_interval must be a positive integer"
            )
        self.crop_semantic_diagnostic_interval = int(
            crop_semantic_diagnostic_interval
        )
        self._crop_semantic_zero_start_verified = False
        self._crop_semantic_zero_start_error = None

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "_query_semantic_base_frozen", False):
            # Keep the pretrained RU feature path deterministic while the
            # semantic head/query adapters train. BoQ itself uses zero dropout;
            # its semantic children remain trainable in either module mode.
            self.backbone.eval()
            self.semantic_region_gate.eval()
        return self

    @staticmethod
    def _split_backbone_output(backbone_output):
        """Return local feature map and a function restoring the backbone output."""
        if isinstance(backbone_output, tuple):
            if not backbone_output or not torch.is_tensor(backbone_output[0]):
                raise ValueError("backbone tuple must start with a local feature map")
            return backbone_output[0], lambda local: (local, *backbone_output[1:])
        if isinstance(backbone_output, list):
            if not backbone_output or not torch.is_tensor(backbone_output[0]):
                raise ValueError("backbone list must start with a local feature map")
            return backbone_output[0], lambda local: [local, *backbone_output[1:]]
        return backbone_output, lambda local: local

    def _student_forward(
        self,
        images,
        return_query_semantic=False,
        return_crop_semantic=False,
        crop_semantic_batch_indices=None,
    ):
        """Run the exact student path shared by train, validation and inference."""
        crop_semantic_tokens = None
        crop_semantic_raw_scale = None
        if return_crop_semantic:
            if not self.crop_semantic_enabled or not hasattr(
                self.backbone, "forward_with_crop_semantics"
            ):
                raise RuntimeError(
                    "crop semantic output requested from an incompatible backbone"
                )
            (
                backbone_output,
                crop_semantic_tokens,
                crop_semantic_raw_scale,
            ) = self.backbone.forward_with_crop_semantics(
                images,
                semantic_batch_indices=crop_semantic_batch_indices,
            )
        else:
            backbone_output = self.backbone(images)
        raw_featmap, restore_output = self._split_backbone_output(backbone_output)
        featmap = raw_featmap
        student_attn = None
        semantic_score = None
        semantic_gate = None
        if self.semantic_region_gate is not None:
            featmap, semantic_score, semantic_gate = self.semantic_region_gate(featmap)
        if self.spatial_attn_head is not None:
            featmap, student_attn = self.spatial_attn_head(featmap)
        query_semantic_logits = None
        if getattr(self.aggregator, "semantic_num_classes", None) is not None:
            # Both the cached teacher loss and the VPR path train the semantic
            # student, but neither is allowed to move the frozen/RU feature
            # extractor through this auxiliary branch.
            query_semantic_logits = self.aggregator.predict_semantics(
                featmap.detach()
            )
            model_output = self.aggregator(
                restore_output(featmap), semantic_logits=query_semantic_logits
            )
        else:
            model_output = self.aggregator(restore_output(featmap))
        result = (
            model_output,
            featmap,
            student_attn,
            raw_featmap,
            semantic_score,
            semantic_gate,
        )
        if return_query_semantic:
            result = (*result, query_semantic_logits)
        if return_crop_semantic:
            result = (
                *result,
                crop_semantic_tokens,
                crop_semantic_raw_scale,
            )
        return result

    def _attention_for_distillation(self, raw_featmap, student_attn):
        """Optionally stop the reliability KL from directly moving backbone."""
        if student_attn is None or self.lambda_attn <= 0:
            return None
        if not self.detach_backbone_for_attn:
            return student_attn
        _, detached_attention = self.spatial_attn_head(raw_featmap.detach())
        return detached_attention

    def forward(self, x):
        model_output, _, _, _, _, _ = self._student_forward(x)
        return model_output

    @staticmethod
    def _descriptor_tensor(model_output):
        if isinstance(model_output, (tuple, list)):
            return model_output[0]
        return model_output

    @torch.no_grad()
    def _verify_crop_semantic_zero_start(self, images):
        """Verify the new branch reproduces the loaded RU descriptor at step 0."""

        if not self.crop_semantic_enabled:
            return None
        if self._crop_semantic_zero_start_verified:
            # Keep emitting the verified value. Lightning only flushes
            # training scalars every ``log_every_n_steps``; a metric emitted
            # solely on batch zero can otherwise be absent from TensorBoard.
            return images.new_tensor(self._crop_semantic_zero_start_error)
        film = self.backbone.crop_semantic_film
        if torch.count_nonzero(film.channel_scale.weight).item() or (
            torch.count_nonzero(film.channel_scale.bias).item()
        ):
            raise RuntimeError(
                "crop-semantic FiLM is not zero-initialised before step 0"
            )

        was_training = self.training
        try:
            self.eval()
            with torch.autocast(
                device_type=images.device.type, enabled=False
            ):
                enabled_output = self._student_forward(images.float())[0]
                with film.bypass():
                    bypass_output = self._student_forward(images.float())[0]
            enabled_descriptor = self._descriptor_tensor(enabled_output)
            bypass_descriptor = self._descriptor_tensor(bypass_output)
            max_abs_error = (
                enabled_descriptor.float() - bypass_descriptor.float()
            ).abs().amax()
        finally:
            self.train(was_training)

        if not bool(torch.isfinite(max_abs_error)) or max_abs_error.item() > 1e-6:
            raise RuntimeError(
                "crop-semantic zero-start descriptor check failed: max absolute "
                f"error={max_abs_error.item():.3e}, required <=1e-6"
            )
        self._crop_semantic_zero_start_verified = True
        self._crop_semantic_zero_start_error = float(max_abs_error.item())
        self.print(
            "Crop-semantic zero-start descriptor max_abs_error="
            f"{max_abs_error.item():.3e}"
        )
        return max_abs_error.detach()

    @torch.no_grad()
    def _measure_crop_descriptor_drift(self, images):
        film = self.backbone.crop_semantic_film
        was_training = self.training
        try:
            self.eval()
            enabled_output = self._student_forward(images)[0]
            with film.bypass():
                bypass_output = self._student_forward(images)[0]
        finally:
            self.train(was_training)
        enabled_descriptor = self._descriptor_tensor(enabled_output)
        bypass_descriptor = self._descriptor_tensor(bypass_output)
        delta = enabled_descriptor.float() - bypass_descriptor.float()
        return {
            "crop_film_descriptor_drift_rms": delta.square().mean().sqrt(),
            "crop_film_descriptor_drift_max_abs": delta.abs().amax(),
        }

    def on_after_backward(self):
        super_method = getattr(super(), "on_after_backward", None)
        if callable(super_method):
            super_method()
        if not self.crop_semantic_enabled:
            return
        def gradient_rms(parameters):
            squared_sum = None
            element_count = 0
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float()
                contribution = gradient.square().sum()
                squared_sum = (
                    contribution
                    if squared_sum is None
                    else squared_sum + contribution
                )
                element_count += gradient.numel()
            if squared_sum is None or not element_count:
                return None
            return (squared_sum / element_count).sqrt()

        film = self.backbone.crop_semantic_film
        gradient_groups = {
            "crop_film_grad_rms": film.parameters(),
            "crop_film_channel_scale_grad_rms": (
                film.channel_scale.parameters()
            ),
            "crop_semantic_projection_grad_rms": (
                film.semantic_projection.parameters()
            ),
        }
        for metric_name, parameters in gradient_groups.items():
            value = gradient_rms(parameters)
            if value is not None:
                self.log(
                    metric_name,
                    value,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                )

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
        if self.semantic_region_gate is not None:
            semantic_trainable = [
                p for p in self.semantic_region_gate.parameters() if p.requires_grad
            ]
            if semantic_trainable:
                optimizer_params.append(
                    {
                        "params": semantic_trainable,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
        return optimizer_params

    @staticmethod
    def _unpack_distillation_batch(batch):
        """Unpack legacy, augmented and metadata-aware training batches."""
        metadata = None
        images_aug = None
        if len(batch) == 2:
            images, labels = batch
        elif len(batch) == 3:
            if isinstance(batch[-1], Mapping):
                images, labels, metadata = batch
            else:
                images, images_aug, labels = batch
        elif len(batch) == 4 and isinstance(batch[-1], Mapping):
            images, images_aug, labels, metadata = batch
        else:
            raise ValueError(
                "unexpected training batch; expected (images, labels), "
                "(images, images_aug, labels), or either tuple plus metadata"
            )
        return images, images_aug, labels, metadata

    def training_step(self, batch, batch_idx):
        """Training step with CLIP teacher distillation."""
        images, images_aug, labels, metadata = self._unpack_distillation_batch(
            batch
        )

        P, K, c, h, w = images.shape
        images = images.view(P * K, c, h, w)
        if images_aug is not None:
            images_aug = images_aug.view(P * K, c, h, w)
        place_labels = labels.reshape(P, K)
        labels = labels.view(-1)
        coordinates = None
        teacher_images = None
        years = None
        months = None
        headings = None
        crop_view_indices = None
        if metadata is not None:
            coordinates = metadata.get("coordinates")
            if coordinates is not None:
                coordinates = coordinates.reshape(P * K, 2)
            teacher_images = metadata.get("teacher_images")
            if teacher_images is not None:
                if teacher_images.ndim != 5 or teacher_images.shape[:2] != (P, K):
                    raise ValueError(
                        "metadata.teacher_images must have shape (P,K,3,H,W)"
                    )
                teacher_images = teacher_images.reshape(
                    P * K, *teacher_images.shape[2:]
                )
            years = metadata.get("years")
            months = metadata.get("months")
            headings = metadata.get("headings")
            if years is not None:
                years = years.reshape(P * K)
            if months is not None:
                months = months.reshape(P * K)
            if headings is not None:
                headings = headings.reshape(P * K)
            crop_view_indices = metadata.get("crop_semantic_view_index")

        crop_semantic_batch_indices = None
        if self.lambda_crop_semantic > 0:
            if crop_view_indices is None:
                raise RuntimeError(
                    "crop semantic metadata contains no view index"
                )
            if crop_view_indices.ndim != 1 or crop_view_indices.numel() != P:
                raise ValueError(
                    "metadata.crop_semantic_view_index must have shape (P,)"
                )
            crop_view_indices = crop_view_indices.to(
                device=images.device, dtype=torch.long
            )
            if bool(
                ((crop_view_indices < 0) | (crop_view_indices >= K)).any()
            ):
                raise ValueError(
                    f"crop semantic view indices must be in [0, {K - 1}]"
                )
            crop_semantic_batch_indices = (
                torch.arange(P, device=images.device) * K
                + crop_view_indices
            )

        # Student forward: backbone -> optional spatial gate -> aggregator.
        # ``forward`` uses this same helper, so validation/checkpoint inference
        # cannot silently bypass the phase-C module.
        crop_zero_start_error = self._verify_crop_semantic_zero_start(images)
        if self.lambda_crop_semantic > 0:
            (
                model_output,
                featmap,
                student_attn,
                raw_featmap,
                semantic_score,
                semantic_gate,
                query_semantic_logits,
                crop_semantic_tokens,
                crop_semantic_raw_scale,
            ) = self._student_forward(
                images,
                return_query_semantic=True,
                return_crop_semantic=True,
                crop_semantic_batch_indices=crop_semantic_batch_indices,
            )
            if (
                crop_semantic_tokens is None
                or crop_semantic_raw_scale is None
            ):
                raise RuntimeError(
                    "enabled crop-semantic backbone returned no student tokens"
                )
            crop_film_stats = self.backbone.crop_semantic_diagnostics()
        else:
            (
                model_output,
                featmap,
                student_attn,
                raw_featmap,
                semantic_score,
                semantic_gate,
                query_semantic_logits,
            ) = self._student_forward(images, return_query_semantic=True)
            crop_semantic_tokens = None
            crop_semantic_raw_scale = None
            crop_film_stats = (
                self.backbone.crop_semantic_diagnostics()
                if self.crop_semantic_enabled
                else {}
            )

        descriptors = self._descriptor_tensor(model_output)

        # VPR loss
        loss_vpr, batch_accuracy = self.compute_loss(descriptors, labels)

        semantic_region_stats = {}
        semantic_batch_valid = P >= 2 and K >= 2
        if self.lambda_semantic_region > 0 and semantic_batch_valid:
            semantic_indices = None
            semantic_weights = None
            semantic_confidence = None
            if metadata is not None:
                semantic_indices = metadata.get("semantic_indices")
                semantic_weights = metadata.get("semantic_weights")
                semantic_confidence = metadata.get("semantic_confidence")
                if semantic_indices is not None:
                    semantic_indices = semantic_indices.flatten(0, 1)
                if semantic_weights is not None:
                    semantic_weights = semantic_weights.flatten(0, 1)
                if semantic_confidence is not None:
                    semantic_confidence = semantic_confidence.flatten(0, 1)
            semantic_target, semantic_region_stats = self.semantic_region_target(
                featmap=raw_featmap,
                place_count=P,
                views_per_place=K,
                semantic_indices=semantic_indices,
                semantic_weights=semantic_weights,
                semantic_confidence=semantic_confidence,
            )
            # Target supervision trains the small gate only. The VPR loss is
            # solely responsible for moving DINO through the gated path.
            detached_score = self.semantic_region_gate.predict(raw_featmap.detach())
            loss_semantic_region = F.smooth_l1_loss(
                detached_score, semantic_target
            )
        else:
            loss_semantic_region = loss_vpr.new_zeros(())

        query_semantic_stats = {}
        if self.lambda_query_semantic > 0:
            if query_semantic_logits is None:
                raise RuntimeError(
                    "query semantic loss is active but the aggregator returned no logits"
                )
            if metadata is None:
                raise RuntimeError(
                    "query semantic loss requires cached targets in batch metadata"
                )
            query_labels = metadata.get("query_semantic_labels")
            query_confidence = metadata.get("query_semantic_confidence")
            query_cache_indices = metadata.get("query_semantic_cache_indices")
            if any(
                value is None
                for value in (
                    query_labels,
                    query_confidence,
                    query_cache_indices,
                )
            ):
                raise RuntimeError(
                    "query semantic metadata must contain labels, confidence, "
                    "and stable cache indices"
                )
            query_labels = query_labels.flatten(0, 1)
            query_confidence = query_confidence.flatten(0, 1)
            query_cache_indices = query_cache_indices.flatten(0, 1)
            loss_query_semantic, query_semantic_stats = self.query_semantic_target(
                query_semantic_logits,
                query_labels,
                query_confidence,
                query_cache_indices,
            )
        else:
            loss_query_semantic = loss_vpr.new_zeros(())

        crop_semantic_stats = {}
        if self.lambda_crop_semantic > 0:
            if metadata is None:
                raise RuntimeError(
                    "crop semantic loss requires a clean teacher view in "
                    "batch metadata"
                )
            crop_teacher_images = metadata.get(
                "crop_semantic_teacher_image"
            )
            if crop_teacher_images is None or crop_view_indices is None:
                raise RuntimeError(
                    "crop semantic metadata must contain teacher image and "
                    "view index"
                )
            if (
                crop_teacher_images.ndim != 4
                or crop_teacher_images.shape[:2] != (P, 3)
            ):
                raise ValueError(
                    "metadata.crop_semantic_teacher_image must have shape "
                    "(P,3,H,W)"
                )
            if not bool((place_labels == place_labels[:, :1]).all()):
                raise ValueError(
                    "each crop-semantic batch item must contain K views from "
                    "one place"
                )
            if (
                getattr(self.crop_semantic_target, "mode", None)
                == "wrong_place"
                and torch.unique(place_labels[:, 0]).numel() != P
            ):
                raise ValueError(
                    "wrong_place requires P distinct place labels so rolled "
                    "teacher donors cannot be positives"
                )
            loss_crop_semantic, crop_semantic_stats = (
                self.crop_semantic_target(
                    crop_semantic_tokens,
                    teacher_images=crop_teacher_images,
                    global_step=int(self.trainer.global_step),
                )
            )
        else:
            loss_crop_semantic = loss_vpr.new_zeros(())

        crop_descriptor_stats = {}
        if (
            self.crop_semantic_enabled
            and int(self.trainer.global_step) > 0
            and int(self.trainer.global_step)
            % self.crop_semantic_diagnostic_interval
            == 0
        ):
            crop_descriptor_stats = self._measure_crop_descriptor_drift(
                images
            )

        # Distillation losses. The lambda=0 architecture control skips CLIP
        # entirely while retaining the exact same student inference path.
        distillation_active = self.distill_module is not None and any(
            value > 0
            for value in (
                self.lambda_global,
                self.lambda_region,
                self.lambda_attn,
                self.lambda_alias,
                self.lambda_positive,
            )
        )
        if distillation_active:
            distill_student_attn = self._attention_for_distillation(
                raw_featmap, student_attn
            )
            distill_out = self.distill_module(
                images,
                images_aug,
                featmap,
                descriptors,
                student_attn=distill_student_attn,
                labels=labels,
                coordinates=coordinates,
                teacher_images=teacher_images,
                years=years,
                months=months,
                headings=headings,
                compute_global=self.lambda_global > 0,
                compute_region=self.lambda_region > 0,
            )
        else:
            zero = loss_vpr.new_zeros(())
            distill_out = {
                "loss_global": zero,
                "loss_region": zero,
                "loss_attn": zero,
                "loss_alias": zero,
                "loss_positive": zero,
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
            + warmup_scale * self.lambda_alias * distill_out["loss_alias"]
            + warmup_scale * self.lambda_positive * distill_out["loss_positive"]
            + warmup_scale * self.lambda_semantic_region * loss_semantic_region
            + warmup_scale * self.lambda_query_semantic * loss_query_semantic
            + warmup_scale * self.lambda_crop_semantic * loss_crop_semantic
        )

        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("loss_vpr", loss_vpr, prog_bar=False, logger=True)
        self.log("loss_global_distill", distill_out["loss_global"], prog_bar=False, logger=True)
        self.log("loss_region_distill", distill_out["loss_region"], prog_bar=False, logger=True)
        self.log("loss_attn_distill", distill_out["loss_attn"], prog_bar=False, logger=True)
        self.log(
            "loss_semantic_region",
            loss_semantic_region,
            prog_bar=False,
            logger=True,
        )
        for metric_name, metric_value in semantic_region_stats.items():
            self.log(metric_name, metric_value, prog_bar=False, logger=True)
        self.log(
            "loss_query_semantic",
            loss_query_semantic,
            prog_bar=False,
            logger=True,
        )
        for metric_name, metric_value in query_semantic_stats.items():
            self.log(metric_name, metric_value, prog_bar=False, logger=True)
        self.log(
            "loss_crop_semantic",
            loss_crop_semantic,
            prog_bar=False,
            logger=True,
        )
        for diagnostics in (
            crop_semantic_stats,
            crop_film_stats,
            crop_descriptor_stats,
        ):
            for metric_name, metric_value in diagnostics.items():
                self.log(
                    metric_name,
                    metric_value,
                    prog_bar=False,
                    logger=True,
                )
        if crop_zero_start_error is not None:
            self.log(
                "crop_film_zero_start_max_abs_error",
                crop_zero_start_error,
                prog_bar=False,
                logger=True,
            )
        if query_semantic_logits is not None:
            self.log(
                "query_semantic_logit_std",
                query_semantic_logits.float().std(unbiased=False),
                prog_bar=False,
                logger=True,
            )
            for metric_name, metric_value in (
                self.aggregator.semantic_diagnostics().items()
            ):
                self.log(
                    metric_name,
                    metric_value,
                    prog_bar=False,
                    logger=True,
                )
        if semantic_gate is not None:
            self.log("semantic_gate_std", semantic_gate.float().std(), logger=True)
            self.log(
                "semantic_gate_max_delta",
                (semantic_gate.float() - 1.0).abs().amax(),
                logger=True,
            )
        self.log(
            "loss_semantic_alias",
            distill_out["loss_alias"],
            prog_bar=False,
            logger=True,
        )
        self.log(
            "loss_semantic_positive",
            distill_out["loss_positive"],
            prog_bar=False,
            logger=True,
        )
        self.log("distill_warmup_scale", warmup_scale, prog_bar=False, logger=True)
        self.log(
            "effective_lambda_attn",
            loss_vpr.new_tensor(warmup_scale * self.lambda_attn),
            prog_bar=False,
            logger=True,
        )
        self.log(
            "effective_lambda_alias",
            loss_vpr.new_tensor(warmup_scale * self.lambda_alias),
            prog_bar=False,
            logger=True,
        )
        self.log(
            "effective_lambda_positive",
            loss_vpr.new_tensor(warmup_scale * self.lambda_positive),
            prog_bar=False,
            logger=True,
        )
        self.log(
            "effective_lambda_query_semantic",
            loss_vpr.new_tensor(
                warmup_scale * self.lambda_query_semantic
            ),
            prog_bar=False,
            logger=True,
        )
        self.log(
            "effective_lambda_crop_semantic",
            loss_vpr.new_tensor(
                warmup_scale * self.lambda_crop_semantic
            ),
            prog_bar=False,
            logger=True,
        )
        for metric_name in (
            "reliability_pos_sim",
            "reliability_neg_sim",
            "reliability_margin",
            "reliability_positive_margin_frac",
            "reliability_target_entropy_norm",
            "reliability_target_peak",
            "reliability_target_top20_mass",
            "reliability_hard_negative_place_sim",
            "reliability_valid_anchor_frac",
        ):
            if metric_name in distill_out:
                self.log(
                    metric_name,
                    distill_out[metric_name],
                    prog_bar=False,
                    logger=True,
                )
        for metric_name, metric_value in distill_out.items():
            if metric_name.startswith("semantic_alias_"):
                self.log(
                    metric_name,
                    metric_value,
                    prog_bar=False,
                    logger=True,
                )
            elif metric_name.startswith("semantic_positive_"):
                self.log(
                    metric_name,
                    metric_value,
                    prog_bar=False,
                    logger=True,
                )
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
            mean_attention = attn_fp32.mean(dim=1)
            spatial_size = mean_attention.shape[-1]
            gate_strength = self.spatial_attn_head.gate_strength
            gate = (
                (1.0 - gate_strength)
                + gate_strength * spatial_size * mean_attention
            )
            self.log("student_gate_std", gate.std(), prog_bar=False, logger=True)
            self.log("student_gate_max", gate.amax(), prog_bar=False, logger=True)
        self.log("batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss
