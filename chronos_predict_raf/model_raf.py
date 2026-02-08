import torch
from typing import Optional
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput


class ChronosBoltFiDModel(ChronosBoltModelForForecasting):
    def forward(
        self,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        raf_context: Optional[torch.Tensor] = None,
        raf_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ChronosBoltOutput:
        batch_size = context.size(0)

        hidden_states, loc_scale_tuple, input_embeds, attention_mask = self.encode(
            context=context, mask=mask
        )

        loc, scale = loc_scale_tuple
        scale = torch.clamp(scale, min=1.0)
        loc_scale = (loc, scale)

        if raf_context is not None:
            has_valid = raf_mask.any(dim=-1) if raf_mask is not None else ~torch.isnan(raf_context).all(dim=-1)
            if has_valid.any():
                raf_hidden, raf_loc_scale, _, raf_attn_mask = self.encode(
                    context=raf_context, mask=raf_mask
                )
                hidden_states = torch.cat([hidden_states, raf_hidden], dim=1)
                attention_mask = torch.cat([attention_mask, raf_attn_mask], dim=1)

        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)

        quantile_preds_shape = (
            batch_size,
            self.num_quantiles,
            self.chronos_config.prediction_length,
        )
        quantile_preds = self.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )

        loss = None
        if target is not None:
            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None
                else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0

            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (
                    *target.shape[:-1],
                    self.chronos_config.prediction_length - target.shape[-1],
                )
                target = torch.cat(
                    [target, torch.zeros(padding_shape).to(target)], dim=-1
                )
                target_mask = torch.cat(
                    [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
                )

            target_norm = ((target - loc) / scale).unsqueeze(1)
            if target_mask.dim() < target_norm.dim():
                target_mask = target_mask.unsqueeze(1)
            target_norm = torch.where(target_mask, target_norm, torch.zeros_like(target_norm))

            loss = (
                2
                * torch.abs(
                    (target_norm - quantile_preds)
                    * (
                        (target_norm <= quantile_preds).float()
                        - self.quantiles.view(1, self.num_quantiles, 1)
                    )
                )
                * target_mask.float()
            )
            loss = loss.mean(dim=-2).mean(dim=-1).mean()

        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1), loc_scale
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)
