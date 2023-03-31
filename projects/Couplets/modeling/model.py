import oneflow as flow
from oneflow import nn

from libai.layers.cross_entropy import ParallelCrossEntropyLoss
from libai.utils import distributed as dist

from .transformer_model import TransformerModel


class Seq2SeqLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_loss = ParallelCrossEntropyLoss()

    def forward(self, logits, lm_labels):
        logits = logits[:, :-1, :]
        lm_labels = lm_labels[:, 1:]
        lm_loss = self.lm_loss(logits, lm_labels)
        lm_loss = lm_loss.mean()
        return lm_loss


class Seq2Seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.language_model = TransformerModel(cfg)
        self.loss_func = Seq2SeqLoss()

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        encoder_decoder_attn_mask,
    ):
        logits = self.language_model(
            encoder_input_ids,
            decoder_input_ids,
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
        )

        if self.training:
            loss = self.loss_func(logits, decoder_input_ids)
            return {"total_loss": loss}

        logits = logits.view(-1, logits.shape[-1])
        return {"prediction_scores": logits}

    def encode(
        self,
        encoder_input_ids,
        encoder_attn_mask,
    ):
        encoder_input_embeddings = self.language_model.embedding(encoder_input_ids)
        if encoder_attn_mask is not None:
            encoder_extended_attn_mask = self.language_model.extended_attn_mask(encoder_attn_mask)
            encoder_states = self.language_model.encoder(
                encoder_input_embeddings,
                encoder_extended_attn_mask,
            )
        else:
            encoder_states = self.language_model.encoder(
                encoder_input_embeddings,
                None,
            )
        return encoder_states

    def decode(
        self,
        decoder_input_ids,
        decoder_attn_mask,
        encoder_states,
        encoder_decoder_attn_mask,
    ):
        decoder_input_embeddings = self.language_model.embedding(decoder_input_ids)
        decoder_extended_attn_mask = self.language_model.extended_attn_mask(decoder_attn_mask)
        if encoder_decoder_attn_mask is not None:
            encoder_decoder_extended_attn_mask = self.language_model.extended_attn_mask(
                encoder_decoder_attn_mask
            )
            decoder_states = self.language_model.decoder(
                decoder_input_embeddings,
                decoder_extended_attn_mask,
                encoder_states,
                encoder_decoder_extended_attn_mask,
            )
        else:
            decoder_states = self.language_model.decoder(
                decoder_input_embeddings,
                decoder_extended_attn_mask,
                encoder_states,
                None,
            )
        logits = self.language_model.lm_head(decoder_states)
        return logits

    @staticmethod
    def set_pipeline_stage_id(model):
        dist_utils = dist.get_dist_util()

        from .transformer_model import ExtendedMask, TransformerEmbedding, TransformerLayer

        # Set pipeline parallelism stage_id
        if hasattr(model.language_model.lm_head, "config"):
            # Old API in OneFlow 0.8
            for module_block in model.modules():
                # module.origin can get the original module
                if isinstance(module_block.origin, TransformerEmbedding):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.origin, ExtendedMask):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.origin, TransformerLayer):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )

            # Set the lm_head stage id
            model.language_model.lm_head.config.set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
        else:
            for module_block in model.modules():
                if isinstance(module_block.to(nn.Module), TransformerEmbedding):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.to(nn.Module), ExtendedMask):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
                    )
                elif isinstance(module_block.to(nn.Module), TransformerLayer):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )

            # Set the lm_head stage id
            model.language_model.lm_head.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
