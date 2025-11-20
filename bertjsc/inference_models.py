"""
Inference-only model classes that don't depend on PyTorch Lightning.

These classes provide the same interface as the Lightning-based models
but are optimized for inference and don't require pytorch-lightning dependency.
"""
import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from typing import Optional


class BertForMaskedLMInference(nn.Module):
    """
    Inference-only version of BERT for Masked Language Model.

    This class provides the same forward() interface as LitBertForMaskedLM
    but doesn't depend on PyTorch Lightning.
    """

    def __init__(self, card: str):
        """
        Initialize the model.

        Args:
            card: HuggingFace model card name (e.g., 'cl-tohoku/bert-base-japanese-whole-word-masking')
        """
        super().__init__()
        self.mlbert = BertForMaskedLM.from_pretrained(card)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for inference.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Labels (not used in inference but kept for compatibility)

        Returns:
            Output from BertForMaskedLM
        """
        output = self.mlbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return output


class SoftMaskedBertInference(nn.Module):
    """
    Inference-only version of Soft-Masked BERT.

    This class provides the same forward() interface as LitSoftMaskedBert
    but doesn't depend on PyTorch Lightning.
    """

    def __init__(self, card: str, mask_token_id: int, vocab_size: int):
        """
        Initialize the Soft-Masked BERT model.

        Args:
            card: HuggingFace model card name
            mask_token_id: ID of the mask token
            vocab_size: Size of the vocabulary
        """
        super().__init__()
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mlbert = BertForMaskedLM.from_pretrained(card)

        # Word embedding
        self.embeddings = self.mlbert.bert.embeddings

        # Detection neural network
        self.bidirectional_gru = nn.GRU(
            input_size=self.mlbert.config.hidden_size,
            hidden_size=self.mlbert.config.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.linear = nn.Linear(self.mlbert.config.hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

        # Correction neural network
        self.encoder = self.mlbert.bert.encoder
        self.cls = self.mlbert.cls

        # Loss function (kept for compatibility but not used in inference)
        self.det_criterion = nn.BCELoss()
        self.cor_criterion = nn.CrossEntropyLoss()

        # Coefficient
        self.coef = 0.8

    @property
    def device(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_ids: Optional[torch.Tensor] = None,
        det_labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass for inference.

        Args:
            input_ids: Input token IDs
            token_type_ids: Token type IDs
            attention_mask: Attention mask
            output_ids: Output IDs (not used in inference but kept for compatibility)
            det_labels: Detection labels (not used in inference but kept for compatibility)

        Returns:
            MaskedLMOutput with logits
        """
        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )

        # Detection
        gru_output, _ = self.bidirectional_gru(embeddings)
        prob = self.sigmoid(self.linear(gru_output))

        masked_e = self.embeddings(
            torch.tensor([[self.mask_token_id]], dtype=torch.long).to(self.device)
        )
        soft_masked_embeddings = prob * masked_e + (1 - prob) * embeddings

        # Get extended attention mask
        extended_attention_mask: torch.Tensor = self.mlbert.get_extended_attention_mask(
            attention_mask,
            input_ids.size()
        )

        # Correction
        bert_out = self.encoder(
            hidden_states=soft_masked_embeddings,
            attention_mask=extended_attention_mask
        )
        h = bert_out[0] + embeddings

        prediction_scores = self.cls(h)

        # Loss calculation is not needed for inference
        loss = None
        if output_ids is not None and det_labels is not None:
            det_loss = self.det_criterion(prob.squeeze(), det_labels)
            cor_loss = self.cor_criterion(
                prediction_scores.view(-1, self.vocab_size),
                output_ids.view(-1)
            )
            loss = self.coef * cor_loss + (1 - self.coef) * det_loss

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores
        )
