from typing import Optional, Tuple, Any, Dict

import numpy as np
import torch
import torch.nn as nn
import pdb
import math
import pickle
import gcsfs

from src.models.modules.base_module import BaseModule
from src.utils.utils import (
    delete_module,
    find_module_shape,
    get_parent_module_and_attr,
    reset_parameters,
)
from src.utils.file_utils import open_local_or_remote
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.experimental.modules.rotary_position_encoding import (
    RotaryTransformerEncoder,
    RotaryTransformerEncoderLayer
)
from itertools import product



class DenseDiscreteDiffusionModule(BaseModule):
    """Module for discrete diffusion models."""

    def __init__(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_function: torch.nn.Module,
        evaluator: Optional[object],
        num_hierarchies: int,
        vocab_size: int,
        embedding_dim: int,
        padding_token_id: int,
        embedding_path: str,
        dense_retrieval_loss_function: Optional[torch.nn.Module] = None,
        dense_retrieval_evaluator: Optional[object] = None,
        dense_retrieval: Optional[dict] = None,
        item_embedding_projection: Optional[torch.nn.Module] = None,
        normalization: Optional[bool] = True,
        positional_embedding: Optional[torch.nn.Module] = None,
        attend_to_padding: Optional[bool] = True,
        data_freqs_path: Optional[str] = None,
        projection: Optional[bool] = True,
        codebooks: Optional[torch.Tensor] = None,
        diffusion_config: Optional[dict] = None,
        training_loop_function: Optional[callable] = None,
        use_rotary_position_encoding: Optional[bool] = False,
        max_position_embeddings: Optional[int] = 2048,
        eval_hierarchy_cutoff: Optional[int] = 1,
        update_frequency: Optional[int] = 1000,
        **kwargs,
    ) -> None:

        my_evaluator = evaluator
        if diffusion_config['inference_type'] == "dense-retrieval":
            my_evaluator = dense_retrieval_evaluator
        
        
        super().__init__(
            model=model,
            optimizer=optimizer, 
            scheduler=scheduler,
            loss_function=loss_function,
            evaluator=my_evaluator,
            training_loop_function=training_loop_function
        )
        
        self.positional_embedding = positional_embedding
        self.use_rotary_position_encoding = use_rotary_position_encoding
        self.item_embedding_projection = item_embedding_projection
                                                                 
        self.transformed_codebooks = None
        self.diffusion_config = diffusion_config
        self.num_hierarchies = num_hierarchies
        self.projection = projection
        self.attend_to_padding = attend_to_padding
        self.eval_hierarchy_cutoff = eval_hierarchy_cutoff
        self.normalization = normalization
        self.update_frequency = update_frequency
        
        self.dense_retrieval = dense_retrieval
        self.dense_retrieval_loss_function = dense_retrieval_loss_function
        self.dense_retrieval_evaluator = dense_retrieval_evaluator
        
        self.num_embeddings_per_hierarchy = vocab_size + 1                  
        self.embedding_dim = embedding_dim

        fs = gcsfs.GCSFileSystem()
        self.freqs = None
        self.sorted_freqs = None
        if data_freqs_path is not None:
            with open(data_freqs_path, "rb") as f:
                data = pickle.load(f)

            self.items = torch.Tensor(data["items"])
            self.freqs = torch.Tensor(data["freqs"])
            self.sorted_freqs = self.freqs.clone()
            self.sorted_freqs = self.sorted_freqs.sort().values
        
        self.item_embeddings = torch.load(open_local_or_remote(embedding_path, "rb"))
        self.masking_token_id = vocab_size
        self.padding_token_id = padding_token_id
        self.saved_batch = None
        self.vocab_size = vocab_size
        
        self.codebooks = codebooks.t()
        assert (
            self.codebooks.size(1) == num_hierarchies
        ), self.codebooks.shape                                                      
                
        self.item_sid_embedding_table_encoder = torch.nn.Embedding(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies + 1,                        
            embedding_dim=embedding_dim
        )

    def encoder_output_to_loss(
        self,
        encoder_output: torch.Tensor,
        labels: torch.Tensor,
        label_locations: torch.Tensor,
    ):
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        query_embeddings = encoder_output[
            label_locations[:, 0], label_locations[:, 1]
        ]
        key_embeddings = self.item_sid_embedding_table_encoder.weight

        if self.loss_function.normalize:
            query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=-1)
            key_embeddings = torch.nn.functional.normalize(key_embeddings, dim=-1)
        
        logits = torch.mm(query_embeddings, key_embeddings.t())
        loss = loss_fn(logits, labels)
        return loss

    def forward(
        self,
        input: SequentialModelInputData, 
        item_id_seqs: Optional[torch.Tensor] = None,
        give_half_way_embedding: Optional[bool] = False):
        """ Forward pass for the discrete diffusion model.
        """
        bs, seq_len = input.shape
        encoded_input = self.item_sid_embedding_table_encoder(input)                                                
        
        if self.use_rotary_position_encoding:
                                                                                           
                                                                   
            inputs_emb = encoded_input
        else:
            position_ids = torch.arange(seq_len, device=input.device).unsqueeze(0).expand(bs, -1)
            pos_embeddings = self.positional_embedding(position_ids)
            
            inputs_emb = encoded_input + pos_embeddings
        
        padding_mask = None
        if not self.attend_to_padding:
            new_padding_value = self.num_embeddings_per_hierarchy * self.num_hierarchies
            padding_mask = (input == new_padding_value)
        
        if self.dense_retrieval['input_item_text_projection'] is not None:
            assert item_id_seqs is not None, "Item id mask should not be none"
            item_id_pads_mask = item_id_seqs == -1
            safe_item_id_seqs = item_id_seqs.clone().clamp(min=0)
            
            item_text_embeddings = self.item_embeddings.to(self.device)[safe_item_id_seqs]
            item_text_embeddings[item_id_pads_mask] = 0.0
            self.dense_retrieval['input_item_text_projection'] = self.dense_retrieval['input_item_text_projection'].to(self.device)
            projected_item_text_embeddings = self.dense_retrieval['input_item_text_projection'](item_text_embeddings)
            projected_item_text_embeddings = projected_item_text_embeddings.reshape(bs, seq_len, -1)

            if self.dense_retrieval['zero_out_sid_mask_projection']:
                mask = (input == self.masking_token_id).unsqueeze(-1)                           
                projected_item_text_embeddings = projected_item_text_embeddings * (~mask)

            inputs_emb = inputs_emb + projected_item_text_embeddings

            inputs_emb = self.dense_retrieval['input_layernorm'].to(self.device)(inputs_emb)
            inputs_emb = self.dense_retrieval['input_dropout'].to(self.device)(inputs_emb)

        if give_half_way_embedding:
            output, half_way_embedding = self.model(
                inputs_emb, src_key_padding_mask=padding_mask, 
                give_half_way_embedding=True, extract_lyr=self.dense_retrieval['extract_embed_from_lyr']
                )
        else:
            output = self.model(inputs_emb, src_key_padding_mask=padding_mask)

        if self.normalization:
            output = output / math.sqrt(self.embedding_dim)

        if self.dense_retrieval['embedding_normalization_factor'] and give_half_way_embedding:
            half_way_embedding = half_way_embedding / math.sqrt(self.embedding_dim)

        if give_half_way_embedding:
            return output, half_way_embedding
        else:
            return output
        


    def get_dense_retrieval_loss(
        self,
        encoder_output: torch.Tensor,
        transformed_masked_input: torch.Tensor,
        transformed_labels: torch.Tensor,
        label_locations: torch.Tensor,
        halfway_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs, seq_len, emb_dim = encoder_output.shape
        self.transformed_codebooks, _ = self.transform_input_and_labels(
            self.codebooks.to(self.device), None, None        
        )
        transformed_labels_reshaped = transformed_labels.view(-1, self.num_hierarchies)
        matches = (transformed_labels_reshaped.unsqueeze(1) == self.transformed_codebooks.to(self.device).unsqueeze(0))
        item_ids = torch.where(matches.all(dim=2))[1]

        concated_embeddings = halfway_output[label_locations[:, 0], label_locations[:, 1]]
        concated_embeddings = concated_embeddings.reshape(-1, emb_dim * self.num_hierarchies)
        
        query_embeddings = self.item_embedding_projection(concated_embeddings)
        return self.dense_retrieval_loss_function(
            query_embeddings=query_embeddings,
            key_embeddings=self.item_embeddings.to(self.device),
            labels=item_ids,
        )


    def model_step(
        self,
        model_input: torch.Tensor,
        label_data: torch.Tensor,
        fractions: torch.Tensor,
        item_id_seqs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the discrete diffusion model.
        """
        masked_input = model_input.transformed_sequences['sequence_data']
        labels = label_data.labels['sequence_data']
        label_locations = label_data.label_location['sequence_data']

                                                                   
        transformed_masked_input, transformed_labels = self.transform_input_and_labels(
            masked_input, labels, label_locations
        )
        masked_locations = (transformed_masked_input == self.masking_token_id)

        encoder_output, halfway_output = self.forward(transformed_masked_input, item_id_seqs, give_half_way_embedding=True)
        weights = 1.0 / fractions[ torch.where(masked_locations)[0] ]

        loss = self.loss_function(
            query_embeddings=encoder_output, 
            key_embeddings=self.item_sid_embedding_table_encoder.weight,
            label_locations=label_locations,
            labels=transformed_labels,
            weights=weights,
            )

        avg_loss = loss / torch.sum(weights)
        
        if self.dense_retrieval_loss_function is not None and self.trainer.global_step % self.dense_retrieval['update_frequency'] == 0:
            avg_loss += self.dense_retrieval['loss_weight'] * self.get_dense_retrieval_loss(
                encoder_output=encoder_output,
                transformed_masked_input=transformed_masked_input,
                transformed_labels=transformed_labels,
                label_locations=label_locations,
                halfway_output=halfway_output
                )
        else:
            concated_embeddings = torch.zeros(masked_input.shape[0], self.embedding_dim * self.num_hierarchies, device=self.device)
            
            concated_embeddings = concated_embeddings.reshape(masked_input.shape[0], -1)
            dummy_projection = self.item_embedding_projection(concated_embeddings)
                                                                   
            avg_loss = avg_loss + 0.0 * dummy_projection.sum()
        
        return encoder_output, avg_loss

    def encoder_output_to_probabilities(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Convert encoder output to probabilities using dot product with embedding weights.
        """        
                                                                                    
        key_embeddings = self.item_sid_embedding_table_encoder.weight
        if self.loss_function.normalize:
            encoder_output = torch.nn.functional.normalize(encoder_output, dim=-1)
            key_embeddings = torch.nn.functional.normalize(key_embeddings, dim=-1)

        logits = torch.matmul(encoder_output, key_embeddings.t()) / self.loss_function.tau
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    
    def find_row_with_sequence(self, target_sequence: torch.Tensor) -> torch.Tensor:
        """
        Find which row in self.codebooks contains the target sequence.
        Args: target_sequence: Tensor of shape [4] containing the four numbers to search for
        Returns: Tensor containing the row indices where the sequence is found
        """
        if target_sequence.dim() == 1:
            target_sequence = target_sequence.unsqueeze(0)          
        target_sequence = target_sequence.to(self.codebooks.device)     
        matches = (self.codebooks == target_sequence)
        
        row_matches = matches.all(dim=1)           
        matching_indices = torch.where(row_matches)[0]
        return matching_indices

    
    def process_masked_input(self, masked_input: torch.Tensor) -> torch.Tensor:
        """
        Process masked input by replacing padding tokens (-1) with new calculated padding value.
        """
        new_padding_value = self.num_embeddings_per_hierarchy * self.num_hierarchies
        
        processed_input = masked_input.clone()
        processed_input[masked_input == self.padding_token_id] = new_padding_value
        
        return processed_input

    def get_modified_eval_inputs(
        self,
        masked_input: torch.Tensor,
        label_locations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate modified masked input and label locations for evaluation steps.

        This function expands label locations to cover self.num_hierarchies consecutive positions
        (corresponding to the self.num_hierarchies hierarchies) and sets masking tokens at those positions.

        """
        modified_masked_input = masked_input.clone()
        row_indices = label_locations[::self.num_hierarchies, 0]                              
        col_indices = label_locations[::self.num_hierarchies, 1]

        offsets = torch.arange(self.num_hierarchies, device=self.device)                                       
        new_col_indices = col_indices.unsqueeze(1) + offsets.unsqueeze(0)                        
                                                          
        new_row_indices = row_indices.unsqueeze(1).repeat(1, self.num_hierarchies)                        
        modified_label_locations = torch.stack([
            new_row_indices.flatten(),
            new_col_indices.flatten()
        ], dim=1)
        
                                                                       
        modified_masked_input[new_row_indices.flatten(), new_col_indices.flatten()] = self.masking_token_id
        
        return modified_masked_input, modified_label_locations


    def get_item_id_seqs(self, model_input: SequentialModelInputData) -> torch.Tensor:
        """Extract item ID sequences from the model input data.
        """
        sid_seqs = model_input.transformed_sequences['sequence_data']
        bs, seq_len = sid_seqs.shape
        sid_seqs = sid_seqs.reshape(bs, -1, self.num_hierarchies)
        matches = (sid_seqs.unsqueeze(2) == self.codebooks.to(self.device).unsqueeze(0).unsqueeze(0))
        
        item_matches = matches.all(dim=3)
        item_ids = torch.argmax(item_matches.float(), dim=2)
        no_match_mask = torch.sum(item_matches, dim=2) == 0
        final_item_ids = torch.where(no_match_mask, torch.tensor(-1, dtype=torch.long), item_ids.to(dtype=torch.long))

        return final_item_ids


    def training_step(
        self,
        batch: Tuple[Tuple[SequentialModelInputData, SequentialModuleLabelData]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
        """                                    
        
        batch = batch[0]
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
                                        
        item_id_seqs = None
        item_id_seqs = self.get_item_id_seqs(model_input)
        model_input, label_data, fractions = self.mask_input_and_labels(model_input, label_data)
        model_output, loss = self.model_step(model_input, label_data, fractions, item_id_seqs=item_id_seqs)

        if self.training_loop_function is not None:
            self.training_loop_function(self, loss)

        return loss


    def project_generated_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """
        For each generated_id, if it exists in transformed_codebooks, keep it as is.
        Otherwise, replace it with the codebook entry with maximum similarity (dot product).
        Args:
            generated_ids: Tensor of shape (bs, num_candidates, num_hierarchies)
        Returns:
            projected_ids: Tensor of same shape as generated_ids
        """
        if self.transformed_codebooks is None:
            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )
        bs, num_candidates, num_hierarchies = generated_ids.shape
        codebooks = self.transformed_codebooks                                    
                                           
        flat_gen = generated_ids.view(-1, num_hierarchies)                                        
        codebooks_exp = codebooks.unsqueeze(0)                                       
        flat_gen_exp = flat_gen.unsqueeze(1)                                             
                               
        matches = (flat_gen_exp == codebooks_exp).sum(dim=2)                                      
        match_idx = matches.float().argmax(dim=1)                        
        projected_ids_flat = codebooks[match_idx]
        return projected_ids_flat.reshape(bs, num_candidates, num_hierarchies)


    def generative_retrieval_eval(
        self, generated_ids, masked_input, labels, label_locations, 
        modified_masked_input, modified_label_locations, 
        transformed_labels, final_probs, model_input, label_data
        ):
        
        
        if self.freqs is not None:
            bs = generated_ids.shape[0]
            labels_reshaped = labels.view(bs, self.num_hierarchies)
            matches = (labels_reshaped.unsqueeze(1) == self.codebooks.to(self.device).unsqueeze(0))
            row_matches = matches.all(dim=2)
            label_indices = row_matches.float().argmax(dim=1)
            label_freqs = self.freqs.to(self.device).clone().detach()[label_indices.to(self.device)]

        first_sid_label_locations = label_locations[0::self.num_hierarchies, 1].clone().detach()
        self.evaluator(
            marginal_probs=final_probs,
            generated_ids=generated_ids,
            labels=transformed_labels,
            label_freqs=label_freqs if self.freqs is not None else None,
            sorted_freqs=self.sorted_freqs,
            first_sid_label_locations=first_sid_label_locations,
            max_seq_len=masked_input.shape[1]
        )

    def eval_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        loss_to_aggregate: Optional[torch.Tensor] = None,
    ):
        """
        Evaluation step for the discrete diffusion model.
        """
        
        model_input, label_data = batch
        
        masked_input = model_input.transformed_sequences['sequence_data']
        labels = label_data.labels['sequence_data']
        label_locations = label_data.label_location['sequence_data']

                                                   
        modified_masked_input, modified_label_locations = self.get_modified_eval_inputs(
            masked_input, label_locations
        )
                                      
        model_input.transformed_sequences['sequence_data'] = modified_masked_input
        label_data.label_location['sequence_data'] = modified_label_locations
                                                                   
        transformed_masked_input, transformed_labels = self.transform_input_and_labels(
            modified_masked_input, labels, modified_label_locations
        )

        if self.diffusion_config['inference_type'] == "beam-search-generation":
            generated_ids, final_probs = self.beam_search_generation(
                model_input, label_data
            )
            if self.projection:
                generated_ids = self.project_generated_ids(generated_ids)

        
        if self.diffusion_config['inference_type'] == "dense-retrieval":
            bs = transformed_masked_input.shape[0]
            item_id_seqs = self.get_item_id_seqs(model_input)
            
            encoder_output, halfway_output = self.forward(transformed_masked_input, item_id_seqs=item_id_seqs, give_half_way_embedding=True)
            generated_embeddings = halfway_output
            
            input_embeddings = generated_embeddings[label_locations[:, 0], label_locations[:, 1]].reshape(bs, -1)
            input_embeddings = input_embeddings.reshape(bs, -1)
            query_embeddings = self.item_embedding_projection(input_embeddings)

            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )

            transformed_labels_reshaped = transformed_labels.view(bs, self.num_hierarchies)
            matches = (transformed_labels_reshaped.unsqueeze(1) == self.transformed_codebooks.to(self.device).unsqueeze(0))
            item_ids = torch.where(matches.all(dim=2))[1]             

        if self.diffusion_config['inference_type'] == "beam-search-generation":
            self.generative_retrieval_eval(
                generated_ids, masked_input, labels, label_locations, 
                modified_masked_input, modified_label_locations, 
                transformed_labels, final_probs, model_input, label_data
                )
        if self.diffusion_config['inference_type'] == "dense-retrieval":
            self.evaluator(
                query_embeddings=query_embeddings.to(self.device), 
                key_embeddings=self.item_embeddings.to(self.device), 
                labels=item_ids.to(self.device), 
            )


    def transform_input_and_labels(
        self, 
        masked_input: torch.Tensor, 
        labels: torch.Tensor, 
        label_locations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform both masked input and labels by applying hierarchical offsets.
        
        Args:
            masked_input: Input tensor with masked values
            labels: Target labels tensor
            label_locations: Tensor with shape [N, 2] where [:, 0] are batch indices 
                           and [:, 1] are sequence positions
                           
        Returns:
            Tuple of (transformed_masked_input, transformed_labels)
        """
                                                        
        if labels is not None:
            assert masked_input.max() < self.num_embeddings_per_hierarchy, "masked_input contains invalid token IDs"
        masked_input = self.process_masked_input(masked_input)

        bs, seq_len = masked_input.shape
        offsets = torch.arange(seq_len, device=self.device) % self.num_hierarchies * self.num_embeddings_per_hierarchy
        
        new_padding_value = self.num_embeddings_per_hierarchy * self.num_hierarchies
        mask = (masked_input != self.masking_token_id) & (masked_input != new_padding_value)
        addition = (mask * offsets).to(self.device)

        transformed_masked_input = masked_input + addition

        if labels is None or label_locations is None:
            return transformed_masked_input, None
        
                                                                        
        transformed_labels = labels.clone()
        if len(label_locations) > 0:
            batch_indices = label_locations[:, 0]  
            seq_positions = label_locations[:, 1] 
            
            label_offsets = seq_positions % self.num_hierarchies * self.num_embeddings_per_hierarchy
            transformed_labels += label_offsets.to(self.device)
        
        return transformed_masked_input, transformed_labels


    def get_expanded_candidates(
        self, cur_beam_candidate, cur_beam_values, num_candidates, item_id_seqs,
        num_mask_per_row, tokens_to_unmask, labels, label_locations
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expands the current beam candidates by enumerating combinations of
        top token predictions for a selected set of masked positions.
        
        For each beam entry:
        - Run the encoder on the current (partially-masked) sequence to get
          per-position probability distributions over the vocabulary.
        - Choose which masked positions to ``transfer`` (unmask) according to
          the configured `unmasking_type` (random, top-prob, left-to-right).
        - For the selected positions, take the `num_candidates` highest-prob
          tokens per position.
        - For each combination, construct a new sequence where the selected
          masked positions are filled with the chosen tokens, and compute the
          candidate score as the product of the per-token probabilities
          multiplied by the originating beam probability.
        
        Returns:
        - expanded_candidates: tensor shape (bs, cur_beam_size * num_combinations, seq_len)
          containing newly created sequences for every beam and candidate combo.
        - expanded_values: tensor shape (bs, cur_beam_size * num_combinations)
          holding the corresponding scores/probabilities.
        """
        
        bs, cur_beam_size, seq_len = cur_beam_candidate.shape
        num_combinations = num_candidates ** tokens_to_unmask
        expanded_candidates = torch.zeros(
            (bs, cur_beam_size * num_combinations, seq_len), 
            dtype=torch.long, device=self.device
        )
        expanded_values = torch.zeros(
            (bs, cur_beam_size * num_combinations), 
            dtype=torch.float32, device=self.device
        )
        if tokens_to_unmask > 1:
            assert self.diffusion_config['unmasking_type'] == 'left-to-right', "unmasking_type must be left-to-right when tokens_to_unmask > 1"

        for i in range(cur_beam_size):
            cur_seq = cur_beam_candidate[:, i, :]
            encoder_output = self.forward(cur_seq, item_id_seqs, give_half_way_embedding=False)
            probabilities = self.encoder_output_to_probabilities(encoder_output)

            if self.diffusion_config['maskout_masking_token_prob_decoding']:                                                
                probabilities[:, :, self.masking_token_id] = 0.0

            cur_mask = (cur_seq == self.masking_token_id)
            x0 = torch.zeros_like(cur_seq[cur_mask], device=self.device, dtype=torch.long) + self.masking_token_id

            transfer_positions = torch.zeros_like(cur_mask, dtype=torch.bool, device=self.device)
            batch_indices, seq_indices = torch.where(cur_mask)
            
                                                                 
            if self.diffusion_config['unmasking_type'] == 'random':
                random_offsets = torch.randint(0, num_mask_per_row, (bs, tokens_to_unmask), device=self.device)
            elif (
                self.diffusion_config['unmasking_type'] == 'top-prob'
                ):
                                                                   
                top_values = probabilities[cur_mask].max(dim=-1).values.reshape(bs, -1)
                                                                       
                temperature = self.diffusion_config['unmasking_temperature']
                logits = top_values / temperature                          
                probs = torch.softmax(logits, dim=1)
                random_offsets = torch.multinomial(probs, tokens_to_unmask)                          
            
            elif self.diffusion_config['unmasking_type'] == 'left-to-right':
                random_offsets = torch.arange(tokens_to_unmask, device=self.device, dtype=torch.long).unsqueeze(0).expand(bs, tokens_to_unmask).clone()
            
            selected_indices = torch.arange(bs, device=self.device)[:, None] * num_mask_per_row + random_offsets
            selected_indices = selected_indices.flatten()

            selected_batch_indices = batch_indices[selected_indices]
            selected_seq_indices = seq_indices[selected_indices]
            
            transfer_positions[selected_batch_indices, selected_seq_indices] = True
                                                                     
            transfer_index_t_s = transfer_positions[cur_mask]
            probs_at_transfer = probabilities[cur_mask][transfer_index_t_s]
            probs_at_transfer = probs_at_transfer.reshape(bs, tokens_to_unmask, -1)
            
            assert num_candidates <= probs_at_transfer.shape[-1], f"num_candidates {num_candidates} exceeds vocab size {probs_at_transfer.shape[-1]}"
            topk_values, topk_indices = torch.topk(probs_at_transfer, num_candidates, dim=-1)                                                                                                                   
            candidate_combinations = list(product(range(num_candidates), repeat=tokens_to_unmask))                                        

            for combo_idx, combo in enumerate(candidate_combinations):
                                                                                                  
                indices = []
                values = []
                for t in range(tokens_to_unmask):
                    indices.append(topk_indices[:, t, combo[t]])         
                    values.append(topk_values[:, t, combo[t]])           
                
                indices = torch.stack(indices, dim=1)                            
                values = torch.stack(values, dim=1)                            


                x0_candidate = x0.clone()
                x0_candidate[transfer_index_t_s] = indices.flatten().to(torch.long).to(self.device)
                
                candidate_prob = cur_beam_values[:, i].clone()
                candidate_prob = candidate_prob * values.prod(dim=1)

                new_seq = cur_seq.clone()
                new_seq[cur_mask] = x0_candidate
                
                expanded_candidates[:, i * num_combinations + combo_idx, :] = new_seq
                expanded_values[:, i * num_combinations + combo_idx] = candidate_prob
        
        return expanded_candidates, expanded_values


    def beam_search_generation(
        self,
        model_input: torch.Tensor,
        label_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs beam search to generate candidate sequences from masked positions.
        
        Args:
            probabilities: Model output probabilities with shape (bs, seq_len, vocab_size)
            modified_masked_input: Input tensor with masked tokens
            num_candidates: Number of top candidates to keep during beam search
            
        Returns:
            Tuple containing:
                - generated_ids: Generated token sequences (bs, num_candidates, num_hierarchies)
                - beam_values: Final probability scores for each candidate (bs, num_candidates)
        """
        num_candidates = self.diffusion_config['num_candidates']
        steps = self.diffusion_config['num_steps']
        
        timesteps = torch.linspace(1, 1e-5, steps + 1, device=self.device)
        bs, seq_len = model_input.transformed_sequences['sequence_data'].shape
        generated_ids = torch.zeros(
            (bs, num_candidates, self.num_hierarchies), 
            dtype=torch.long, device=self.device)
        probs = torch.ones((bs, num_candidates), device=self.device)

        label_locations = label_data.label_location['sequence_data']
        batch_indices, seq_indices = label_locations[:, 0], label_locations[:, 1]
        transformed_masked_input, transformed_labels = self.transform_input_and_labels( 
            model_input.transformed_sequences['sequence_data'], label_data.labels['sequence_data'], label_locations 
            )

        cur_beam_candidate = torch.zeros((bs, 1, seq_len), dtype=torch.long, device=self.device)
        cur_beam_values = torch.ones((bs, 1), dtype=torch.float32, device=self.device)
        cur_beam_candidate[:, 0, :] = transformed_masked_input

        num_mask_per_row = self.num_hierarchies
        if 'unmasking_nums' in self.diffusion_config and self.diffusion_config['unmasking_nums'] is not None:
            steps = len(self.diffusion_config['unmasking_nums'])

        item_id_seqs = self.get_item_id_seqs(model_input)

        for j in range(steps):
            if 'unmasking_nums' in self.diffusion_config and self.diffusion_config['unmasking_nums'] is not None:
                tokens_to_unmask = self.diffusion_config['unmasking_nums'][j]
            else:
                tokens_to_unmask = self.num_hierarchies // steps
            
            expanded_candidates, expanded_values = self.get_expanded_candidates(
                cur_beam_candidate, cur_beam_values, num_candidates, item_id_seqs,
                num_mask_per_row, tokens_to_unmask, transformed_labels, label_locations
            )
            num_mask_per_row -= tokens_to_unmask
                                                       
            topk_values, topk_indices = torch.topk(expanded_values, num_candidates, dim=-1)
            new_beam_candidate = torch.zeros((bs, num_candidates, seq_len), dtype=torch.long, device=self.device)
            new_beam_values = torch.zeros((bs, num_candidates), dtype=torch.float32, device=self.device)

            for i in range(num_candidates):
                new_beam_candidate[:, i, :] = expanded_candidates[torch.arange(bs), topk_indices[:, i], :]
                new_beam_values[:, i] = expanded_values[torch.arange(bs), topk_indices[:, i]]

            cur_beam_candidate = new_beam_candidate
            cur_beam_values = new_beam_values

        
        for i in range(num_candidates):
            cur_seq = cur_beam_candidate[:, i, :]
            generated_ids_flatten = cur_seq[batch_indices, seq_indices]
            generated_ids[:, i, :] = generated_ids_flatten.reshape(-1, self.num_hierarchies)
        
        return generated_ids, new_beam_values


    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        self.eval_step(batch, self.val_loss)
        self.log("val_loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set."""
        self.eval_step(batch, self.test_loss)
        self.log("test_loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)


    def mask_input_and_labels(self, model_input, label_data):
        """
        Mask random fractions of positions in each row of input sequences.
        
        Args:
            model_input: Model input data containing sequences
            label_data: Label data containing labels and locations
            
        Returns:
            model_input, label_data: Modified input and label data with masking applied
        """
        input_seq = model_input.transformed_sequences['sequence_data']
        labels_seq = label_data.labels['sequence_data']
        label_locations_seq = label_data.label_location['sequence_data']
        
        batch_size, seq_len = input_seq.shape
        device = input_seq.device
        unpadded_positions = (input_seq != self.padding_token_id)
        
                                                                          
        cur_noise_schedule = self.diffusion_config['noise_schedule']
        
        if self.trainer.global_step % self.dense_retrieval['update_frequency'] == 0:
            cur_noise_schedule = self.dense_retrieval['noise']
        
        if cur_noise_schedule == 'uniform':
            fractions = torch.rand(batch_size, device=device) * self.diffusion_config['max_mask_fraction']
        elif cur_noise_schedule == 'mask-multiple-random-item':
            max_items = seq_len // self.num_hierarchies
            random_vals = torch.rand(batch_size, max_items, device=device)
            fractions = torch.rand(batch_size, device=device)
            mask_item_indices = random_vals < fractions.unsqueeze(1)
            mask_positions_ind = torch.where(mask_item_indices)[1][:, None] * self.num_hierarchies + torch.arange(self.num_hierarchies, device=device)[None, :]

            mask_positions = torch.zeros_like(input_seq, dtype=torch.bool)
            mask_positions[ torch.where(mask_item_indices)[0][:, None], mask_positions_ind ] = True
            unpadded_mask_positions = mask_positions & unpadded_positions

        if cur_noise_schedule == 'uniform':
            random_vals = torch.rand(batch_size, seq_len, device=device)
            mask_positions = random_vals < fractions.unsqueeze(1)
            unpadded_mask_positions = mask_positions & unpadded_positions
        
                             
        masked_input = input_seq.clone()
        masked_input[unpadded_mask_positions] = self.masking_token_id
        
        labels = input_seq[unpadded_mask_positions]
        batch_indices, col_indices = torch.where(unpadded_mask_positions)
        label_locations = torch.vstack((batch_indices, col_indices)).t()
        
                                    
        model_input.transformed_sequences['sequence_data'] = masked_input
        label_data.labels['sequence_data'] = labels
        label_data.label_location['sequence_data'] = label_locations
        
        return model_input, label_data, fractions

    def on_load_checkpoint(self, checkpoint):
        """Override checkpoint loading to reset learning rate and scheduler state."""                                  

        checkpoint['lr_schedulers'][0]['min_ratio'] = self.scheduler.keywords['min_ratio']
        checkpoint['lr_schedulers'][0]['base_lrs'][0] = self.optimizer.keywords['lr']
        checkpoint['lr_schedulers'][0]['scheduler_steps'] = self.scheduler.keywords['scheduler_steps']
        
        super().on_load_checkpoint(checkpoint)

