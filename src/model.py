import numpy as np
import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_rank(x, indices, one_indexed=True):
    """
    Adapted from https://stephantul.github.io/python/pytorch/2020/09/18/fast_topk/
    """
    vals = x[range(len(x)), indices]
    rank = (x > vals[:, None]).long().sum(1)
    if one_indexed:
        rank += 1
    return rank

class LlamaWrapper():
    """Model class for Llama evaluated in our experiments."""
    def __init__(
        self, 
        model_name: str, 
        **load_kwargs
    ) -> None:
        # Store basic meta data about the model.
        self.model_name = model_name
        print(load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        if tokenizer.pad_token is None:
            print("No pad token found; setting pad token to eos token")
            tokenizer.pad_token = tokenizer.eos_token
        self.model = LanguageModel(model, tokenizer=tokenizer)

    def logprobs_and_logit_diffs_all_layers(
        self,
        text: str
    ) -> torch.Tensor:
        """
        ADAPTED FROM THE FOLLOWING:
        https://github.com/Butanium/llm-latent-language/blob/7519b4a3b528dacbc4c37aca0f74497bda18e57f/exp_tools.py#L51
        """
        self.model.eval()
        with self.model.trace(text) as tracer:
            # Useful if we want to do multiple sentences at a time
            # inds = torch.arange(len(inputs.input_ids))
            hiddens_l = [
                layer.output[0][0, :].unsqueeze(1)
                for layer in self.model.model.layers
            ]
            # ~~~~~~~~~~~~~~ Get "raw" logits and logprobs (normal forward pass)
            # SHAPE: n_tokens x n_layers x hidden_size
            hiddens = torch.cat(hiddens_l, dim=1).cpu().save()
            rms_out = self.model.model.norm(hiddens)
            # SHAPE: n_tokens x n_layers x vocab_size
            logits = self.model.lm_head(rms_out).cpu().save()
            logprobs = logits.log_softmax(dim=-1).cpu().save()

            # ~~~~~~~~~~~~~~ Get "delta" logits and logprobs
            # Get deltas between hidden states (i --> i+1)
            # SHAPE: n_tokens x (n_layers - 1) x hidden_size
            hidden_deltas = hiddens[:, 1:, :]  - hiddens[:, :-1, :]
            rms_out_deltas = self.model.model.norm(hidden_deltas)
            # SHAPE: n_tokens x (n_layers-1) x vocab_size
            logits_deltas = self.model.lm_head(rms_out_deltas).cpu().save()
            logprobs_deltas = logits_deltas.log_softmax(dim=-1).cpu().save()

        return logits, logprobs, logits_deltas, logprobs_deltas
        
    def rank_of_token_all_layers(
        self, 
        prefix: str,
        token_ids: list[int],
        one_indexed: bool = True
    ):
        """
        Get ranks of specified tokens at each layer.
        Returns a list of N lists, where N is the length of `token_ids`.
        Each nested list has one rank per layer.
        """
        # logprobs = self.logprobs_all_layers(prefix, return_logits=False)
        _, logprobs, _, _ = self.logprobs_and_logit_diffs_all_layers(prefix)

        # Just look at the predictions at the last position
        # (look at the logits at position i-1 for target token i).
        final_logprobs = logprobs[-1, :, :] # n_layers x vocab_size

        ranks = []
        for token_id in token_ids:
            rank_of_token = get_rank(
                final_logprobs, 
                token_id, 
                one_indexed=one_indexed
            ).tolist()
            ranks.append(rank_of_token)
        return ranks

    def conditional_score_all_layers(
        self, 
        prefix: str, 
        continuation: str, 
        sep: str = " ", 
        check_tokenization: bool = True
    ) -> dict:
        """
        Get scores of the continuation conditioned on the prefix at each layer.
        Scores are defined as a reduction of the raw token-level log probabilities.

        Args:
            prefix: str (what you want to condition on)
            continuation: str (what you want to compute probability of)
        """
        text = prefix + sep + continuation

        logits, logprobs, logits_deltas, _ = \
            self.logprobs_and_logit_diffs_all_layers(text)

        # logprobs, logits = self.logprobs_all_layers(text, return_logits=True)

        # Tokenize the continuation separately so we know which ones to keep.
        continuation_tokens = self.model.tokenizer(continuation)["input_ids"]

        # Remove BOS token if necessary, since it might have been
        # automatically prepended when tokenizing the continuation separately.
        if self.model.tokenizer.bos_token:
            continuation_tokens = [
                t for t in continuation_tokens 
                if t != self.model.tokenizer.bos_token_id
            ]

        # OPTIONAL: Double check that we are "reconstructing" tokens correctly.
        # Skipping this step might slightly speed up the evaluation process.
        if check_tokenization:
            # "Original" tokens.
            inputs = self.model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            text_tokens = inputs["input_ids"][0].tolist()

            # "New" tokens by combining prefix and continuation separately.
            prefix_tokens = self.model.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            new_text_tokens = prefix_tokens + continuation_tokens
            
            assert all(n == t for n, t in zip(new_text_tokens, text_tokens))

        # Only keep the logits corresponding to the continuation.
        # We look at the logits at position i-1 for target token i, so we need to
        # shift logits and logprobs to the left by 1.
        n_continuation_tokens = len(continuation_tokens)
        continuation_logprobs = logprobs[-n_continuation_tokens-1:-1, :, :]
        continuation_logits = logits[-n_continuation_tokens-1:-1, :, :]
        continuation_logits_deltas = logits_deltas[-n_continuation_tokens-1:-1, :, :]

        # Compute entropy at first token of continuation for each layer.
        # FINAL SHAPE: n_layers
        first_token_of_continuation_logprobs = continuation_logprobs[0, :, :]
        all_layer_entropy = (
            # Exponentiate logprobs to get probs
            -first_token_of_continuation_logprobs.exp() * \
            first_token_of_continuation_logprobs
        ).sum(dim=1).detach().numpy()

        # Get logprobs corresponding to each token in the continuation,
        # for each layer. FINAL SHAPE: n_layers x n_continuation_tokens
        n_layers = len(self.model.model.layers)
        all_layer_logprobs = np.array([
            [
                continuation_logprobs[i][layer_id][token_id].detach().cpu()
                for i, token_id in enumerate(continuation_tokens)
            ]
            for layer_id in range(n_layers)
        ])
        all_layer_logits = np.array([
            [
                continuation_logits[i][layer_id][token_id].detach().cpu()
                for i, token_id in enumerate(continuation_tokens)
            ]
            for layer_id in range(n_layers)
        ])
        # Do the same, but for the layer *deltas*. 
        # FINAL SHAPE: (n_layers-1) x n_continuation_tokens
        all_layer_logits_deltas = np.array([
            [
                continuation_logits_deltas[i][layer_id][token_id].detach().cpu()
                for i, token_id in enumerate(continuation_tokens)
            ]
            for layer_id in range(n_layers-1)
        ])

        # Perform reduction on token-level logprobs to get final scores.
        sum_logprobs = np.sum(all_layer_logprobs, axis=1)
        mean_logprobs = np.mean(all_layer_logprobs, axis=1)
        first_logprobs = all_layer_logprobs[:, 0]
        return {
            "entropy": all_layer_entropy,
            "sum": sum_logprobs, 
            "mean": mean_logprobs, 
            "first": first_logprobs, 
            "logits": all_layer_logits,
            "logits_deltas": all_layer_logits_deltas
        }