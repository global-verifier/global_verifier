from .base_explorer_model import BaseExplorerModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .model_util import extract_mistral_response
import torch

class MistralExplorerModel(BaseExplorerModel):
    def __init__(self, model_path: str, max_new_tokens: int = 128):
        # Some Mistral weights (model_type="mistral3") are not wired into AutoModelForCausalLM
        # in certain transformers versions, so we special-case them.
        cfg = AutoConfig.from_pretrained(model_path)

        if getattr(cfg, "model_type", None) == "mistral3":
            from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto",
            )
            # Mistral3 models typically use Tekken tokenizer (tekken.json). AutoTokenizer mapping
            # may not include Mistral3Config, so use the dedicated tokenizer wrapper.
            try:
                from transformers.tokenization_mistral_common import MistralCommonTokenizer
            except Exception as e:
                raise RuntimeError(
                    "Failed to import MistralCommonTokenizer for Mistral3. "
                    "Please install `mistral-common` in this conda env.\n"
                    f"Original error: {e}"
                ) from e
            self.tokenizer = MistralCommonTokenizer.from_pretrained(model_path)
        else:
            # Load the model with device_map="auto" to handle large models across GPUs
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.max_new_tokens = max_new_tokens

    def digest_goal(self, goal: str) -> None:
        pass
    
    def get_next_action(self, get_action_prompt: str) -> str:
        # Tokenize the prompt
        model_inputs = self.tokenizer(
            [get_action_prompt],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7, # Mistral models often prefer slightly lower temperature
                top_p=0.95,
            )

        # Some implementations return full sequences (prompt + continuation) while others may return
        # only the continuation. Only strip the prompt when it is an exact prefix of the output.
        input_ids = model_inputs.input_ids[0]
        output_ids = generated_ids[0]
        if output_ids.shape[0] >= input_ids.shape[0] and torch.equal(output_ids[: input_ids.shape[0]], input_ids):
            new_token_ids = output_ids[input_ids.shape[0] :]
        else:
            new_token_ids = output_ids

        # Decode the continuation (keeping special tokens to handle cleanup in utility function)
        response = self.tokenizer.decode(new_token_ids, skip_special_tokens=False)

        # Combine to form full text for consistency with extraction logic
        full_text = get_action_prompt + response
        action = extract_mistral_response(full_text)
        return action