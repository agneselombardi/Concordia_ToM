from collections.abc import Collection, Sequence
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import text
from typing_extensions import override
import vertexai
from vertexai.preview import language_models as vertex_models
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MAX_MULTIPLE_CHOICE_ATTEMPTS = 3

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")
CUDA_LAUNCH_BLOCKING=1

class HfLanguageModel(language_model.LanguageModel):
    def __init__(
       self,
       model_name: str,
       cache_dir: str,
       measurements: measurements_lib.Measurements | None = None,
       channel: str = language_model.DEFAULT_STATS_CHANNEL,
      ):
      self._model = AutoModelForCausalLM.from_pretrained(model_name, **cache_dir).to(DEVICE)
      self._tokenizer =  AutoTokenizer.from_pretrained(model_name, **cache_dir)
      self._measurements = measurements
      self._channel = channel


    @override
    def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      max_characters: int = language_model.DEFAULT_MAX_CHARACTERS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
    ) -> str:

      prompt = self._tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
      output = self._model.generate(
        **prompt,
        temperature=temperature,
        max_length=max_tokens,
        seed=seed,
      )
      print(output.shape)
      
      if isinstance(output, list) and isinstance(output[0], dict) and 'sequences' in output[0]:
        generated_answer = self._tokenizer.decode(output[0]['sequences'][0])
      elif isinstance(output, dict) and 'sequences' in output:
        generated_answer = self._tokenizer.decode(output['sequences'][0])
      elif 'sequences' in output:
        generated_answer = self._tokenizer.decode(output['sequences'][0])
      else:
        generated_answer = self._tokenizer.decode(output[0])

      if self._measurements is not None:
       self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(generated_answer)},
      )
      return generated_answer
    
    @override
    def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
      max_characters = max([len(response) for response in responses])

      attempts = 1
      for _ in range(MAX_MULTIPLE_CHOICE_ATTEMPTS):
        sample = self.sample_text(
          prompt,
          max_tokens=1,
          max_characters=max_characters,
          temperature=0.0,
          seed=seed,
      )
        try:
         idx = responses.index(sample)
        except ValueError:
         attempts += 1
         continue
        else:
         if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel,
              {'choices_calls': attempts})
         debug = {}
         return idx, responses[idx], debug

      raise language_model.InvalidResponseError(
        'Too many multiple choice attempts.'
      )






