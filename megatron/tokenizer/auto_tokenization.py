from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

class TokenizerSpec(ABC):
    """
    Inherit this class to implement a new tokenizer.
    """

    @abstractmethod
    def text_to_tokens(self, text):
        pass

    @abstractmethod
    def tokens_to_text(self, tokens):
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens):
        pass

    @abstractmethod
    def ids_to_tokens(self, ids):
        pass

    @abstractmethod
    def text_to_ids(self, text):
        pass

    @abstractmethod
    def ids_to_text(self, ids):
        pass

    def add_special_tokens(self, special_tokens: List[str]):
        raise NotImplementedError("To be implemented")

    @property
    def name(self):
        return type(self).__name__

    @property
    def unique_identifiers(self):
        """Property required for use with megatron-core datasets."""
        return OrderedDict({"class": f"{type(self).__module__}.{type(self).__qualname__}"})

    @property
    def cls(self):
        """Property alias to match MegatronTokenizer; returns cls_id if available."""
        if hasattr(self, 'cls_id'):
            return self.cls_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'cls' or 'cls_id'")

    @property
    def sep(self):
        """Property alias to match MegatronTokenizer; returns sep_id if available."""
        if hasattr(self, 'sep_id'):
            return self.sep_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'sep' or 'sep_id'")

    @property
    def pad(self):
        """Property alias to match MegatronTokenizer; returns pad_id if available."""
        if hasattr(self, 'pad_id'):
            return self.pad_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'pad' or 'pad_id'")

    @property
    def eod(self):
        """Property alias to match MegatronTokenizer; returns eod_id if available."""
        if hasattr(self, 'eod_id'):
            return self.eod_id
        if hasattr(self, 'eos_id'):
            # Default to end-of-sentence id if end-of-document is not defined.
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eod', 'eod_id', 'eos', or 'eos_id'")

    @property
    def bos(self):
        """Property alias to match MegatronTokenizer; returns bos_id if available."""
        if hasattr(self, 'bos_id'):
            return self.bos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'bos' or 'bos_id'")

    @property
    def eos(self):
        """Property alias to match MegatronTokenizer; returns eos_id if available."""
        if hasattr(self, 'eos_id'):
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eos' or 'eos_id'")

    @property
    def mask(self):
        """Property alias to match MegatronTokenizer; returns mask_id if available."""
        if hasattr(self, 'mask_id'):
            return self.mask_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'mask' or 'mask_id'")

from transformers import AutoTokenizer as AUTOTOKENIZER
from collections import OrderedDict
from typing import Optional

class AutoTokenizer(TokenizerSpec):
    """
        Wrapper of HuggingFace AutoTokenizer https://huggingface.co/transformers/model_doc/auto.html#autotokenizer.

    """

    def __init__(
        self,
        pretrained_model_name: str,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        use_fast: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
    ):

        """
        Args:
            pretrained_model_name: corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument. 
                For more details please refer to https://huggingface.co/transformers/_modules/transformers/tokenization_auto.html#AutoTokenizer.from_pretrained. 
                The list of all supported models can be found here: ALL_PRETRAINED_CONFIG_ARCHIVE_MAP
            vocab_file: path to file with vocabulary which consists
                of characters separated by newlines.
            mask_token: mask token 
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
            use_fast: whether to use fast HuggingFace tokenizer
        """
        try:
            # this logic deals with different huggingface tokenizers having different positional args
            if vocab_file is None:
                self.tokenizer = AUTOTOKENIZER.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                )
            elif merges_file is None:
                self.tokenizer = AUTOTOKENIZER.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name,
                    vocab_file=vocab_file,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                )
            else:
                self.tokenizer = AUTOTOKENIZER.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name,
                    vocab_file=vocab_file,
                    merges_file=merges_file,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                )
        except Exception as e:
            raise ValueError(
                f'Unable to instantiate HuggingFace AUTOTOKENIZER for {pretrained_model_name}. Exception: {e}'
            )

        self.original_vocab_size = len(self.tokenizer)
        special_tokens_dict = {}

        # # setting special tokens, by default the default model's special tokens will be preserved
        # # unless passes new values to the special tokens
        if unk_token is not None:
            special_tokens_dict["unk_token"] = unk_token
        if mask_token is not None:
            special_tokens_dict["mask_token"] = mask_token
        if pad_token is not None:
            special_tokens_dict["pad_token"] = pad_token

        # if the model does not have eos_token but has sep_token,
        # set eos_token = sep_token, and vice versa
        if sep_token is not None:
            special_tokens_dict["sep_token"] = sep_token
        elif self.tokenizer.sep_token is None and self.tokenizer.eos_token:
            special_tokens_dict["sep_token"] = self.tokenizer.eos_token
        if eos_token is not None:
            special_tokens_dict["eos_token"] = eos_token
        elif self.tokenizer.eos_token is None and self.tokenizer.sep_token:
            special_tokens_dict["eos_token"] = self.tokenizer.sep_token
    
        # if the model does not have bos_token but has cls_token,
        # set bos_token = cls_token, and vice versa
        if bos_token is not None:
            special_tokens_dict["bos_token"] = bos_token
        elif self.tokenizer.bos_token is None and self.tokenizer.cls_token:
            special_tokens_dict["bos_token"] = self.tokenizer.cls_token
        if cls_token is not None:
            special_tokens_dict["cls_token"] = cls_token
        elif self.tokenizer.cls_token is None and self.tokenizer.bos_token:
            special_tokens_dict["cls_token"] = self.tokenizer.bos_token

        #print(special_tokens_dict)
        new_tokens_in_vocab = []
        for token in [mask_token, bos_token, eos_token, pad_token, sep_token, cls_token, unk_token]:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)

        if len(new_tokens_in_vocab) > 0:
            """
            Special tokens that were not previously included in the tokenizer's vocabulary file will be added to 
            the vocabulary and, as a result, the model should be resized, for example:
            
            # define your model
            pretrained_model_name = 'roberta-base'
            model = nemo_nlp.modules.get_lm_model(pretrained_model_name=pretrained_model_name)
            
            # define pretrained tokenizer
            tokenizer_default = nemo_nlp.modules.get_tokenizer(tokenizer_name=pretrained_model_name)
            
            special_tokens = {'bos_token': '<BOS>',
                              'cls_token': '<CSL>',
                              'additional_special_tokens': ['<MY_NER_TOKEN>', '<ANOTHER_TOKEN>']}
            tokenizer_default.add_special_tokens(special_tokens_dict=special_tokens)
            
            # resize your model so that the embeddings for newly added tokens are updated during training/finetuning
            model.resize_token_embeddings(tokenizer_default.vocab_size)
            
            See NLP_Tokenizers.ipynb for more details.
            """
            print(
                f'{new_tokens_in_vocab} \n will be added to the vocabulary.\n'
                f'Please resize your model accordingly, '
                f'see NLP_Tokenizers.ipynb for more details.'
            )
        self.add_special_tokens(special_tokens_dict)
        self.space_sensitive = self.text_to_tokens('x y') != self.text_to_tokens('x') + self.text_to_tokens('y')

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...). If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).

        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].
                Tokens are only added if they are not already in the vocabulary.

        Returns:
            Number of tokens added to the vocabulary.
        """
        num_tokens_added = self.tokenizer.add_special_tokens(special_tokens_dict)

        if num_tokens_added > 0:
            print(f'{num_tokens_added} special tokens added, resize your model accordingly.')
        for k in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, k, getattr(self.tokenizer, k, None))
        return num_tokens_added

    @property
    def additional_special_tokens_ids(self):
        """Returns a list of the additional special tokens (excluding bos, eos, pad, unk). Used to return sentinel tokens for e.g. T5."""
        return [self.token_to_id(token) for token in self.additional_special_tokens]

    def text_to_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens):
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def tokenize(self, text):
        return self.text_to_ids(text)

    def detokenize(self, ids):
        return self.ids_to_text(ids)

    def token_to_id(self, token):
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        tokens_clean = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
        text = self.tokens_to_text(tokens_clean)
        return text

    @property
    def vocab(self):
        id2vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2vocab[i] for i in range(len(id2vocab))]

    @property
    def pad_id(self):
        if getattr(self, 'pad_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'pad_token')])[0]

    @property
    def bos_id(self):
        if getattr(self, 'bos_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'bos_token')])[0]

    @property
    def eos_id(self):
        if getattr(self, 'eos_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def eod(self):
        """Returns EOS token id. Exact copy of the eos_id function. Required for megatron-core."""
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def sep_id(self):
        if getattr(self, 'sep_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'sep_token')])[0]

    @property
    def cls_id(self):
        if getattr(self, 'cls_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'cls_token')])[0]

    @property
    def unk_id(self):
        if getattr(self, 'unk_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'unk_token')])[0]

    @property
    def mask_id(self):
        if getattr(self, 'mask_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'mask_token')])[0]

    @property
    def name(self):
        return type(self.tokenizer).__name__

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self.tokenizer.save_vocabulary(save_directory=save_directory, filename_prefix=filename_prefix)