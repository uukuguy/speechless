import numpy as np
from mlx import core as mx

def encode_and_pad(text_list, tokenizer):
    """
    :param text_list: A list of text strings to be tokenized and padded.
    :param tokenizer: A tokenizer object capable of encoding text and providing an end-of-sequence token ID.
    :return: A list of encoded and padded sequences, each ending with the tokenizer's end-of-sequence token.
    """
    encoded_batch = [tokenizer.encode(text) for text in text_list]
    for sequence in encoded_batch:
        if sequence[-1] != tokenizer.eos_token_id:
            sequence.append(tokenizer.eos_token_id)
    return encoded_batch

def calculate_max_length(lengths, pad_to, max_seq_length):
    """
    :param lengths: List of integer lengths representing the lengths of sequences in a batch.
    :param pad_to: Integer value to pad the sequences to the nearest multiple.
    :param max_seq_length: Maximum allowed sequence length.
    :return: The maximum length of the sequences after padding, constrained by max_seq_length.
    """
    max_length = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
    return min(max_length, max_seq_length)

def create_delineated_batches(input_texts, output_texts, tokenizer, max_seq_length=2048, pad_to = 8):
    """
    Takes input and output text and creates batches that preserves the distinction between
    input and output tokens by keeping track of the length of input tokens in order to be
    able to (for example) mask input tokens when calculating the loss during training.
    """
    batch_size = len(input_texts)

    if len(input_texts) != len(output_texts):
        raise ValueError("input_text and output_text must be the same size")

    input_batch = encode_and_pad(input_texts, tokenizer)
    output_batch = encode_and_pad(output_texts, tokenizer)

    input_lengths = list(map(len, input_batch))
    output_lengths = list(map(len, output_batch))

    full_token_sequence = [input_batch[idx] + output_batch[idx] for idx in range(batch_size)]
    full_sequence_lengths = list(map(len, full_token_sequence))

    if max(full_sequence_lengths) > max_seq_length:
        print(
            f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
            f"The longest sentence {max(full_sequence_lengths)} will be truncated to {max_seq_length}. "
            "Consider pre-splitting your data to save memory."
        )
    max_length_in_batch = calculate_max_length(full_sequence_lengths, pad_to, max_seq_length)

    #Start out with a batch matrix with zeros
    batch = np.zeros((batch_size, max_length_in_batch), np.int32)

    adjusted_lengths = []
    for j in range(batch_size):
        input_length = input_lengths[j]
        #Minimum between output length and remaining space after the input sequences
        min_output_length = min(output_lengths[j], max_length_in_batch - input_length)
        #Index of sequence end prior to start of padding (if any)
        end_idx = input_length + min_output_length
        adjusted_lengths.append(end_idx)
        #Fill out the batch matrix with the corresponding full token sequence (leaving any padding)
        batch[j, :end_idx] = full_token_sequence[j][:end_idx]

    input_lengths = mx.array(input_lengths)
    full_sequence_lengths = mx.array(adjusted_lengths)

    return mx.array(batch), input_lengths, full_sequence_lengths
