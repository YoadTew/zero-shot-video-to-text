from typing import List

import numpy as np
from torch import nn
from itertools import chain, groupby
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt_neo import GPTNeoForCausalLM
from transformers.generation_logits_process import NoRepeatNGramLogitsProcessor, RepetitionPenaltyLogitsProcessor, \
    NoBadWordsLogitsProcessor, MinLengthLogitsProcessor
import torch
import clip
from PIL import Image
from datetime import datetime
from enum import Enum
import sys
import random
import logging


def log_info(text, verbose=True):
    if verbose:
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string} | {text}')
        sys.stdout.flush()


def add_context(x, y):
    return (x[0] + y[0], x[1] + y[1])


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


class CLIPTextGenerator:

    class SchedType(Enum):
        Nothing = "nothing"
        Cosine = "cosine"
        Exponential = "expo"

    def __init__(self,
                 seed=0,
                 lm_model='gpt-2',
                 clip_checkpoints='./clip_checkpoints',
                 target_seq_length=15,
                 randomized_prompt=False,
                 token_wise=False,
                 num_dummy_tokens=5,
                 sentence_iterations=64,
                 clip_loss_temperature=1.0,
                 sampling_top_k=3,
                 clip_scale=1.,
                 ce_scale=0.2,
                 learning_rate=0.01,
                 scheduler_type: SchedType = SchedType.Nothing,
                 weight_decay_scale=0.3,
                 repetition_penalty=2.0,
                 entity_penalty=2,
                 ending_bonus=1,
                 end_token='.',
                 beam_size=5,
                 **kwargs):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set random seed
        self.seed = seed
        self.reset_random_seed()

        # Turn off annoying logs
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        # Initialize Language model
        self.context_prefix = ''
        if lm_model == 'gpt-neo':
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
            self.lm_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M', output_hidden_states=True)
        elif lm_model == 'gpt-2':
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.lm_model = GPT2LMHeadModel.from_pretrained('gpt2-medium', output_hidden_states=True)
            self.context_prefix = self.lm_tokenizer.bos_token
        self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.lm_model.to(self.device)
        self.lm_model.eval()
        # Freeze LM weights
        for param in self.lm_model.parameters():
            param.requires_grad = False

        dummy_tokens = self.lm_tokenizer.encode([self.lm_tokenizer.bos_token for _ in range(num_dummy_tokens)])
        dummy_tokens = torch.tensor(dummy_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            self.dummy_context_init = self.lm_model(dummy_tokens)["past_key_values"]
        # initialize the dummy context as parameters for optimization (can be fed into lm_model)
        self.dummy_context_offset = [(torch.nn.Parameter(torch.zeros(k.shape).type_as(k)),
                                      torch.nn.Parameter(torch.zeros(v.shape).type_as(v)))
                                     for k, v in self.dummy_context_init]
        self.dummy_optimizer = torch.optim.AdamW([x for pair in self.dummy_context_offset for x in pair],
                                                 lr=learning_rate, weight_decay=learning_rate * weight_decay_scale)
        self.dummy_optimizer_init = self.dummy_optimizer.state_dict()

        # Below options must be 2 tokens only to avoid alignment issues
        if randomized_prompt:
            print('using random prompts...')
            self.context_options = ['Image of', 'Picture of', 'Photo of', 'Video of',
                                    'Image shows', 'Picture shows', 'Photo shows', 'Video shows',
                                    'Image showing', 'Picture showing', 'Photo showing', 'Video showing']
            prompt_len = 2
        else:
            self.context_options = ['']
            prompt_len = 0
        test_prefixes = [self.context_prefix + choice for choice in self.context_options]
        test_generated_tokens = self.lm_tokenizer.batch_encode_plus(
            test_prefixes, return_tensors='pt', return_attention_mask=False, padding=True)["input_ids"].to(self.device)
        prefix_len = prompt_len + 1
        assert test_generated_tokens.shape[1] == prefix_len, \
            f"All appended context options must be of exactly length {prefix_len}!"

        self.end_token = self.lm_tokenizer.encode(end_token)[0]

        # Special char signifying a space for GPT-2, decoded into space by tokenizer
        spacer = 'Ġ'
        # Avoid tokens that are not completely whitelisted
        lower_chars = [chr(char_ord) for char_ord in range(ord('a'), ord('z') + 1)]
        upper_chars = [chr(char_ord) for char_ord in range(ord('A'), ord('Z') + 1)]
        special_chars = [' ', ',', '\'', spacer]
        # The ending token is legit, if forbidden, re-allow it.
        forbidden_tokens = [key for key, value in self.lm_tokenizer.decoder.items()
                            if not set(value).issubset(set(lower_chars + upper_chars + special_chars))
                            or key == self.end_token]
        # Try to have a certain styling, specifically to avoid too much Entities mid sentence
        unwanted_first_tokens = [key for key, value in self.lm_tokenizer.decoder.items()
                                 if not set(value).issubset(set(lower_chars + upper_chars))]
        unwanted_later_tokens = [key for key, value in self.lm_tokenizer.decoder.items()
                                 if not set(value).issubset(set(lower_chars + special_chars))]
        self.first_token_offset = torch.zeros((self.lm_tokenizer.vocab_size,), device=self.device)
        self.first_token_offset[unwanted_first_tokens] = -entity_penalty
        self.other_token_offset = torch.zeros((self.lm_tokenizer.vocab_size,), device=self.device)
        self.other_token_offset[unwanted_later_tokens] = -entity_penalty

        # A logit updater to avoid the blacklist
        self.prevent_forbidden_tokens = NoBadWordsLogitsProcessor([[token] for token in forbidden_tokens], self.end_token)

        # Map from each token to its equivalent tokens, to better eliminate duplications, using a tensor for speed
        self.token_to_similar_indices = torch.zeros((self.lm_tokenizer.vocab_size, 8)).to(torch.long).to(self.device)
        reduced_token_form = {
            key: value.replace(spacer, '').upper() for key, value in self.lm_tokenizer.decoder.items()
            if key not in forbidden_tokens}
        for reduced_form, token_iter in groupby(sorted(reduced_token_form, key=reduced_token_form.get),
                                                key=reduced_token_form.get):
            tokens = set(token_iter)
            for token in tokens:
                for index, similar in enumerate(tokens):
                    self.token_to_similar_indices[token][index] = similar
        # Prevent small repetitions of similar tokens
        self.deter_small_repeat = RepetitionPenaltyLogitsProcessor(repetition_penalty)

        # Prevent larger repetition of the exact same tokens. Set to 3 to support 2 token word repeat if necessary.
        self.prevent_large_repeat = NoRepeatNGramLogitsProcessor(3)

        # Prevent early completion of caption
        self.prevent_early_finish = MinLengthLogitsProcessor(target_seq_length // 2, self.end_token)

        # Initialize CLIP
        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=self.device,
                                                    download_root=clip_checkpoints, jit=False)
        self.clip = self.clip.eval()
        # Freeze CLIP weights
        for param in self.clip.parameters():
            param.requires_grad = False

        self.logit_scale = self.clip.logit_scale.exp()

        # Init attributes
        self.ending_bonus = ending_bonus
        self.token_wise = token_wise
        self.target_seq_length = target_seq_length
        self.sampling_top_k = sampling_top_k
        self.sentence_iterations = sentence_iterations
        self.clip_loss_temperature = clip_loss_temperature
        self.clip_scale = clip_scale
        self.ce_scale = ce_scale
        self.scheduler_type = scheduler_type
        self.beam_size = beam_size

    def reset_random_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    @property
    def dummy_context(self):
        return [(k + k_delta, v + v_delta) for (k, v), (k_delta, v_delta) in
                zip(self.dummy_context_init, self.dummy_context_offset)]

    def dummy_context_reset(self):
        # Restart the cross sentence context before starting on a new image
        with torch.no_grad():
            for k_delta, v_delta in self.dummy_context_offset:
                k_delta.zero_()
                v_delta.zero_()
        # Initialize the dummy, the fair thing to do
        self.dummy_optimizer.load_state_dict(self.dummy_optimizer_init)

    def get_img_feature(self, img_path, weights):
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]

        with torch.no_grad():
            image_fts = [self.clip.encode_image(x) for x in clip_imgs]

            if weights is not None:
                image_features = sum([x * weights[i] for i, x in enumerate(image_fts)])
            else:
                image_features = sum(image_fts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.detach()

    def get_video_feature(self, imgs_path):
        imgs = [Image.open(x) for x in imgs_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]

        with torch.no_grad():
            image_fts = torch.cat([self.clip.encode_image(x) for x in clip_imgs])

            image_features = nn.functional.normalize(image_fts, dim=-1)

        return image_features.detach()

    def get_txt_features(self, text):
        clip_texts = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            text_features = self.clip.encode_text(clip_texts)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach()

    def get_combined_feature(self, img_path, texts, weights_i, weights_t):
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]
        clip_texts = [clip.tokenize(x).to(self.device) for x in texts]

        with torch.no_grad():
            image_fts = [self.clip.encode_image(x) for x in clip_imgs]
            text_fts = [self.clip.encode_text(x) for x in clip_texts]

            features = sum([x * weights_i[i] for i, x in enumerate(image_fts)])
            if weights_t is not None:
                features += sum([x * weights_t[i] for i, x in enumerate(text_fts)])

            features = features / features.norm(dim=-1, keepdim=True)
            return features.detach()

    def generate(self, image_features: torch.Tensor):
        self.image_features = image_features
        self.dummy_context_reset()

        # set random seed for better reproducibility
        self.reset_random_seed()

        if self.scheduler_type == self.SchedType.Cosine:
            sentence_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.dummy_optimizer, eta_min=1e-4, T_max=self.sentence_iterations)
        elif self.scheduler_type == self.SchedType.Exponential:
            sentence_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.dummy_optimizer, gamma=0.9)
        else:
            sentence_scheduler = None

        decoded_options = []
        avg_perplexities = []
        avg_frame_similarities = []
        for gd_iter in range(self.sentence_iterations):
            context_tokens = self.lm_tokenizer.encode(self.context_prefix + random.choice(self.context_options))
            generated_tokens = torch.tensor(context_tokens, device=self.device, dtype=torch.long).unsqueeze(0)

            clip_losses = []
            fluency_losses = []
            total_perplexity = 0
            for i in range(self.target_seq_length):
                unshifted_outputs = self.lm_model(generated_tokens)
                unshifted_logits = unshifted_outputs["logits"][:, -1, :]
                unshifted_probs = nn.functional.softmax(unshifted_logits, dim=-1)

                shifted_outputs = self.lm_model(generated_tokens, past_key_values=self.dummy_context)
                logits = shifted_outputs["logits"][:, -1, :]
                probs = nn.functional.softmax(logits, dim=-1) + 2e-45

                clip_loss, _ = self.clip_loss(probs, generated_tokens, 512)
                clip_losses.append(clip_loss.item())
                fluency_loss = -(unshifted_probs * probs.log()).sum()
                fluency_losses.append(fluency_loss.item())

                # Take gradients
                loss = (self.clip_scale * clip_loss) + (self.ce_scale * fluency_loss)
                loss.backward()

                if self.token_wise:
                    # step the dummy context every token
                    self.dummy_optimizer.step()
                    self.dummy_optimizer.zero_grad()

                # construct the next generation of sequences
                next_token = self.sample_next_token(logits, i, generated_tokens)
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                total_perplexity += torch.gather(unshifted_probs.log(), 1, next_token).sum(dim=0)

                if next_token == self.end_token:
                    break

            if not self.token_wise:
                # step the dummy context every sentence
                self.dummy_optimizer.step()
                self.dummy_optimizer.zero_grad()

            if self.scheduler_type != self.SchedType.Nothing:
                # Update the scheduler to reduce individual steps impact
                sentence_scheduler.step()

            # Compute, log & record training intermediate results
            decoded_text = self.lm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            avg_perplexity = total_perplexity / i
            avg_frame_sim = self.text_to_video_similarity(decoded_text)
            print(f"(Index: {gd_iter}, Len: {i + 1}, "
                         f"Avg clip: {avg_frame_sim.mean().item():.4f}, "
                         f"Avg ppl: {avg_perplexity.mean().item():.4f}): "
                         f"\"{decoded_text}\"")
            decoded_options.extend(decoded_text)
            avg_perplexities.append(avg_perplexity)
            avg_frame_similarities.append(avg_frame_sim.mean(-1))

            assert len(clip_losses) == len(fluency_losses), "expect balanced amount of losses per type"
            logging.debug(f'Unscaled CLIP loss: {sum(clip_losses)}, Unscaled Fluency loss: {sum(fluency_losses)}')

        avg_frame_similarities = torch.cat(avg_frame_similarities)
        caption_to_all_frame_prob = nn.functional.softmax(avg_frame_similarities * self.clip_scale)
        caption_to_language_prob = nn.functional.softmax(torch.cat(avg_perplexities, dim=0) * self.ce_scale, 0)
        mixed_score = caption_to_all_frame_prob * caption_to_language_prob

        clip_ordered_caption_prob, clip_caption_ordering = avg_frame_similarities.sort(descending=True)
        mixed_ordered_caption_prob, mixed_caption_ordering = mixed_score.sort(descending=True)
        logging.debug(f"The 'best' clip score & sorted indices: "
                      f"{list(zip(clip_ordered_caption_prob.tolist(), clip_caption_ordering.tolist()))}")
        logging.debug(f"The 'best' mixed score & sorted indices: "
                      f"{list(zip(mixed_ordered_caption_prob.tolist(), mixed_caption_ordering.tolist()))}")
        clip_sorted_captions = [decoded_options[index] for index in clip_caption_ordering]
        mixed_sorted_captions = [decoded_options[index] for index in mixed_caption_ordering]
        beam_tokens, beam_caps = self.decode_beam_search(self.dummy_context, self.beam_size)

        return clip_sorted_captions, mixed_sorted_captions, decoded_options, beam_caps

    def decode_beam_search(self, shifted_context, beam_size, prefix_text='Image of a'):
        context_tokens = self.lm_tokenizer.encode(self.context_prefix + prefix_text)
        generated_tokens = torch.tensor(context_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
        gen_tokens = None
        scores = None
        seq_lengths = torch.ones(beam_size, device=self.device)
        is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)

        for i in range(self.target_seq_length):
            with torch.no_grad():
                shifted_outputs = self.lm_model(generated_tokens, past_key_values=shifted_context)
            logits = shifted_outputs["logits"][:, -1, :]
            logits = self.update_special_tokens_logits(generated_tokens, i, logits)
            logits = torch.log_softmax(logits, dim=-1)

            if scores is None:
                for lid in range(len(shifted_context)):
                    shifted_context[lid] = (shifted_context[lid][0].repeat(beam_size, 1, 1, 1),
                                            shifted_context[lid][1].repeat(beam_size, 1, 1, 1))
                scores, next_tokens = logits.topk(beam_size, -1)
                generated_tokens = generated_tokens.repeat(beam_size, 1)
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                gen_tokens = next_tokens
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                gen_tokens = gen_tokens[next_tokens_source]
                gen_tokens = torch.cat((gen_tokens, next_tokens), dim=-1)
                generated_tokens = generated_tokens[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            generated_tokens = torch.cat((generated_tokens, next_tokens), dim=1)
            is_stopped = is_stopped + next_tokens.eq(self.end_token).squeeze()
            if is_stopped.all():
                break

        scores = scores / seq_lengths
        output_list = gen_tokens.cpu().numpy()
        output_texts = [
            self.lm_tokenizer.decode(output[: int(length)])
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]

        # for x in output_texts:
        #     print('@@@@@'*5)
        #     print(x)

        return generated_tokens, output_texts

    def sample_next_token(self, logits, i, generated_tokens):
        fixed_logits = self.update_special_tokens_logits(generated_tokens, i, logits.detach())
        option_logits, next_token_options = fixed_logits.topk(self.sampling_top_k)
        option_probs = nn.functional.softmax(option_logits, dim=-1)
        next_token_option_index = torch.multinomial(option_probs, num_samples=1)[0]
        return next_token_options[:, next_token_option_index]

    def update_special_tokens_logits(self, context_tokens, i, logits):
        # Deter tokens that appeared in similar form in recent history (single repeats allowed for remote sentences)
        history_len = 16
        history_tokens = context_tokens[:, -history_len:]
        history_similar_tokens = self.token_to_similar_indices[history_tokens].view((-1, history_tokens.shape[1] * 8))
        logits = self.deter_small_repeat(history_similar_tokens, logits)

        # Prevent exact duplicate n-grams from anywhere in context(not including similarities)
        logits = self.prevent_large_repeat(context_tokens, logits)

        # Prevent premature ending
        logits = self.prevent_early_finish(context_tokens, logits)

        # Prevent generation of tokens from the forbidden token list
        logits = self.prevent_forbidden_tokens(context_tokens, logits)

        # Reduce probability of unwanted styled words (entities, etc...)
        logits += (self.first_token_offset if i == 0 else self.other_token_offset).unsqueeze(0)

        # Give a tiny constant nudge towards ending, since we tend to end up with too long sentences
        logits[:, self.end_token] += self.ending_bonus

        # Reduce all prob above the ending prob, except those that are still left to consider
        # Like we force prevent short sentences, this promotes ending sentences
        if i >= self.target_seq_length - self.prevent_early_finish.min_length:
            ending_threshold = logits[:, self.end_token].clone()
            # disqualify all logit below the set of active considerations (and reserve a spot for the ending option)
            top_threshold_values, top_threshold_indices = logits.topk(self.sampling_top_k, 1)
            logits[logits < top_threshold_values[:, -1:]] = -np.inf
            logits[:, self.end_token] = ending_threshold
            # Remove the last candidate if ending isn't yet considered
            required_bump = top_threshold_values[:, -1] > ending_threshold
            logits[required_bump, top_threshold_indices[required_bump, -1]] = -np.inf

        # Finally, if somehow a NaN was introduced, set it -inf to remove its impact
        logits[logits != logits] = -np.inf

        return logits

    def text_to_video_similarity(self, text_cont: List[str]):
        with torch.no_grad():
            text_features = self.get_txt_features(text_cont)
        return self.logit_scale * (text_features @ self.image_features.T).type(torch.float32)

    def clip_loss(self, probs, context_tokens, samples: int):
        _, top_next_tokens = probs.topk(samples, -1)

        prefix_texts = self.lm_tokenizer.batch_decode(context_tokens, skip_special_tokens=True)

        prefix_losses = []
        for prefix_text, prefix_top_next_tokens, prefix_next_token_prob in zip(prefix_texts, top_next_tokens, probs):
            # Get the embedding of all likely continuations for the prefix
            top_prefix_continuation = []
            for next_token in prefix_top_next_tokens:
                top_prefix_continuation.append(prefix_text + self.lm_tokenizer.decode(next_token))

            cont_video_similarity = self.text_to_video_similarity(top_prefix_continuation)
            cont_video_prob = nn.functional.softmax(cont_video_similarity / self.clip_loss_temperature, dim=0)

            # Take the clip probabilities as ground truth prior over the continuations. Take cross entropy of predicted
            # next tokens based on this target prior.
            next_token_probs = prefix_next_token_prob[prefix_top_next_tokens]
            cont_video_loss = -cont_video_prob * next_token_probs.log().unsqueeze(-1)
            prefix_losses.append(cont_video_loss.sum(0).mean())

        return torch.sum(torch.stack(prefix_losses)), prefix_losses
