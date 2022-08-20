import argparse
import logging
import clip
from model.CapGenerator import CLIPTextGenerator
import torch
import os
# from data_loader import VideosDataset, ImagesDataset, ImagesPairsDataset
from datetime import datetime
import shutil
import json
import sys
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomized_prompt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--db_filter_path", type=str, default=None, help="file to filter db items, e.g karpathy split")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=20)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--token_wise", action="store_true", help="Should we step the optimization at each token gen")
    parser.add_argument("--num_dummy_tokens", type=int, default=5)
    parser.add_argument("--sentence_iterations", type=int, default=30)
    parser.add_argument("--sampling_top_k", type=int, default=3)
    parser.add_argument("--db_start_idx", type=int, default=0)
    parser.add_argument("--db_num_images", type=int, default=0)
    parser.add_argument("--clip_loss_temperature", type=float, default=1.0)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.8)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.006)
    parser.add_argument("--scheduler_type", type=CLIPTextGenerator.SchedType, default='cosine')
    parser.add_argument("--weight_decay_scale", type=float, default=0.3)
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help='How much much to deter deter repeats')
    parser.add_argument("--entity_penalty", type=float, default=2, help='How much to deter CapsLock in middle of sent')
    parser.add_argument("--ending_bonus", type=float, default=2, help='How much to help the sentence to end')
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--pairs_path", type=str, default="")

    parser.add_argument('--data_path', type=str, default='/home/work/Datasets/MSR-VTT/examples/video7157.mp4')
    parser.add_argument('--run_type',
                        default='caption_images',
                        nargs='?',
                        choices=['caption_images', 'caption_videos'])
    return parser

def filter_video(image_fts, similiarities):
    THRESHOLD = 0.9
    groups = []
    curr_group = []
    for i in range(similiarities.size(0)):
        if len(curr_group) == 0:
            curr_group.append(i)

        if i + 1 == similiarities.size(0):
            if len(curr_group) >= 1:
                groups.append(curr_group)
            break

        if similiarities[curr_group[0]][i + 1] > THRESHOLD:
            curr_group.append(i + 1)
        else:
            if len(curr_group) >= 1:
                groups.append(curr_group)
            curr_group = []

    result_features = []
    selected_indices = []
    if len(groups) >= 1:
        for i, group in enumerate(groups):
            result_features.append(image_fts[group[0]])
            selected_indices.append(group[0])

    return torch.stack(result_features), selected_indices

def get_clip_video_frames(video_path, clip_preprocess):
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    sample_time = FPS // 3
    imgs = []

    i = 0
    while (cap.isOpened()):
        ret, cv2_im = cap.read()

        if ret and i % sample_time == 0:
            converted = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(converted)
            imgs.append(pil_im)
        elif not ret:
            break

        i += 1

    cap.release()

    images = torch.cat([clip_preprocess(x).unsqueeze(0) for x in imgs])

    return images

def get_clip_image(image_path, clip_preprocess):
    images = torch.cat([clip_preprocess(Image.open(image_path)).unsqueeze(0)])

    return images

def run_video(args, video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_generator = CLIPTextGenerator(**vars(args))

    video_frames = get_clip_video_frames(video_path, text_generator.clip_preprocess).to(device)

    with torch.no_grad():
        frames_fts = text_generator.clip.encode_image(video_frames).detach()
        frames_fts = torch.nn.functional.normalize(frames_fts, dim=-1).detach()

        similiarities = frames_fts @ frames_fts.T
        image_fts, selected_frames_indices = filter_video(frames_fts, similiarities)

    clip_sorted_captions, mixed_sorted_captions, decoded_options, beam_caps = text_generator.generate(image_fts)

    print(clip_sorted_captions)

    return clip_sorted_captions[0]

def run_image(args, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_generator = CLIPTextGenerator(**vars(args))

    image = get_clip_image(image_path, text_generator.clip_preprocess).to(device)

    with torch.no_grad():
        image_fts = text_generator.clip.encode_image(image).detach()
        image_fts = torch.nn.functional.normalize(image_fts, dim=-1).detach()

    clip_sorted_captions, mixed_sorted_captions, decoded_options, beam_caps = text_generator.generate(image_fts)

    print(clip_sorted_captions)

    return clip_sorted_captions[0]

if __name__ == "__main__":
    torch.set_num_threads(3)
    cli_args = get_parser().parse_args()

    if cli_args.run_type == 'caption_videos':
        run_video(cli_args, cli_args.data_path)
    elif cli_args.run_type == 'caption_images':
        run_image(cli_args, cli_args.data_path)