import argparse
import re
import os
import json
import glob
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

disable_torch_init()

PROMPT_SETS ={
    "detailed":[
        "Please imagine the video based on the sequence of frames, and provide a faithfully detailed description of this video in more than three sentences.",
        "You are given a sequence of equally spaced video frames. Based on these frames, imagine the full video and provide a detailed description of what is happening in more than three sentences.",
        "The following set contains equally spaced video frames. Imagine the video from which these frames were taken and describe it in detail in at least three sentences.",
        "Below are equally spaced frames from a video. Use these frames to visualize the entire video and provide a detailed description in more than three sentences.",
        "A sequence of equally spaced video frames is presented. Please imagine the full video and write a faithfully detailed description of the events in more than three sentences.",
        "The images provided include equally spaced frames from a video. Based on these frames, imagine the video and describe it comprehensively in at least three sentences.",
        "You are given equally spaced frames from a video. Use these frames to envision the entire video and provide a detailed description of the events in more than three sentences.",
        "The sequence includes equally spaced frames from a video. Imagine the full video based on these frames and provide a detailed description in more than three sentences.",
        "The provided images contain equally spaced frames from a video. Visualize the video from these frames and describe it in detail in more than three sentences.",
        "Here are equally spaced frames from a video. Based on these frames, imagine the video and provide a detailed, faithful description of it in more than three sentences.",
        "The set of images includes equally spaced video frames. Please imagine the video these frames come from and describe it comprehensively in at least three sentences.",
        "Describe the video based on these frames in a few sentences.",
        "Imagine the video from these frames and describe it in detail in a few sentences.",
        "Based on these frames, provide a narrative of the video in more than three sentences.",
        "Describe the events in the video shown by these frames in at least three sentences.",
        "Visualize the video from these frames and explain what is happening in more than three sentences.",
        "Describe the sequence of events in the video depicted by these frames in a detailed manner.",
        "Given these equally spaced frames, imagine the entire video and provide a detailed description of the events, including the setting, characters, and actions, in more than three sentences.",
        "Visualize the video based on these frames and write a comprehensive description of what happens, describing the beginning, middle, and end in at least three sentences.",
        "Using these frames as a reference, imagine the full video and provide a thorough description of the plot, including key details and actions, in more than three sentences.",
        "Based on the sequence of these frames, describe the entire video in detail, mentioning important aspects such as the context, movements, and transitions in more than three sentences.",
        "Imagine the video that corresponds to these frames and provide an elaborate description, covering the storyline, visual elements, and any notable features in at least three sentences.",
    ],
    "background":[
        "The images are given containing equally spaced video frames.Summary of the background. This should also include the objects, location, weather, and time.",
        "Describe the background, including objects, location, weather, and time.",
        "Summarize the background setting of the video based on these frames.",
        "What is the environment like in these frames?",
        "What background objects and settings are visible in these frames?",
        "Summarize the background of the video, including details about the location, objects, weather, and time.",
        "Describe the environment shown in these frames, covering objects, location, weather, and time.",
        "Provide a detailed background description based on these frames, mentioning objects, location, weather, and time.",
        "Explain the setting of the video, focusing on the background elements like objects, location, weather, and time.",
        "Describe the overall environment in these frames, including details about objects, location, weather, and time.",
        "Given these equally spaced frames, provide a comprehensive background description, covering the objects, location, weather, and time.",
        "Imagine the environment from these frames and write a detailed description of the background, including objects, location, weather, and time.",
        "Based on these frames, describe the setting in detail, mentioning the objects present, the specific location, the weather conditions, and the time of day.",
        "Provide an elaborate background description based on these frames, covering all aspects of the environment such as objects, location, weather, and time.",
        "Using these frames as a reference, give a thorough description of the background, including details about the objects, location, weather, and time.",
    ],
    "short":[
        "Write a one-sentence summary of the video.",
        "Summarize the video in one concise sentence.",
        "Provide a brief description of the video in one sentence.",
        "What is the video about? Summarize it in one sentence.",
        "Provide a one-sentence summary that captures the main subject and action in the video.",
        "Write a concise one-sentence description that encapsulates the essence of the video.",
        "Describe the main theme or action of the video in a single sentence.",
        "What is happening in the video? Provide a one-sentence summary.",
        "Given these frames, write a brief one-sentence summary that captures the essence of the video's visual and artistic style.",
        "Summarize the key visual and thematic elements of the video in one concise sentence.",
        "Provide a one-sentence description that highlights the main subject and action depicted in the video.",
        "In one sentence, describe the primary visual and artistic elements of the video.",
        "Write a concise one-sentence summary that encapsulates the main action and visual style of the video.",
        "Briefly one-sentence Summary of the visual, Photographic and artistic style.",
    ],
    "main_object":[
        "Description of the main subject actions or status sequence. This suggests including the main subjects (person, object, animal, or none) and their attributes, their action, their position, and movements during the video frames.",
        "Summarize the primary subject's attributes and actions.",
        "Describe the main subject, including their attributes and movements throughout the video.",
        "Provide a detailed description of the main object's actions and positions in these frames.",
        "Summarize the main subject's actions, attributes, and movements during the video.",
        "What are the main object's attributes and how do they move throughout the video?",
        "Given these equally spaced frames, provide a comprehensive description of the main subject, including their attributes, actions, positions, and movements.",
        "Describe the primary object or subject in the video, detailing their attributes, actions, positions, and movements in these frames.",
        "Based on these frames, provide a detailed description of the main subject, including their attributes, actions, positions, and how they navigate through the video.",
        "Using these frames, describe the main subject's attributes, actions, and movements, detailing their positions and how they interact with the environment.",
        "Provide an elaborate description of the main object in the video, covering their attributes, actions, positions, and movements as shown in these frames.",
    ],
    "camera":[
        "Summary of the view shot, camera movement and changes in shooting angles in the sequence of video frames.",
        "Describe the camera movements in these frames.",
        "What are the camera angles and movements throughout the video?",
        "Summarize the camera actions and perspectives.",
        "Describe any camera zooms, pans, or angle changes.",
        "What camera movements are present in these frames?",
        "Describe the camera's movements, including pans, zooms, and angle changes in these frames.",
        "Summarize the camera actions and changes in shooting angles during the video.",
        "Provide a detailed description of the camera's movements and perspectives.",
        "Describe the camera's actions and how it follows the main subject.",
        "What are the camera movements and angle shifts in these frames?",
        "Given these equally spaced frames, provide a comprehensive description of the camera's movements, including any pans, zooms, and changes in shooting angles.",
        "Describe the camera's movements and angles in detail, explaining how it follows the main subject and changes perspectives.",
        "Based on these frames, provide a detailed description of the camera's actions, including any pans, zooms, angle shifts, and how it captures the scene.",
        "Using these frames, describe the camera's movements, including its tracking of the main subject, changes in angles, and any zooms or pans.",
        "Provide an elaborate description of the camera movements, covering pans, zooms, and changes in shooting angles as shown in these frames.",
    ]
}


class VideoDataset(Dataset):
    def __init__(self, video_list_file, caption_folder):
        caption_files = list(glob.glob(os.path.join(caption_folder, '*.json')))
        processed_videos = {os.path.splitext(os.path.basename(f))[0]: 1 for f in caption_files}

        videos = []
        if not os.path.exists(caption_folder):
            os.makedirs(caption_folder)
        with open(video_list_file, 'r') as f:
            for line in f.readlines():
                v = line.strip()
                videos.append(v)

        self.videos = []
        for v in tqdm(videos):
            if os.path.splitext(os.path.basename(v))[0] not in processed_videos:
                self.videos.append(v)

        print(f"Total {len(videos)} videos, left {len(self.videos)} to process")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video_file = self.videos[index]
        return video_file
    
def collate_fn(samples):
    return {
        'videos': samples
    }

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def load_video(video_path, max_frames_num):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        frame_idx = np.linspace(0, total_frame_num - 2, max_frames_num, dtype=int)
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(img) for img in spare_frames]
    except Exception as e:
        print(f"Failed to load video {video_path} with error: {e}")
        return [Image.new("RGB", (448, 448), (0, 0, 0))] * max_frames_num

@torch.inference_mode()
def main(args):
    if args.pdb_debug:
        import pdb; pdb.set_trace()
        args.num_workers = 0
    
    if not args.pdb_debug and (torch.cuda.device_count() > 1):
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
        )
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    

    # ======================================================
    # 1. read video dataset
    # ======================================================
    dataset = VideoDataset(video_list_file=args.video_list_file, caption_folder=args.caption_folder)
    if args.pdb_debug or torch.cuda.device_count() == 1:
        sampler = None 
    else:
        sampler = InferenceSampler(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    # ======================================================
    # 2. load model and prepare inputs
    # ======================================================
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    # ======================================================
    # 3. generate captions
    # ======================================================
    if args.error_report_file == None:
        current_directory = os.getcwd()
        args.error_report_file = os.path.join(current_directory, "failed_captioning_error.log")
    print("Setting error logging path: {}".format(args.error_report_file))
    error_logger = open(args.error_report_file, mode='a', encoding='utf-8')
    # prepare prompts
    assert args.prompt_set in PROMPT_SETS.keys()
    prompts = PROMPT_SETS[args.prompt_set]

    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # prepare inputs
        video_path = data['videos'][0] # the batch must be 1
        try:
            images = load_video(video_path, args.num_video_frames)
            video = process_images(images, image_processor, model.config).half().cuda()
            video = [video]

            qs = f"<video>\n {random.choice(prompts)}"
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=video,
                    attention_mask=attention_masks,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # print("Question:", qs)
            # print("Answer:", outputs)            
            # save results
            caption_file = os.path.join(args.caption_folder, os.path.splitext(os.path.basename(video_path))[0] + ".json").encode('utf-8')
            os.makedirs(os.path.dirname(caption_file), exist_ok=True)
            with open(caption_file, 'w', encoding='utf-8') as f:
                json.dump({"video": video_path, "captions": outputs}, f, indent=4, ensure_ascii=False)
        except Exception as e:
            error_logger.write(video_path + '\n')
            print("Error processing : ", video_path)
            print(e)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-list-file", type=str)
    parser.add_argument("--caption-folder", type=str)
    parser.add_argument("--error-report-file", type=str, default=None )
    parser.add_argument('--local-rank', type=int, default=0)    
    parser.add_argument("--model-path", type=str, default="Fr0zencr4nE/Cockatiel-13B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--prompt_set", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--pdb_debug", action="store_true")
    args = parser.parse_args()

    main(args)