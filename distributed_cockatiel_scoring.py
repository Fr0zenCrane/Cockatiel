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

PROMPT_TEMPLATES = {
    "system": "You are an expert in evaluating video captions, specifically focusing on how effectively they align with and describe the content of the videos.",

    "object": "Evaluate how well the caption describes the objects in the video. Use the following scoring options:\n 0.Not Involved: The video and the caption do not involve any objects.\n 1.Totally Incorrect: All descriptions are incorrect or missed.\n 2.Mainly Incorrect: Most descriptions are incorrect or missed, with only a few correct.\n 3.Moderately Incorrect: Some descriptions are correct, while others are incorrect or missed.\n 4.Mainly Correct: Most descriptions are correct, with only a few incorrect or missed.\n 5.Totally Correct: All descriptions are correct.",

    "static_attribute": "Evaluate how well the caption describes the static attributes of objects in the video (e.g., color, shape, size, and texture). Use the following scoring options:\n 0.Not Involved: The video and caption do not involve any static attributes.\n 1.Totally Incorrect: All descriptions are incorrect or missed.\n 2.Mainly Incorrect: Most descriptions are incorrect or missed, with only a few correct.\n 3.Moderately Incorrect: Some descriptions are correct, while others are incorrect or missed.\n 4.Mainly Correct: Most descriptions are correct, with only a few incorrect or missed.\n 5.Totally Correct: All descriptions are correct.",

    "action": "Evaluate how well the caption describes the dynamic attributes of objects in the video, such as movement, action and interaction. Use the following scoring options:\n 0.Not Involved: The video and caption do not involve any dynamic attributes.\n 1.Totally Incorrect: All descriptions are incorrect or missed.\n 2.Mainly Incorrect: Most descriptions are incorrect or missed, with only a few correct.\n 3.Moderately Incorrect: Some descriptions are correct, while others are incorrect or missed.\n 4.Mainly Correct: Most descriptions are correct, with only a few incorrect or missed.\n 5.Totally Correct: All descriptions are correct.",

    "camera": "Evaluate how well the caption describes the camera movement in the video, including pans, tilts, and zooms. Use the following scoring options:\n 0.Not Involved: The video and caption do not involve any camera movements.\n 1.Totally Incorrect: All descriptions are incorrect or missed.\n 2.Mainly Incorrect: Most descriptions are incorrect or missed, with only a few correct.\n 3.Moderately Incorrect: Some descriptions are correct, while others are incorrect or missed.\n 4.Mainly Correct: Most descriptions are correct, with only a few incorrect or missed.\n 5.Totally Correct: All descriptions are correct.",

    "background": "Evaluate how well the caption describes the background (such as setting and context) in the video. Use the following scoring options:\n 0.Not Involved: The video and caption do not involve any background elements.\n 1.Totally Incorrect: All descriptions are incorrect or missed.\n 2.Mainly Incorrect: Most descriptions are incorrect or missed, with only a few correct.\n 3.Moderately Incorrect: Some descriptions are correct, while others are incorrect or missed.\n 4.Mainly Correct: Most descriptions are correct, with only a few incorrect or missed.\n 5.Totally Correct: All descriptions are correct."
}

OPTION_TEMPLATES = [
    "0.Not Involved: The video and caption do not involve any {}.",
    "1.Totally Incorrect: All descriptions are incorrect or missed.",
    "2.Mainly Incorrect: Most descriptions are incorrect or missed, with only a few correct.",
    "3.Moderately Incorrect: Some descriptions are correct, while others are incorrect or missed.",
    "4.Mainly Correct: Most descriptions are correct, with only a few incorrect or missed.",
    "5.Totally Correct: All descriptions are correct."
]
ZERO_SCORE_OPTION_TEMPLATE = "0.Not Involved: The video and caption do not involve any {}."
SUPPORTED_DIMENSIONS = ["object", "static_attribute", "action", "camera", "background"]


class VideoDataset(Dataset):
    def __init__(self, data_file, caption_folder):
        self.data = json.load(open(data_file))    
        if not os.path.exists(caption_folder):
            os.makedirs(caption_folder, exist_ok=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dataItem = self.data[index]
        video = dataItem["video"]
        caption = dataItem["captions"] # expected to be a list of strings or a single string
        if type(dataItem["captions"]) == str:
            caption = [caption]
        else:
            assert type(caption) == list
    
        finalDataItem = {
            "video": video,
            "captions":caption
        }
        return finalDataItem
    

def collate_fn(samples):
    return {
        'videos': [sample["video"] for sample in samples],
        "captions": [sample["captions"] for sample in samples]
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
    dataset = VideoDataset(data_file=args.data_json_file, caption_folder=args.result_folder)
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
        args.error_report_file = os.path.join(current_directory, "failed_scoring_error.log")
    print("Setting error logging path: {}".format(args.error_report_file))
    error_logger = open(args.error_report_file, mode='a', encoding='utf-8')
    
    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # prepare inputs
        video_path = data['videos'][0] # the batch must be 1
        captions = data['captions'][0]

        images = load_video(video_path, args.num_video_frames)
        video = process_images(images, image_processor, model.config).half().cuda()
        video = [video]
        vila_selected_scores = {}
        valid_scores = ["0", "1", "2", "3", "4", "5"]
        for dimension in SUPPORTED_DIMENSIONS:
            vila_selected_scores[dimension] = []
            OPTION_TEMPLATES[0] = ZERO_SCORE_OPTION_TEMPLATE.format(dimension)
            for caption in captions:
                prompt = PROMPT_TEMPLATES["system"] + '\n' + PROMPT_TEMPLATES[dimension] + '\n' + "Caption:\n" + caption + "\nOptions:" + '\n'.join(OPTION_TEMPLATES) + "\nAnswer:"
                qs = f"<video>\n{prompt}"
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
                    logits = model.get_logits(
                        input_ids=input_ids,
                        images=video,
                        attention_mask=attention_masks,
                    ).logits
            
                score_logits = {}
                for score in valid_scores:
                    score_idx = tokenizer.convert_tokens_to_ids(score)
                    score_logits[score] = logits[0, -1, score_idx].item()
                vila_selected_score = max(score_logits, key=score_logits.get)
                vila_selected_scores[dimension].append(int(vila_selected_score))
        
        average_scores = [0] * len(captions)
        for i in range(len(average_scores)):
            for dimension in SUPPORTED_DIMENSIONS:
                average_scores[i] += vila_selected_scores[dimension][i]
            average_scores[i] /= len(SUPPORTED_DIMENSIONS)
        
        vila_selected_scores["average"] = average_scores

        result_file = os.path.join(args.result_folder, os.path.splitext(os.path.basename(video_path))[0] + ".json").encode('utf-8')
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "video": video_path, 
                    "captions": captions, 
                    "scores": vila_selected_scores,
                }, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json-file", type=str)
    parser.add_argument("--result-folder", type=str, default="/zju_0038/datasets/text-to-video/webvid/llava_caption")
    parser.add_argument("--error-report-file", type=str, default=None )
    parser.add_argument('--local-rank', type=int, default=0)    
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-2.7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=32)
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