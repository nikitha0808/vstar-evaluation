import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from collections import defaultdict
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.nn import CrossEntropyLoss
from PIL import Image
import math
import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def multiple_choices_inference(image, question, options, tokenizer, model, image_processor):
    conv = conv_templates[args.conv_mode].copy()
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    question_input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image_tensor = image_processor(image, return_tensors='pt')['pixel_values'][0]

    output_question = model(
        question_input_ids,
        use_cache=True,
        images=image_tensor.unsqueeze(0).half().cuda())

    question_logits = output_question.logits
    question_past_key_values = output_question.past_key_values

    loss_list = []

    for option in options:
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], option)
        full_prompt = conv.get_prompt()

        full_input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]

        output_option = model(input_ids=option_answer_input_ids,
                              use_cache=True,
                              attention_mask=torch.ones(1, question_logits.shape[1]+option_answer_input_ids.shape[1], device=full_input_ids.device),
                              past_key_values=question_past_key_values)
        
        logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

        loss_fct = CrossEntropyLoss()
        logits = logits.view(-1, model.config.vocab_size)
        labels = option_answer_input_ids.view(-1)
        loss = loss_fct(logits, labels)

        loss_list.append(loss)

    option_chosen = torch.stack(loss_list).argmin()

    return option_chosen.cpu().item()

    

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    results = {}
    per_type_acc = defaultdict(list)
    all_acc = []
    for test_type in ['direct_attributes', 'relative_position']:
        results[test_type] = []
        folder = os.path.join(args.image_folder, test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for image_file in tqdm(image_files):
            result_single_sample = {}
            image_path = os.path.join(folder, image_file)
            annotation_path = image_path.split('.')[0] + '.json'
            image = Image.open(image_path).convert('RGB')
            annotation = json.load(open(annotation_path))
            question = annotation['question']
            options = annotation['options']
            option_chosen = multiple_choices_inference(image, question, options, tokenizer, model, image_processor)
            correct = 1 if option_chosen==0 else 0
            per_type_acc[test_type].append(correct)
            all_acc.append(correct)

            result_single_sample['question'] = question
            result_single_sample['options'] = options
            result_single_sample['image'] = image_file
            result_single_sample['option_chosen'] = option_chosen
            result_single_sample['correct'] = correct
            results[test_type].append(result_single_sample)
        print(test_type, np.mean(per_type_acc[test_type]))

    print(np.mean(all_acc))
    
    with open(args.answers_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="test_question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
