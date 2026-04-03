import random
import torch
import os
import os.path as osp
import cv2
import numpy as np
from run_infinity import *
from utiles import*
from nudenet import NudeDetector

OBJECT_LIST = ['church', 'french_horn', 'gas_pump', 'golf_ball', 'parachute']


detector = NudeDetector()
def dec_img_nude(img_path, beta=0.5):

    target_labels = {
        'FEMALE_GENITALIA_EXPOSED',
        'MALE_GENITALIA_EXPOSED',
        'FEMALE_BREAST_EXPOSED',
        'BUTTOCKS_EXPOSED'
    }

    detections = detector.detect(img_path)
    is_nude = any(
        detection['class'] in target_labels and detection['score'] > beta
        for detection in detections
    )
    
    return is_nude

model_path = './weights/infinity_2b_reg.pth'
vae_path = './weights/infinity_vae_d32reg.pth'
text_encoder_ckpt = './weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001'



base_args = argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch',
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    enable_model_cache=0
)


parser = argparse.ArgumentParser(description='config')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--concept', type=str, default='nudity', help='concept to remove')

parser.add_argument('--pr', type=str, default='nude500', help='prompts to generate, naked coco adv') 
parser.add_argument('--cfg', type=float, default=2.5, help='cfg para less cfg means more diversity')
parser.add_argument('--multi', type=int, default=20, help='how times evert prompt repeat')
parser.add_argument('--attn', type=int, default=20, help='how many top percent codes in attn map')
parser.add_argument('--ssi', type=int, default=2, help='the scale index which start to use our method') 
parser.add_argument('--esi', type=int, default=9, help='the scale index which stop using our method')

args = parser.parse_args(namespace=argparse.Namespace(**base_args.__dict__))



os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


torch.cuda.set_device(args.gpu)


current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir, 'model-infinity')

current_dir = os.path.join(current_dir, "concept-" + args.concept)


current_dir = os.path.join(current_dir, 'SaCo')


temp_path = args.pr + '_multi' + str(args.multi) + '_cfg' + f"{args.cfg:.1f}" + '_attn' + str(args.attn) + '_' + str(args.ssi) + 'si' + str(args.esi)



current_dir = os.path.join(current_dir, temp_path)
os.makedirs(current_dir, exist_ok=True)
output_dir = current_dir

# LOAD
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
vae = load_visual_tokenizer(args)
infinity = load_transformer(vae, args)







if args.pr == 'nude500':
    pos_prompts = load_dict_from_csv('./prompts_data/nude500.csv')
elif args.pr in OBJECT_LIST:
    neg_prompts = load_dict_from_csv('./prompts_data/' + args.concept + '_neg.csv')
    pos_prompts = load_dict_from_csv('./prompts_data/' + args.concept + '_pos.csv')

if args.concept == "nudity":
    target_word = "naked"
else:
    target_word = args.concept
    
    

# OUTPUT


img_path = os.path.join(output_dir, 'img')
os.makedirs(img_path, exist_ok=True)

all_sen_code_list = [[] for _ in range(args.ssi, args.esi)]

all_pos_code_list = [[] for _ in range(args.ssi, args.esi)]


# GEN IMG
for ind, prompt in pos_prompts.items():
    cfg = args.cfg
    tau = 0.5
    h_div_w = 1/1 # Aspect Ratio
    
    enable_positive_prompt = 0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    pos_prompt = prompt
    if args.pr == 'nude500':
        neg_prompt = insert_naked(pos_prompt)
    elif args.pr in OBJECT_LIST:
        neg_prompt = neg_prompts[ind]
    else:
        neg_prompt = neg_prompts[ind]


    if target_word not in neg_prompt:
        print("continue")
        continue


    for k in range(args.multi):

        seed = random.randint(0, 10000)
        
        neg_image, neg_idx_list, neg_code_list, pos_code_list = infer4code(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            neg_prompt,
            pos_prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=enable_positive_prompt,
            target_word=target_word,
            top_attn=args.attn,
            ssi=args.ssi,
            esi=args.esi,

        )


        # SAVE
        save_path = osp.join(img_path, f"re_{ind}_test_{k}.jpg")
        cv2.imwrite(save_path, neg_image.cpu().numpy())



        if args.concept == 'nudity':
            if dec_img_nude(save_path):
                for si in range(args.ssi, args.esi):
                    if neg_code_list[si - args.ssi] is not None:
                        all_sen_code_list[si - args.ssi].append(neg_code_list[si - args.ssi])
                        all_pos_code_list[si - args.ssi].append(pos_code_list[si - args.ssi])
        else:
            for si in range(args.ssi, args.esi):
                if neg_code_list[si - args.ssi] is not None:
                    all_sen_code_list[si - args.ssi].append(neg_code_list[si - args.ssi])
                    all_pos_code_list[si - args.ssi].append(pos_code_list[si - args.ssi])


for si in range(args.ssi, args.esi):
    temp_sen_code = all_sen_code_list[si - args.ssi]
    temp_pos_code = all_pos_code_list[si - args.ssi]
    if temp_sen_code:
        temp_all_mult_scale_sen = torch.cat(temp_sen_code, dim=0).cpu()
        temp_all_mult_scale_pos = torch.cat(temp_pos_code, dim=0).cpu()
        unq_neg, unq_pos = unq_code(temp_all_mult_scale_sen, temp_all_mult_scale_pos)

        # print(unq_neg.shape[0])

        torch.save(unq_neg, os.path.join(output_dir, "scale" + str(si) + "_sen.pt"))
        torch.save(unq_pos, os.path.join(output_dir, "scale" + str(si) + "_pos.pt"))

    else:
        continue
    
print("Code saved to: " + output_dir)

