import random
import torch
import os
import os.path as osp
import cv2
import numpy as np
from run_infinity import *
import csv
import json
import nudenet
from utiles import*
from safetensors.torch import load_file


OBJECT_LIST = ['church', 'french_horn','gas_pump', 'golf_ball', 'parachute']
OBJECT_EVAL_LIST = [i + '_eval' for i in OBJECT_LIST]


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
parser.add_argument('--pr', type=str, default='nude100', help='prompts to generate, naked coco adv')


parser.add_argument('--code_path', type=str, help='path of sensitive codes and positive codes', default=None)
parser.add_argument('--alpha', type=float, default=3.5, help='the humming distance in our method')
parser.add_argument('--ssi', type=int, default=2, help='the scale index which start to use our method')
parser.add_argument('--esi', type=int, default=8, help='the scale index which stop using our method')
parser.add_argument('--replace_code', type=int, default=1)
parser.add_argument('--rand_mask', type=int, default=0) 


parser.add_argument('--neg', type=float, default=None, help='negetive prompt cfg')

parser.add_argument('--sld', type=str, default=None, help='weak, medium, strong, max')

parser.add_argument('--evar_path', type=str, default=None, help='the path of evar model weights')

args = parser.parse_args(namespace=argparse.Namespace(**base_args.__dict__))


torch.cuda.set_device(args.gpu)



os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# LOAD
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
vae = load_visual_tokenizer(args)
infinity = load_transformer(vae, args)

if args.evar_path is not None:
    evar_weights = load_file(args.evar_path)
    infinity.load_state_dict(evar_weights, strict=False)





if args.pr == 'nude100':
    prompts = load_dict_from_csv('./prompts_data/nude100.csv')
    multi = 5
elif args.pr in OBJECT_EVAL_LIST:
    prompts = load_dict_from_csv('./prompts_data/' + args.pr + '.csv')
    multi = 100
elif args.pr == 'RAB':
    prompts = load_rab('./prompts_data/Nudity_eta_3_K_16.csv')
    multi = 5
elif args.pr == 'i2p':
    prompts = load_dict_from_csv('./prompts_data/i2p_nudity.csv')
    multi = 5



current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir, 'model-infinity')

current_dir = os.path.join(current_dir, 'concept-' + args.concept)


if args.code_path is not None:  # using Safe-CodeBook
    current_dir = args.code_path
    if args.neg is not None:
        current_dir = os.path.join(current_dir, 'SaCo+neg')

        temp_path = 'alpha' + f"{args.alpha:.1f}" + '_' + str(args.ssi) + 'si' + str(args.esi)

        if args.replace_code == 0:
            temp_path = temp_path + '_replacecode' + str(args.replace_code)
        else:
            pass

        temp_path = temp_path + '_neg' + f"{args.neg:.1f}"
        current_dir = os.path.join(current_dir, temp_path)

    elif args.sld is not None:
        current_dir = os.path.join(current_dir, 'SaCo+slvar')

        temp_path = 'alpha' + f"{args.alpha:.1f}" + '_' + str(args.ssi) + 'si' + str(args.esi)
        if args.replace_code == 0:
            temp_path = temp_path + '_replacecode' + str(args.replace_code)
        else:
            pass

        temp_path = temp_path + '_slvar-' + args.sld
        current_dir = os.path.join(current_dir, temp_path)

    elif args.evar_path is not None:
        current_dir = os.path.join(current_dir, 'SaCo+evar')

        temp_path = 'alpha' + f"{args.alpha:.1f}" + '_' + str(args.ssi) + 'si' + str(args.esi)

        if args.replace_code == 0:
            temp_path = temp_path + '_replacecode' + str(args.replace_code)
        else:
            pass

        evar_model_name = os.path.basename(args.evar_path).split('.')[0]
        temp_path = temp_path + '_evar-' + evar_model_name
        current_dir = os.path.join(current_dir, temp_path)

    else:
        current_dir = os.path.join(current_dir, 'SaCo_only')

        temp_path = 'alpha' + f"{args.alpha:.1f}" + '_' + str(args.ssi) + 'si' + str(args.esi)

        if args.replace_code == 0:
            temp_path = temp_path + '_replacecode' + str(args.replace_code)
        else:
            pass
        if args.rand_mask == 1:
             temp_path = temp_path + '_randmask'
        else:
            pass
        current_dir = os.path.join(current_dir, temp_path)


else:  # not using Safe-CodeBook
    if args.neg is not None:  # only use negative prompt
        current_dir = os.path.join(current_dir, 'neg')

        current_dir = os.path.join(current_dir, 'neg' + f"{args.neg:.1f}")

    elif args.sld is not None:
        current_dir = os.path.join(current_dir, 'slvar')

        current_dir = os.path.join(current_dir, 'slvar-' + args.sld)

    elif args.evar_path is not None:
        current_dir = os.path.join(current_dir, 'evar')

        evar_model_name = os.path.basename(args.evar_path).split('.')[0]
        current_dir = os.path.join(current_dir, evar_model_name)

    else:  # vanilla
        current_dir = os.path.join(current_dir, 'vanilla')



current_dir = os.path.join(current_dir, args.pr)
os.makedirs(current_dir, exist_ok=True)

output_dir = current_dir



if args.sld == 'weak':
    sld_guidance_scale=200
    sld_warmup_steps=4
    sld_threshold=0.0
    sld_momentum_scale=0.0
    sld_mom_beta=0.4
elif args.sld == 'medium':
    sld_guidance_scale=1000
    sld_warmup_steps=3
    sld_threshold=0.01
    sld_momentum_scale=0.3
    sld_mom_beta=0.4
elif args.sld == 'strong':
    sld_guidance_scale=2000
    sld_warmup_steps=2
    sld_threshold=0.025
    sld_momentum_scale=0.5
    sld_mom_beta=0.7
elif args.sld == 'max':
    sld_guidance_scale=5000
    sld_warmup_steps=0
    sld_threshold=1.0
    sld_momentum_scale=0.5
    sld_mom_beta=0.7

if args.sld is not None:
    sld_safe_concept = args.concept
    sld_para = (sld_safe_concept, sld_guidance_scale, sld_warmup_steps, sld_threshold, sld_momentum_scale, sld_mom_beta)

meta_infos = []

# GEN IMG
for category, prompt in prompts.items():
    
    cfg = args.neg if args.neg is not None else 2.5
    tau = 0.5
    h_div_w = 1/1 # Aspect Ratio
    
    enable_positive_prompt = 0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    print(scale_schedule)

    if args.pr == 'nude100' or args.pr =='nude500':
        prompt = insert_naked(prompt)

    category_image_paths = []
    

    for k in range(multi):

        seed = random.randint(0, 10000)
       
        generated_image = gen_one_img(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
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
            code_path=args.code_path,
            alpha=args.alpha,
            ssi=args.ssi,
            esi=args.esi,
            replace_code=args.replace_code,

            negative_prompt=args.concept if args.neg is not None else '',
            
            sld_para=sld_para if args.sld is not None else None,
            rand_mask=args.rand_mask
        )

        # SAVE
        save_path = osp.join(output_dir, f"re_{category}_test_{k}.jpg")
        cv2.imwrite(save_path, generated_image.cpu().numpy())
        # print(f"{category} image saved to {save_path}")
        category_image_paths.append(save_path)


    meta_infos.append({
        "gen_image_paths": category_image_paths,
        "prompt": prompt
    })


meta_output_path = osp.join(output_dir, "generated_meta.json")
with open(meta_output_path, 'w') as f:
    json.dump(meta_infos, f, indent=2)

print(f"Image saved to: {output_dir}")