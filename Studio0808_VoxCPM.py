# -*- coding: utf-8 -*-
import os
import sys
import json
import random
import argparse
import threading
import time

# Force UTF-8 encoding for stdout/stderr in Windows to prevent cp950 codec encode crashes with emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Set HF_ENDPOINT to hf-mirror for fast downloads in Asia
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Monkey-patch torch.jit.script in packaged EXE to bypass "TorchScript requires source access" errors
if getattr(sys, 'frozen', False):
    try:
        import torch
        def dummy_script_method(fn, _rcb=None):
            return fn
        def dummy_script(obj, optimize=True, _frames_up=0, _rcb=None):
            return obj
        torch.jit.script_method = dummy_script_method
        torch.jit.script = dummy_script
    except Exception:
        pass

import tkinter as tk
from tkinter import filedialog, messagebox
import soundfile as sf
import sounddevice as sd
import numpy as np
from PIL import Image

# Try importing CustomTkinter
try:
    import customtkinter as ctk
except ImportError:
    print("未安裝 CustomTkinter / CustomTkinter not installed. Please run: pip install customtkinter")
    sys.exit(1)

# Global variables
VOXCPM_MODEL = None
PLAYBACK_THREAD = None
PLAYBACK_STOP_EVENT = threading.Event()
IS_PLAYING = False

# ==========================================
# 🌐 Complete Multilingual Translation Engine
# ==========================================
I18N = {
    "en": {
        "app_title": "Studio0808 VoxCPM - AI Speech Synthesis Workstation (V20260619)",
        "nav_home": " Home",
        "nav_record": " Live Record",
        "nav_design": " Voice Design",
        "nav_clone": " Voice Clone",
        "nav_ultimate": " Ultimate Clone",
        "nav_settings": " Settings",
        "lang_label": "🌐 Language / ភាសា",
        
        # Home
        "home_title": "Studio0808 VoxCPM AI Speech Synthesis Workstation",
        "home_intro": "An offline desktop workstation powered by OpenBMB's open-source VoxCPM2 2B speech foundation model.",
        "home_notice": "💡 First Run Setup: This tool runs 100% offline. If this is your first time opening the app, please go to 'Settings' to download and verify the official VoxCPM2 weights (7 files, ~4.6GB). Once downloaded, all features are ready for use.",
        "home_hl_title": "Key Features & Highlights",
        "home_hl_content": "• Offline direct continuous audio generation without discrete tokenization for studio realism.\n• [Voice Design]: Create unique voices from scratch with natural language descriptions (age, gender, tone, accent).\n• [Voice Clone]: Replicate any speaker's timbre and emotion using only 3-10s of clean reference audio.\n• [Ultimate Clone]: Pair audio with transcript for seamless, ultra-realistic continuation.\n• Multilingual speech generation supporting 30+ languages, dialects, and accents.\n• 48kHz studio-quality output audio.",
        "home_hw_title": "Hardware Recommendations",
        "home_hw_content": "Model size is 2B parameters (~4.63GB on disk). Inference requires ~3GB-5GB VRAM. An NVIDIA GPU with CUDA is strongly recommended for near-instant generation.",
        "home_lic_title": "Open Source License & Credits",
        "home_lic_content": "• VoxCPM2 model and core inference code originate from OpenBMB's VoxCPM (https://github.com/OpenBMB/VoxCPM).\n• Licensed under the Apache License 2.0.\n• Copyright © 2026 OpenBMB. For academic research and evaluation only. Please comply with local regulations and do not use for unauthorized or illegal speech generation.",
        
        # Design
        "design_title": "Voice Design (Create Voice from Scratch)",
        "design_desc": "No reference audio needed. Add a description in parentheses at the start of your text to generate a unique voice.\nExample: (A gentle young female voice, smiling) Hello, welcome to the speech synthesis workstation!\n💡 Batch Mode: Enter one sentence per line and click 'Batch Synthesis' to generate all lines sequentially.",
        "design_input_lbl": "Target Synthesis Text (include voice description; batch mode: 1 sentence/line):",
        "design_presets_lbl": " Voice Feature Presets",
        "preset_select": "Select Voice",
        
        # Presets menu titles
        "preset_menu_km": "Khmer (Select Voice)",
        "preset_menu_en": "US English (Select Voice)",
        "preset_menu_zh": "Taiwan Mandarin (Select Voice)",
        "preset_menu_ja": "Japanese (Select Voice)",
        "preset_menu_ko": "Korean (Select Voice)",
        
        # Roles
        "role_grandfather": "Elderly Man",
        "role_grandmother": "Elderly Woman",
        "role_young_male": "Young Man",
        "role_young_female": "Young Woman",
        "role_boy": "Young Boy",
        "role_girl": "Young Girl",
        
        # Clone
        "clone_title": "Voice Clone (Fast Replication)",
        "clone_desc": "Only 3-10 seconds of clean reference speech is needed to quickly replicate the speaker's tone, emotion, and cadence.",
        "clone_ref_lbl": "Reference Audio (WAV/MP3/FLAC):",
        "clone_ref_ph": "Select a 3-10s WAV/MP3/FLAC audio file...",
        "clone_target_lbl": "Target Synthesis Text:",
        "clone_default_text": "Hello! This is a voice generated by cloning your vocal timbre. How natural does it sound to you?",
        
        # Ultimate
        "ult_title": "Ultimate Clone (Seamless Continuation)",
        "ult_desc": "Provide both reference audio and its exact transcript. The model seamlessly continues speaking while preserving breathing, inflection, and cadence.",
        "ult_ref_lbl": "Reference Audio (WAV/MP3/FLAC):",
        "ult_ref_ph": "Select reference audio file...",
        "ult_prompt_lbl": "Reference Audio Transcript (Required):",
        "ult_prompt_ph": "Enter the exact transcript spoken in the reference audio...",
        "ult_target_lbl": "Target Synthesis Text (Continues seamlessly after reference speech):",
        "ult_default_text": "Continue speaking this sentence, fully preserving your natural emotional cadence, breathing rhythm, and intonation.",
        
        # Record
        "record_title": "Live Recording Studio (Mic Test)",
        "record_desc": "Record your voice here to test cloning capabilities. The recording can be applied directly to Voice Clone or Ultimate Clone with one click.",
        "record_dev_lbl": "Audio Input Device (Microphone):",
        "record_btn_refresh": "🔄 Refresh",
        "record_card_title": "Please read the text below clearly at normal volume and pace:",
        "record_prompt_text": "This is a live real-time voice recording to test voice cloning. Let's see how realistic it sounds!",
        "record_tip": "💡 Tip: This text takes ~5-7 seconds to read, the golden length for voice cloning (recommended: 3-10 seconds).",
        "record_btn_start": "🔴 Start Recording",
        "record_btn_stop": "⏹ Stop Recording",
        "record_status_ready": "Ready. Click 'Start Recording' to begin reading.",
        "record_status_recording": "🎙️ Recording in progress ({elapsed:.1f}s)...",
        "record_status_done": "✅ Recording finished! {duration:.1f}s (Saved to outputs/user_record_test.wav)",
        "record_post_lbl": "Post-Recording Actions:",
        "record_btn_play": "▶ Play Recording",
        "record_btn_apply_clone": "👥 Apply to Voice Clone",
        "record_btn_apply_ult": "👑 Apply to Ultimate Clone",
        
        # Settings
        "settings_title": "System Settings & Model Management",
        "settings_model_path": "Local Model Directory:",
        "settings_out_path": "Output Audio Directory:",
        "settings_browse": "Browse Folder...",
        "settings_device_lbl": "Model Execution Device:",
        "settings_device_tip": "💡 Tip: If GPU VRAM < 4GB, choose CPU mode to prevent out-of-memory crashes.",
        "settings_dl_header": "VoxCPM2 Official Model Download & Verification",
        "settings_dl_info": "This application runs 100% offline. Download the OpenBMB/VoxCPM2 model weights on first run to deploy locally.\n⚠️ The model consists of 7 required files totaling ~4.63 GB. Download may take several minutes:\n   1. config.json (4.2 KB)           2. special_tokens_map.json (1.6 KB)    3. tokenization_voxcpm2.py (2.8 KB)\n   4. tokenizer_config.json (4.9 KB)  5. tokenizer.json (3.5 MB)             6. audiovae.pth (359.5 MB)\n   7. model.safetensors (4.27 GB)",
        "settings_source_lbl": "Download Mirror Source:",
        "settings_btn_download": "Start Download / Check Model",
        "settings_btn_stop": "Stop Download",
        
        # Devices
        "device_auto": "Auto (Auto-Detect)",
        "device_cuda": "CUDA (NVIDIA GPU)",
        "device_cpu": "CPU (Processor Only)",
        
        # Mirrors
        "mirror_hf": "Hugging Face Official (Recommended)",
        "mirror_hf_mirror": "Hugging Face Mirror",
        "mirror_modelscope": "ModelScope",
        
        # Inference params
        "params_header": "Advanced Inference Settings",
        "param_cfg": "Guidance Scale (CFG):",
        "param_cfg_tip": "Lower = more natural audio, Higher = adheres more strictly to description. (Rec: 2.0-3.0)",
        "param_steps": "Denoising Steps (Steps):",
        "param_steps_tip": "Higher = finer audio quality but takes more time. 10-15 steps is optimal.",
        "param_speed": "Speech Speed (Speed):",
        "param_speed_tip": "Adjust voice speed (0.5x slow - 2.0x fast, default 1.0x)",
        "param_norm": "Text Normalization (Convert numbers/symbols to words)",
        "param_denoise": "Auto-Denoise Reference Audio",
        "param_seed": "Use Fixed Random Seed (Lock Timbre)",
        
        # Actions
        "btn_run": "🚀 Start Speech Synthesis",
        "btn_batch": "📋 Batch Synthesis (Line-by-line)",
        "btn_play": "▶ Play Audio",
        "btn_stop_play": "⏹ Stop Playback",
        "btn_open_folder": "📂 Open Output Folder",
        "btn_browse_short": "Browse...",
        
        # Dialogs / Common
        "dialog_warning": "Warning",
        "dialog_error": "Error",
        "dialog_info": "Information",
        "dialog_confirm": "Confirm",
        "dialog_no_model": "config.json not found in model directory! The model may not be downloaded yet.\nPlease go to 'Settings' to download the model or set the correct model directory.",
        "dialog_invalid_ref": "Please select a valid reference audio file!",
        "dialog_need_prompt": "Ultimate Clone mode requires the transcript of the reference audio!",
        "dialog_invalid_seed": "Random seed must be an integer!",
        "dialog_empty_text": "No text provided for synthesis!",
        "dialog_dl_confirm": "The following missing model files will be downloaded:\n{files}\n\nDo you want to proceed with download?",
        "dialog_dl_complete": "All missing model files have been successfully downloaded to:\n{dir}",
        "dialog_all_files_exist": "All required model files are already present locally. No download needed!",
        "dialog_batch_done": "Batch synthesis completed!\n\n✅ Success: {success} lines\n❌ Failed: {failed} lines\n⏱️ Total time: {elapsed}s\n\nOutput folder:\n{out_dir}",
        "dialog_single_done": "Speech synthesis successful!\nElapsed: {elapsed}s\n\nSaved to:\n{filepath}",
        "status_stopping_dl": "Stopping download...",
        "status_dl_stopped": "Download stopped.",
        "status_dl_failed": "Download failed.",
        "status_dl_complete": "Download complete.",
        "status_preparing_infer": "🔄 Preparing inference in background, please wait...",
    },
    "km": {
        "app_title": "Studio0808 VoxCPM — កម្មវិធីសំយោគសំឡេង AI (V20260619)",
        "nav_home": " ផ្ទាំងដើម",
        "nav_record": " ថតសំឡេងផ្ទាល់",
        "nav_design": " រចនាសំឡេង",
        "nav_clone": " ក្លូនសំឡេង",
        "nav_ultimate": " ក្លូនកម្រិតខ្ពស់",
        "nav_settings": " ការកំណត់",
        "lang_label": "🌐 ភាសា / Language",
        
        # Home
        "home_title": "Studio0808 VoxCPM កម្មវិធីសំយោគសំឡេង AI",
        "home_intro": "កម្មវិធីលើកុំព្យូទ័រក្រៅបណ្តាញ (Offline) ដំណើរការដោយម៉ូដែលកូដចំហ VoxCPM2 ទំហំ 2B របស់ OpenBMB។",
        "home_notice": "💡 សេចក្តីជូនដំណឹងលើកដំបូង៖ កម្មវិធីនេះដំណើរការក្រៅបណ្តាញ ១០០%។ ប្រសិនបើនេះជាលើកដំបូងរបស់អ្នក សូមចូលទៅកាន់ផ្ទាំង 'ការកំណត់' ដើម្បីទាញយក និងពិនិត្យម៉ូដែលផ្លូវការ VoxCPM2 (សរុប ៧ ឯកសារ ប្រហែល 4.6GB)។ បន្ទាប់ពីទាញយករួច មុខងារទាំងអស់នឹងរួចរាល់សម្រាប់ប្រើប្រាស់។",
        "home_hl_title": "លក្ខណៈពិសេសចម្បង និងចំណុចលេចធ្លោ",
        "home_hl_content": "• ដំណើរការក្រៅបណ្តាញ បង្កើតសំឡេងបន្តគុណភាពខ្ពស់ដោយមិនបាច់ប្រើ discrete tokenization សម្រាប់សំឡេងកម្រិតស្ទូឌីយោ។\n• 【រចនាសំឡេង】：មិនត្រូវការសំឡេងយោងឡើយ ដោយគ្រាន់តែពណ៌នាលក្ខណៈសំឡេង (ភេទ, អាយុ, ទឹកដម, ភាសា) ក្នុងវង់ក្រចកដើម្បីបង្កើតសំឡេងថ្មី។\n• 【ក្លូនសំឡេង】：គាំទ្រសំឡេងខ្លី 3-10 វិនាទី ដើម្បីចម្លងសំឡេង និងអារម្មណ៍របស់អ្នកនិយាយបានយ៉ាងត្រឹមត្រូវ។\n• 【ក្លូនកម្រិតខ្ពស់】：រួមបញ្ចូលសំឡេងយោង និងអត្ថបទដើម ដើម្បីបន្តការនិយាយបានយ៉ាងរលូន និងមានលក្ខណៈធម្មជាតិបំផុត។\n• គាំទ្រការសំយោគរហូតដល់ជាង ៣០ ភាសា រួមទាំងគ្រាមភាសាផ្សេងៗ។\n• គុណភាពសំឡេងចេញកម្រិតស្ទូឌីយោ 48kHz។",
        "home_hw_title": "អនុសាសន៍ផ្នែករឹង (Hardware)",
        "home_hw_content": "ម៉ូដែលនេះមានទំហំ 2B (ប្រហែល ២ ពាន់លានប៉ារ៉ាម៉ែត្រ) ទំហំឯកសារប្រហែល 4.63GB។ ត្រូវការ VRAM ប្រហែល 3GB-5GB។ ណែនាំយ៉ាងខ្លាំងឱ្យប្រើក្រាហ្វិកកាត NVIDIA GPU (CUDA) ដើម្បីទទួលបានល្បឿនលឿនក្នុងរយៈពេលប៉ុន្មានវិនាទី។",
        "home_lic_title": "អាជ្ញាប័ណ្ណកូដចំហ និងការរក្សាសិទ្ធិ",
        "home_lic_content": "• ម៉ូដែល VoxCPM2 និងកូដសំយោគចម្បងបានមកពីគម្រោងកូដចំហ VoxCPM របស់ OpenBMB (https://github.com/OpenBMB/VoxCPM)។\n• ស្ថិតក្រោមអាជ្ញាប័ណ្ណ Apache License 2.0។\n• រក្សាសិទ្ធិ © 2026 OpenBMB។ សម្រាប់តែការស្រាវជ្រាវ ការធ្វើតេស្ត និងការវាយតម្លៃប៉ុណ្ណោះ។ សូមគោរពច្បាប់ និងកុំប្រើប្រាស់ក្នុងគោលបំណងខុសច្បាប់។",
        
        # Design
        "design_title": "រចនាសំឡេង (Voice Design)",
        "design_desc": "មិនត្រូវការសំឡេងយោងទេ។ គ្រាន់តែបន្ថែមការពណ៌នាក្នុងវង់ក្រចកនៅដើមអត្ថបទ ដើម្បីបង្កើតសំឡេងជាក់លាក់។\nឧទាហរណ៍៖ (A gentle young female voice, smiling) សួស្តី! សូមស្វាគមន៍មកកាន់កម្មវិធីសំយោគសំឡេង AI!\n💡 របៀបជាបាច់៖ មួយជួរមួយប្រយោគ ចុច 'សំយោគជាបាច់' ដើម្បីបង្កើតតាមលំដាប់ជួរទាំងអស់។",
        "design_input_lbl": "អត្ថបទគោលដៅសម្រាប់សំយោគ (សូមដាក់ការពណ៌នាក្នុងវង់ក្រចក; របៀបជាបាច់: ១ ប្រយោគ/ជួរ)៖",
        "design_presets_lbl": " គំរូសំឡេងរហ័ស",
        "preset_select": "ជ្រើសរើសសំឡេង",
        
        # Presets menu titles
        "preset_menu_km": "ភាសាខ្មែរ (ជ្រើសរើសសំឡេង)",
        "preset_menu_en": "ភាសាអង់គ្លេស (ជ្រើសរើសសំឡេង)",
        "preset_menu_zh": "ភាសាចិនតៃវ៉ាន់ (ជ្រើសរើសសំឡេង)",
        "preset_menu_ja": "ភាសាជប៉ុន (ជ្រើសរើសសំឡេង)",
        "preset_menu_ko": "ភាសាកូរ៉េ (ជ្រើសរើសសំឡេង)",
        
        # Roles
        "role_grandfather": "លោកតា (បុរសវ័យចំណាស់)",
        "role_grandmother": "លោកយាយ (ស្ត្រីវ័យចំណាស់)",
        "role_young_male": "បុរសវ័យក្មេង",
        "role_young_female": "ស្ត្រីវ័យក្មេង",
        "role_boy": "ក្មេងប្រុស",
        "role_girl": "ក្មេងស្រី",
        
        # Clone
        "clone_title": "ក្លូនសំឡេង (Voice Clone)",
        "clone_desc": "ត្រូវការតែសំឡេងយោងច្បាស់ 3-10 វិនាទីប៉ុណ្ណោះ ដើម្បីចម្លងសំឡេង អារម្មណ៍ និងល្បឿននិយាយរបស់អ្នកនិយាយ។",
        "clone_ref_lbl": "ឯកសារសំឡេងយោង (WAV/MP3/FLAC)៖",
        "clone_ref_ph": "ជ្រើសរើសឯកសារសំឡេង 3-10 វិនាទី (WAV/MP3/FLAC)...",
        "clone_target_lbl": "អត្ថបទគោលដៅសម្រាប់សំយោគ៖",
        "clone_default_text": "សួស្តី! នេះគឺជាសំឡេងដែលខ្ញុំបានក្លូនចេញពីសំឡេងរបស់អ្នក។ តើអ្នកស្តាប់ទៅមានអារម្មណ៍ធម្មជាតិដែរឬទេ?",
        
        # Ultimate
        "ult_title": "ក្លូនកម្រិតខ្ពស់ (Ultimate Clone)",
        "ult_desc": "ផ្តល់ទាំងឯកសារសំឡេងយោង និងអត្ថបទដើមដែលនិយាយក្នុងសំឡេងនោះ។ ម៉ូដែលនឹងបន្តការនិយាយដោយរក្សាអារម្មណ៍ ការដកដង្ហើម និងទឹកដមសំនៀងយ៉ាងរលូនឥតខ្ចោះ។",
        "ult_ref_lbl": "ឯកសារសំឡេងយោង (WAV/MP3/FLAC)៖",
        "ult_ref_ph": "ជ្រើសរើសឯកសារសំឡេងយោង WAV/MP3/FLAC...",
        "ult_prompt_lbl": "អត្ថបទដើមនៃសំឡេងយោង (ចាំបាច់)៖",
        "ult_prompt_ph": "បញ្ចូលអត្ថបទដែលបាននិយាយក្នុងសំឡេងយោង...",
        "ult_target_lbl": "អត្ថបទគោលដៅសម្រាប់សំយោគ (នឹងនិយាយបន្តបន្ទាប់ពីសំឡេងយោង)៖",
        "ult_default_text": "បន្តនិយាយប្រយោគនេះ ដោយរក្សាអារម្មណ៍ដើម ចង្វាក់ដកដង្ហើម និងទឹកដមសំនៀងរបស់អ្នកទាំងស្រុង។",
        
        # Record
        "record_title": "បន្ទប់ថតសំឡេងផ្ទាល់ (Live Recording)",
        "record_desc": "ថតសំឡេងរបស់អ្នកនៅទីនេះដើម្បីសាកល្បងការក្លូន។ សំឡេងដែលថតរួចអាចយកទៅប្រើក្នុង 'ក្លូនសំឡេង' ឬ 'ក្លូនកម្រិតខ្ពស់' ដោយចុចតែម្តង។",
        "record_dev_lbl": "ឧបករណ៍បញ្ចូលសំឡេង (មីក្រូហ្វូន)៖",
        "record_btn_refresh": "🔄 ផ្ទុកឡើងវិញ",
        "record_card_title": "សូមអានអត្ថបទខាងក្រោមឱ្យបានច្បាស់ ក្នុងកម្រិតសំឡេង និងល្បឿនធម្មតា៖",
        "record_prompt_text": "នេះគឺការថតសំឡេងផ្ទាល់ក្នុងពេលជាក់ស្តែង សម្រាប់ប្រើក្នុងមុខងារក្លូនសំឡេង ដើម្បីមើលថាតើវាមានប្រសិទ្ធភាពកម្រិតណា!",
        "record_tip": "💡 គន្លឹះ៖ អត្ថបទនេះចំណាយពេលអានប្រហែល 5-7 វិនាទី ដែលជាប្រវែងល្អបំផុតសម្រាប់ក្លូនសំឡេង (ណែនាំ 3-10 វិនាទី)។",
        "record_btn_start": "🔴 ចាប់ផ្តើមថត",
        "record_btn_stop": "⏹ បញ្ឈប់ការថត",
        "record_status_ready": "រួចរាល់។ សូមចុច 'ចាប់ផ្តើមថត' ដើម្បីចាប់ផ្តើមអាន។",
        "record_status_recording": "🎙️ កំពុងថតសំឡេង ({elapsed:.1f} វិនាទី)...",
        "record_status_done": "✅ ការថតរួចរាល់! {duration:.1f} វិនាទី (បានរក្សាទុកក្នុង outputs/user_record_test.wav)",
        "record_post_lbl": "សកម្មភាពបន្ទាប់ពីថតរួច៖",
        "record_btn_play": "▶ ចាក់ការថត",
        "record_btn_apply_clone": "👥 ប្រើក្នុង ក្លូនសំឡេង",
        "record_btn_apply_ult": "👑 ប្រើក្នុង ក្លូនកម្រិតខ្ពស់",
        
        # Settings
        "settings_title": "ការកំណត់ប្រព័ន្ធ និងគ្រប់គ្រងម៉ូដែល",
        "settings_model_path": "ផ្លូវផ្ទុកម៉ូដែលក្នុងម៉ាស៊ីន៖",
        "settings_out_path": "ផ្លូវផ្ទុកសំឡេងដែលបានបង្កើត៖",
        "settings_browse": "ជ្រើសរើសថត...",
        "settings_device_lbl": "ឧបករណ៍ដំណើរការម៉ូដែល៖",
        "settings_device_tip": "💡 គន្លឹះ៖ ប្រសិនបើ GPU VRAM < 4GB សូមជ្រើសរើស CPU ដើម្បីការពារកុំឱ្យខ្វះអង្គចងចាំ។",
        "settings_dl_header": "ការទាញយក និងពិនិត្យម៉ូដែលផ្លូវការ VoxCPM2",
        "settings_dl_info": "កម្មវិធីនេះដំណើរការក្រៅបណ្តាញទាំងស្រុង។ សម្រាប់ការប្រើប្រាស់លើកដំបូង សូមទាញយកម៉ូដែល OpenBMB/VoxCPM2 ដើម្បីដំឡើងក្នុងម៉ាស៊ីនរបស់អ្នក។\n⚠️ ម៉ូដែលនេះរួមបញ្ចូលឯកសារចាំបាច់ចំនួន ៧ សរុបប្រហែល 4.63 GB។ ការទាញយកអាចចំណាយពេលខ្លះ៖\n   1. config.json (4.2 KB)           2. special_tokens_map.json (1.6 KB)    3. tokenization_voxcpm2.py (2.8 KB)\n   4. tokenizer_config.json (4.9 KB)  5. tokenizer.json (3.5 MB)             6. audiovae.pth (359.5 MB)\n   7. model.safetensors (4.27 GB)",
        "settings_source_lbl": "ប្រភពម៉ាស៊ីនបម្រើទាញយក៖",
        "settings_btn_download": "ចាប់ផ្តើមទាញយក / ពិនិត្យម៉ូដែល",
        "settings_btn_stop": "បញ្ឈប់ការទាញយក",
        
        # Devices
        "device_auto": "Auto (រកឃើញដោយស្វ័យប្រវត្តិ)",
        "device_cuda": "CUDA (ក្រាហ្វិកកាត NVIDIA)",
        "device_cpu": "CPU (ដំណើរការដោយប្រព័ន្ធដំណើរការ)",
        
        # Mirrors
        "mirror_hf": "Hugging Face ផ្លូវការ (ណែនាំ)",
        "mirror_hf_mirror": "Hugging Face ម៉ាស៊ីនឆ្លុះ (Mirror)",
        "mirror_modelscope": "ModelScope",
        
        # Inference params
        "params_header": "ការកំណត់ប៉ារ៉ាម៉ែត្រកម្រិតខ្ពស់",
        "param_cfg": "មេគុណណែនាំ (CFG)៖",
        "param_cfg_tip": "ទាប = សំឡេងធម្មជាតិល្អ, ខ្ពស់ = ត្រូវតាមការពណ៌នាកាន់តែច្បាស់ (ណែនាំ 2.0-3.0)",
        "param_steps": "ជំហានកាត់បន្ថយសំឡេងរំខាន (Steps)៖",
        "param_steps_tip": "កាន់តែច្រើន = សំឡេងកាន់តែច្បាស់តែចំណាយពេលយូរ។ 10-15 គឺល្អបំផុត។",
        "param_speed": "ល្បឿននៃការនិយាយ (Speed)៖",
        "param_speed_tip": "កែសម្រួលល្បឿននិយាយ (0.5x យឺត - 2.0x លឿន, លំនាំដើម 1.0x)",
        "param_norm": "បម្លែងអត្ថបទទៅជាស្តង់ដារ (លេខ/និមិត្តសញ្ញាទៅជាពាក្យ)",
        "param_denoise": "កាត់បន្ថយសំឡេងរំខាននៃឯកសារយោងដោយស្វ័យប្រវត្តិ",
        "param_seed": "ប្រើគ្រាប់ពូជចៃដន្យថេរ (ចាក់សោរសំឡេង)",
        
        # Actions
        "btn_run": "🚀 ចាប់ផ្តើមសំយោគសំឡេង",
        "btn_batch": "📋 សំយោគជាបាច់ (តាមជួរ)",
        "btn_play": "▶ ចាក់សំឡេងដែលបានបង្កើត",
        "btn_stop_play": "⏹ បញ្ឈប់ការចាក់",
        "btn_open_folder": "📂 បើកថតផ្ទុកឯកសារ",
        "btn_browse_short": "ស្វែងរក...",
        
        # Dialogs / Common
        "dialog_warning": "ការព្រមាន",
        "dialog_error": "កំហុស",
        "dialog_info": "ព័ត៌មាន",
        "dialog_confirm": "ការបញ្ជាក់",
        "dialog_no_model": "រកមិនឃើញ config.json ក្នុងផ្លូវផ្ទុកម៉ូដែលឡើយ! ប្រហែលជាមិនទាន់បានទាញយកម៉ូដែល។\nសូមចូលទៅកាន់ផ្ទាំង 'ការកំណត់' ដើម្បីទាញយកម៉ូដែល ឬជ្រើសរើសថតដែលត្រឹមត្រូវ។",
        "dialog_invalid_ref": "សូមជ្រើសរើសឯកសារសំឡេងយោងត្រឹមត្រូវ!",
        "dialog_need_prompt": "របៀបក្លូនកម្រិតខ្ពស់តម្រូវឱ្យមានអត្ថបទដើមនៃសំឡេងយោង!",
        "dialog_invalid_seed": "គ្រាប់ពូជចៃដន្យត្រូវតែជាចំនួនគត់!",
        "dialog_empty_text": "គ្មានអត្ថបទសម្រាប់សំយោគឡើយ!",
        "dialog_dl_confirm": "ឯកសារម៉ូដែលដែលខ្វះខាតខាងក្រោមនឹងត្រូវទាញយក៖\n{files}\n\nតើអ្នកពិតជាចង់ទាញយកមែនទេ?",
        "dialog_dl_complete": "ឯកសារម៉ូដែលដែលខ្វះទាំងអស់ត្រូវបានទាញយកដោយជោគជ័យទៅកាន់៖\n{dir}",
        "dialog_all_files_exist": "ឯកសារម៉ូដែលចាំបាច់ទាំងអស់មានរួចរាល់ហើយ មិនចាំបាច់ទាញយកឡើងវិញទេ។",
        "dialog_batch_done": "ការសំយោគជាបាច់បានបញ្ចប់!\n\n✅ ជោគជ័យ: {success} ប្រយោគ\n❌ បរាជ័យ: {failed} ប្រយោគ\n⏱️ រយៈពេលសរុប: {elapsed} វិនាទី\n\nថតផ្ទុកលទ្ធផល:\n{out_dir}",
        "dialog_single_done": "ការសំយោគសំឡេងជោគជ័យ!\nរយៈពេល: {elapsed} វិនាទី\n\nបានរក្សាទុកក្នុង:\n{filepath}",
        "status_stopping_dl": "កំពុងបញ្ឈប់ការទាញយក...",
        "status_dl_stopped": "ការទាញយកត្រូវបានបញ្ឈប់។",
        "status_dl_failed": "ការទាញយកបរាជ័យ។",
        "status_dl_complete": "ការទាញយកបានបញ្ចប់។",
        "status_preparing_infer": "🔄 កំពុងរៀបចំការសំយោគនៅផ្ទៃខាងក្រោយ សូមរង់ចាំ...",
    },
    "zh": {
        "app_title": "Studio0808 VoxCPM 語音合成工具站 (V20260619)",
        "nav_home": " 首頁",
        "nav_record": " 即時錄音",
        "nav_design": " 語音設計",
        "nav_clone": " 聲音複製",
        "nav_ultimate": " 極限複製",
        "nav_settings": " 系統設定",
        "lang_label": "🌐 語言選擇 / Language",
        
        # Home
        "home_title": "Studio0808 VoxCPM 語音合成工具站",
        "home_intro": "本工具是基於 OpenBMB 最新開源的語音合成模型 VoxCPM2 所開發的獨立測試程式。",
        "home_notice": "💡 首次使用提示：本工具為全離線運行。若您是第一次開啟本程式，請務必先前往「系統設定」分頁下載並檢查 VoxCPM2 官方模型權重（共 7 個檔案，約 4.6GB），下載完成後即可開始使用語音合成與複製功能。",
        "home_hl_title": "主要特色與亮點",
        "home_hl_content": "• 離線免除 discrete tokenization，直接產生高品質連續音訊表現。\n• 【語音設計】：免參考語音，僅用語音特徵描述（性別、語氣、年齡等）即可無中生有生成聲音。\n• 【聲音複製】：支援 3~10 秒極短音訊即可模仿對方音色與情緒。\n• 【極限複製】：配合參考逐字稿，達成無瑕疵、極致真實的語氣接續合成。\n• 支援高達 30 種語言合成（包含粵語、閩南語等漢語方言）。\n• 輸出品質直接高達 48kHz 錄音室等級音質。",
        "home_hw_title": "硬體建議",
        "home_hw_content": "本模型參數量為 2B (約 20億)，推論需要約 3GB~5GB VRAM。強烈建議配備 NVIDIA 顯示卡以獲得秒級生成體驗。",
        "home_lic_title": "開源授權與版權宣告 (Open Source License & Credits)",
        "home_lic_content": "• 本專案使用之 VoxCPM2 模型及其核心推論代碼源自開源專案 VoxCPM (https://github.com/OpenBMB/VoxCPM)。\n• VoxCPM 原專案採用 Apache License 2.0 授權協議。\n• 版權所有 © 2026 OpenBMB。本工具僅供個人學術研究、測試與開發評估之用，請遵守相關法律法規，勿用於非法或未授權的語音合成傳播。",
        
        # Design
        "design_title": "語音設計 (Voice Design)",
        "design_desc": "無須上傳參考音訊。直接在文字開頭加上括號與音色特徵描述即可生成特定聲線。\n例如：(A gentle young female voice, smiling) 哈囉，歡迎來到語音合成工作站！\n💡 批次模式：每行一句，按「批次合成」可逐行依序全部生成。",
        "design_input_lbl": "目標合成文字 (請包含描述，批次模式每行一句):",
        "design_presets_lbl": " 語音特徵快速範本",
        "preset_select": "選擇音色",
        
        # Presets menu titles
        "preset_menu_km": "高棉語/柬埔寨語 (選擇音色)",
        "preset_menu_en": "美國英語 (選擇音色)",
        "preset_menu_zh": "臺灣國語 (選擇音色)",
        "preset_menu_ja": "日本語 (選擇音色)",
        "preset_menu_ko": "韓語 (選擇音色)",
        
        # Roles
        "role_grandfather": "老爺爺",
        "role_grandmother": "老奶奶",
        "role_young_male": "年輕男生",
        "role_young_female": "年輕女生",
        "role_boy": "小男孩",
        "role_girl": "小女孩",
        
        # Clone
        "clone_title": "聲音複製 (Voice Clone)",
        "clone_desc": "只需 3-10 秒乾淨人聲音訊，即可完美複製對方的音色、情感與說話節奏。",
        "clone_ref_lbl": "參考語音檔案 (WAV/MP3/FLAC):",
        "clone_ref_ph": "請選取 3~10秒 的 WAV/MP3/FLAC 音檔...",
        "clone_target_lbl": "目標合成文字:",
        "clone_default_text": "你好，這是我複製你的音色所說出來的一段話。聽起來效果如何？",
        
        # Ultimate
        "ult_title": "極限複製 (Ultimate Clone)",
        "ult_desc": "同時提供參考音檔及其對應的逐字稿。模型會無縫延續該段語氣，達成最高品質與真實度的模仿。",
        "ult_ref_lbl": "參考語音檔案 (WAV/MP3/FLAC):",
        "ult_ref_ph": "請選取 WAV/MP3/FLAC 參考音檔...",
        "ult_prompt_lbl": "參考語意逐字稿 (必填):",
        "ult_prompt_ph": "輸入參考音檔中說出的完整逐字內容...",
        "ult_target_lbl": "目標合成文字 (將接續在參考語音後面說):",
        "ult_default_text": "接下去說這句話，完全保留你原本說話的情緒、呼吸起伏與語調。",
        
        # Record
        "record_title": "即時錄音測試 (Live Recording Test)",
        "record_desc": "在此錄製您的聲音以測試複製功能。系統將使用您的錄音作為「聲音複製」與「極限複製」的參考音訊。",
        "record_dev_lbl": "選擇錄音輸入設備 (麥克風):",
        "record_btn_refresh": "🔄 重新整理",
        "record_card_title": "請看著下方文本，並以正常音量與速度朗讀：",
        "record_prompt_text": "這是現場即時錄音，可以用在聲音複製功能，看看效果有多逼真！",
        "record_tip": "💡 提示：本文字長度約 5~7 秒，是最適合聲音複製的黃金長度（建議控制在 3~10 秒內）。",
        "record_btn_start": "🔴 開始錄音",
        "record_btn_stop": "⏹ 停止錄音",
        "record_status_ready": "就緒。請點選「開始錄音」以開始讀稿。",
        "record_status_recording": "🎙️ 正在錄音中 ({elapsed:.1f} 秒)...",
        "record_status_done": "✅ 錄音完成！共 {duration:.1f} 秒 (已存至 outputs/user_record_test.wav)",
        "record_post_lbl": "錄音後續作業：",
        "record_btn_play": "▶ 播放錄音",
        "record_btn_apply_clone": "👥 套用至 聲音複製",
        "record_btn_apply_ult": "👑 套用至 極限複製",
        
        # Settings
        "settings_title": "系統設定與模型管理",
        "settings_model_path": "本機模型儲存路徑:",
        "settings_out_path": "生成語音儲存路徑:",
        "settings_browse": "選擇資料夾...",
        "settings_device_lbl": "模型執行裝置:",
        "settings_device_tip": "💡 說明：顯示卡記憶體(VRAM)小於 4GB 建議改選 CPU 模式以防載入崩潰",
        "settings_dl_header": "VoxCPM2 官方模型下載與修補",
        "settings_dl_info": "本程式為全離線運作。首次使用請下載 OpenBMB/VoxCPM2 模型權重，下載完成後會自動部署於本機路徑。\n⚠️ 本模型共包含下列 7 個必要檔案，總計約 4.63 GB，下載過程需要一些時間，請耐心等候：\n   1. config.json (4.2 KB)           2. special_tokens_map.json (1.6 KB)    3. tokenization_voxcpm2.py (2.8 KB)\n   4. tokenizer_config.json (4.9 KB)  5. tokenizer.json (3.5 MB)             6. audiovae.pth (359.5 MB)\n   7. model.safetensors (4.27 GB)",
        "settings_source_lbl": "下載伺服器來源:",
        "settings_btn_download": "開始下載 / 檢查模型",
        "settings_btn_stop": "停止下載",
        
        # Devices
        "device_auto": "Auto (自動偵測)",
        "device_cuda": "CUDA (NVIDIA 顯示卡)",
        "device_cpu": "CPU (純處理器)",
        
        # Mirrors
        "mirror_hf": "Hugging Face 官方 (推薦)",
        "mirror_hf_mirror": "Hugging Face 鏡像",
        "mirror_modelscope": "ModelScope",
        
        # Inference params
        "params_header": "推論進階參數設定",
        "param_cfg": "引導係數 (CFG):",
        "param_cfg_tip": "越小音質越好但描述度越弱，建議 2.0-3.0",
        "param_steps": "去噪步數 (Steps):",
        "param_steps_tip": "步數越多生成越精細但耗時增加，一般設為 10-15 即可",
        "param_speed": "語速設定 (Speed):",
        "param_speed_tip": "調整生成語音速度 (0.5x 低速 - 2.0x 快速，預設 1.0x)",
        "param_norm": "文字標準化 (將數字/符號轉文字)",
        "param_denoise": "參考音訊自動降噪",
        "param_seed": "使用固定隨機種子 (固定音色)",
        
        # Actions
        "btn_run": "🚀 開始語音合成",
        "btn_batch": "📋 批次合成 (逐行)",
        "btn_play": "▶ 播放生成音訊",
        "btn_stop_play": "⏹ 停止播放",
        "btn_open_folder": "📂 開啟輸出資料夾",
        "btn_browse_short": "瀏覽...",
        
        # Dialogs / Common
        "dialog_warning": "警告",
        "dialog_error": "錯誤",
        "dialog_info": "資訊",
        "dialog_confirm": "確認",
        "dialog_no_model": "本機模型路徑未檢測到 config.json，可能尚未下載模型！\n請至「系統設定」分頁下載模型，或將本機模型路徑指向正確的資料夾。",
        "dialog_invalid_ref": "請提供有效的參考音檔路徑！",
        "dialog_need_prompt": "極限複製模式必須提供參考音訊的逐字稿！",
        "dialog_invalid_seed": "隨機種子必須是整數！",
        "dialog_empty_text": "文字框中沒有任何內容可供合成！",
        "dialog_dl_confirm": "偵測到本機缺少下列模型檔案，即將開始下載：\n{files}\n\n是否確認開始下載？",
        "dialog_dl_complete": "所有缺失模型檔案已成功部署至:\n{dir}",
        "dialog_all_files_exist": "本機已存在所有完整模型檔案，無須重新下載！",
        "dialog_batch_done": "批次合成完成！\n\n✅ 成功: {success} 句\n❌ 失敗: {failed} 句\n⏱️ 總耗時: {elapsed} 秒\n\n輸出資料夾:\n{out_dir}",
        "dialog_single_done": "語音合成成功！耗時: {elapsed} 秒\n\n儲存路徑:\n{filepath}",
        "status_stopping_dl": "正在停止...",
        "status_dl_stopped": "下載已中斷",
        "status_dl_failed": "下載失敗",
        "status_dl_complete": "下載完成",
        "status_preparing_infer": "🔄 正在背景準備推論作業，請稍候...",
    }
}

class SoundPlayer:
    @staticmethod
    def play_audio(filepath, stop_event, on_finished):
        global IS_PLAYING
        try:
            data, fs = sf.read(filepath)
            IS_PLAYING = True
            
            # Play using sounddevice
            sd.play(data, fs)
            
            # Wait until playback finishes or stop event is set
            chunk_size = int(fs * 0.1) # Check every 100ms
            position = 0
            while position < len(data) and not stop_event.is_set():
                time.sleep(0.1)
                position += chunk_size
            
            sd.stop()
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            IS_PLAYING = False
            on_finished()

class UILogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def log(self, msg):
        try:
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, f"{time.strftime('%H:%M:%S')} | {msg}\n")
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        except Exception as e:
            print(f"Log write failed: {e}")
        
        # Safe console print to prevent cp950 crashes
        try:
            enc = sys.stdout.encoding or 'utf-8'
            safe_msg = msg.encode(enc, errors='replace').decode(enc)
            print(safe_msg)
        except Exception:
            try:
                print(msg.encode('ascii', errors='ignore').decode('ascii'))
            except Exception:
                pass

class App(ctk.CTk):
    def __init__(self, default_lang=None):
        super().__init__()

        # Paths
        if getattr(sys, 'frozen', False):
            self.script_dir = os.path.dirname(sys.executable)
            self.resource_dir = sys._MEIPASS
        else:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
            self.resource_dir = os.path.dirname(os.path.abspath(__file__))

        self.config_path = os.path.join(self.script_dir, "app_config.json")

        # Determine initial language
        self.current_lang = "en"
        cli_lang = self._get_cli_language()
        if cli_lang:
            self.current_lang = cli_lang
        elif default_lang:
            self.current_lang = default_lang
        else:
            saved_lang = self._load_saved_language()
            if saved_lang:
                self.current_lang = saved_lang

        # Set appearance
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Configure Fonts dynamically based on language
        self._init_fonts()

        # Window settings
        self.title(self.t("app_title"))
        self.geometry("1120x760")
        self.minsize(980, 700)
        
        # Taskbar ID on Windows
        try:
            import ctypes
            myappid = 'studio0808.voxcpm.app.v20260619'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

        # App Icon
        try:
            icon_path = os.path.join(self.resource_dir, "app.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
                self.iconbitmap(default=icon_path)
                self.wm_iconbitmap(icon_path)
        except Exception as e:
            print(f"Warning: Could not load application icon app.ico. Error: {e}")
        
        # Center Window
        ws = self.winfo_screenwidth()
        hs = self.winfo_screenheight()
        x = int((ws/2) - (1120/2))
        y = int((hs/2) - (760/2) - 40)
        self.geometry(f"1120x760+{x}+{y}")
        self.after(100, lambda: self.state('zoomed'))

        # Shared Variables
        self.model_dir_var = ctk.StringVar(value=os.path.join(self.script_dir, "models", "VoxCPM2"))
        self.output_dir_var = ctk.StringVar(value=os.path.join(self.script_dir, "outputs"))
        self.device_var = ctk.StringVar(value=self.t("device_auto"))
        self.downloader_source_var = ctk.StringVar(value=self.t("mirror_hf"))
        self.download_cancelled_event = threading.Event()
        self.is_downloading = False
        
        # Inference variables dictionaries (independent per view)
        self.cfg_scale_vars = {
            "design": ctk.DoubleVar(value=2.0),
            "clone": ctk.DoubleVar(value=2.0),
            "ultimate": ctk.DoubleVar(value=2.0)
        }
        self.timesteps_vars = {
            "design": ctk.IntVar(value=10),
            "clone": ctk.IntVar(value=10),
            "ultimate": ctk.IntVar(value=10)
        }
        self.speed_rate_vars = {
            "design": ctk.DoubleVar(value=1.0),
            "clone": ctk.DoubleVar(value=1.0),
            "ultimate": ctk.DoubleVar(value=1.0)
        }
        self.chk_norm_vars = {
            "design": ctk.BooleanVar(value=False),
            "clone": ctk.BooleanVar(value=False),
            "ultimate": ctk.BooleanVar(value=False)
        }
        self.chk_denoise_vars = {
            "design": ctk.BooleanVar(value=False),
            "clone": ctk.BooleanVar(value=False),
            "ultimate": ctk.BooleanVar(value=False)
        }
        self.chk_seed_vars = {
            "design": ctk.BooleanVar(value=False),
            "clone": ctk.BooleanVar(value=False),
            "ultimate": ctk.BooleanVar(value=False)
        }
        self.seed_vars = {
            "design": ctk.StringVar(value=""),
            "clone": ctk.StringVar(value=""),
            "ultimate": ctk.StringVar(value="")
        }
        
        # Audio playback state
        self.last_generated_audio = None
        self.play_buttons = {}
        self.run_buttons = {}
        
        # Recording state variables
        self.is_recording = False
        self.recorded_frames = []
        self.recording_stream = None
        self.record_prompt_text = self.t("record_prompt_text")
        self.recorded_filepath = None
        self.recorded_play_thread = None
        self.recorded_play_stop_event = threading.Event()
        self.is_playing_recorded = False
        self.recording_device_var = ctk.StringVar()
        self.vis_buffer = np.zeros(4000)
        
        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#181A1F")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1) # Spacer between nav buttons and lang switcher

        # Navigation Buttons
        self.nav_buttons = []
        self.nav_images = {}
        icons_dir = os.path.join(self.resource_dir, "assets", "icons")
        self.nav_items_keys = [
            ("home", "nav_home", "#1E88E5", "home.png"),
            ("record", "nav_record", "#E91E63", "realtime_vc.png"),
            ("design", "nav_design", "#9C27B0", "tts.png"),
            ("clone", "nav_clone", "#4CAF50", "clone.png"),
            ("ultimate", "nav_ultimate", "#FF9800", "audio.png"),
            ("settings", "nav_settings", "#607D8B", "toolbox.png")
        ]

        for i, (name, label_key, color, icon_file) in enumerate(self.nav_items_keys, start=0):
            img = None
            icon_path = os.path.join(icons_dir, icon_file)
            if os.path.exists(icon_path):
                img = ctk.CTkImage(light_image=Image.open(icon_path), size=(22, 22))
                self.nav_images[name] = img

            btn = ctk.CTkButton(
                self.sidebar_frame, 
                text=self.t(label_key), 
                font=self.font_bold, 
                image=img,
                compound="left",
                height=45,
                border_spacing=10,
                fg_color="transparent", 
                text_color=("gray20", "gray85"),
                hover_color=("#2A2D35"),
                anchor="w",
                command=lambda n=name: self.select_view(n)
            )
            pady_val = (16, 2) if i == 0 else 2
            btn.grid(row=i, column=0, padx=10, pady=pady_val, sticky="ew")
            self.nav_buttons.append((name, btn, color, label_key))

        # Language Selector at bottom of Sidebar
        self.lbl_lang = ctk.CTkLabel(self.sidebar_frame, text=self.t("lang_label"), font=self.font_bold, anchor="w", text_color="#64B5F6")
        self.lbl_lang.grid(row=7, column=0, padx=15, pady=(8, 2), sticky="w")

        lang_display_map = {
            "en": "English",
            "km": "ភាសាខ្មែរ (Khmer)",
            "zh": "繁體中文 (Traditional Chinese)"
        }
        self.lang_menu_var = ctk.StringVar(value=lang_display_map.get(self.current_lang, "English"))
        self.lang_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["English", "ភាសាខ្មែរ (Khmer)", "繁體中文 (Traditional Chinese)"],
            variable=self.lang_menu_var,
            command=self._on_language_menu_select,
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#2196F3",
            button_hover_color="#1976D2"
        )
        self.lang_menu.grid(row=8, column=0, padx=10, pady=(0, 16), sticky="ew")

        # 2. Main Content Area
        self.content_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=(15, 5))
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # 3. Persistent Bottom Log Frame
        self.log_frame = ctk.CTkFrame(self, height=85, corner_radius=8, fg_color="#111216")
        self.log_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 15))
        self.log_frame.grid_columnconfigure(0, weight=1)
        self.log_frame.grid_rowconfigure(1, weight=1)
        
        self.progress_bar = ctk.CTkProgressBar(self.log_frame, mode="indeterminate", height=6, fg_color="#1A1D24", progress_color="#E91E63")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=15, pady=(5, 2))
        self.progress_bar.set(0)
        
        self.log_box = ctk.CTkTextbox(self.log_frame, font=self.font_mono, fg_color="#000000", text_color="#FFFFFF", state="disabled")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(2, 6))
        
        self.logger = UILogger(self.log_box)

        # Dynamic widgets tracker for updating UI texts on language switch
        self.ui_registry = {}

        # Initialize Views
        self.views = {}
        self.create_views()
        
        # Select Default View
        self.current_view_name = "home"
        self.select_view("home")
        
        # Check GPU status
        self.check_gpu()

    # --- Language & Font Utilities ---
    def t(self, key, **kwargs):
        """Translate key into the active language, with fallback to English."""
        lang_dict = I18N.get(self.current_lang, I18N["en"])
        template = lang_dict.get(key, I18N["en"].get(key, key))
        if kwargs:
            try:
                return template.format(**kwargs)
            except Exception:
                return template
        return template

    def _init_fonts(self):
        """Set native font families matching current language."""
        if self.current_lang == "km":
            family = "Leelawadee UI"
            title_sz, hdr_sz, ui_sz = 19, 14, 12
        elif self.current_lang == "zh":
            family = "Microsoft JhengHei UI"
            title_sz, hdr_sz, ui_sz = 20, 15, 13
        else:
            family = "Segoe UI"
            title_sz, hdr_sz, ui_sz = 20, 15, 13

        self.font_title = (family, title_sz, "bold")
        self.font_header = (family, hdr_sz, "bold")
        self.font_ui = (family, ui_sz)
        self.font_bold = (family, ui_sz, "bold")
        self.font_mono = (family, ui_sz - 1)

    def _get_cli_language(self):
        """Parse --lang or -l command line argument."""
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--lang", "-l", choices=["en", "km", "zh", "zh-tw", "zh-TW"], default=None)
            args, _ = parser.parse_known_args()
            if args.lang:
                l = args.lang.lower()
                return "zh" if "zh" in l else l
        except Exception:
            pass
        return None

    def _load_saved_language(self):
        """Load user preference from app_config.json."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return cfg.get("language")
            except Exception:
                pass
        return None

    def _save_language(self, lang_code):
        """Save user language preference to app_config.json."""
        try:
            cfg = {}
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    cfg = {}
            cfg["language"] = lang_code
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save language config: {e}")

    def _on_language_menu_select(self, choice):
        """Handle selection from sidebar language dropdown."""
        if "Khmer" in choice or "ខ្មែរ" in choice:
            new_lang = "km"
        elif "Chinese" in choice or "繁體" in choice:
            new_lang = "zh"
        else:
            new_lang = "en"
        self.set_language(new_lang)

    def set_language(self, lang_code):
        """Dynamically switch language and refresh all UI elements."""
        if lang_code not in I18N:
            lang_code = "en"
        self.current_lang = lang_code
        self._save_language(lang_code)
        self._init_fonts()

        # Update window title
        self.title(self.t("app_title"))

        # Update sidebar navigation
        for name, btn, color, label_key in self.nav_buttons:
            btn.configure(text=self.t(label_key), font=self.font_bold)
        self.lbl_lang.configure(text=self.t("lang_label"), font=self.font_bold)

        # Update registered labels and buttons
        for key, widget in self.ui_registry.items():
            try:
                if isinstance(widget, ctk.CTkLabel):
                    widget.configure(text=self.t(key))
                elif isinstance(widget, ctk.CTkButton):
                    widget.configure(text=self.t(key))
                elif isinstance(widget, ctk.CTkCheckBox):
                    widget.configure(text=self.t(key))
            except Exception:
                pass

        # Update device options menu
        curr_dev = self.device_var.get()
        dev_values = [self.t("device_auto"), self.t("device_cuda"), self.t("device_cpu")]
        self.menu_device.configure(values=dev_values)
        if "CUDA" in curr_dev:
            self.device_var.set(self.t("device_cuda"))
        elif "CPU" in curr_dev:
            self.device_var.set(self.t("device_cpu"))
        else:
            self.device_var.set(self.t("device_auto"))

        # Update download mirrors
        curr_mir = self.downloader_source_var.get()
        mirror_values = [self.t("mirror_hf"), self.t("mirror_hf_mirror"), self.t("mirror_modelscope")]
        self.menu_download_mirror.configure(values=mirror_values)
        if "ModelScope" in curr_mir:
            self.downloader_source_var.set(self.t("mirror_modelscope"))
        elif "Mirror" in curr_mir or "鏡像" in curr_mir or "ម៉ាស៊ីនឆ្លុះ" in curr_mir:
            self.downloader_source_var.set(self.t("mirror_hf_mirror"))
        else:
            self.downloader_source_var.set(self.t("mirror_hf"))

        # Update prompt reading text for Live Recording
        self.record_prompt_text = self.t("record_prompt_text")
        self.lbl_prompt_text.configure(text=f"「{self.record_prompt_text}」")

        # Update Voice Presets dropdown labels
        self._refresh_voice_preset_menus()

        # Re-apply nav button highlight
        self.select_view(self.current_view_name)
        self.logger.log(f"🌐 Interface language updated to: {self.current_lang.upper()}")

    def _refresh_voice_preset_menus(self):
        """Update preset menu options and titles based on language."""
        roles = [
            ("role_grandfather", "老爺爺"),
            ("role_grandmother", "老奶奶"),
            ("role_young_male", "年輕男生"),
            ("role_young_female", "年輕女生"),
            ("role_boy", "小男孩"),
            ("role_girl", "小女孩"),
        ]
        role_labels = [self.t(k) for k, _ in roles]

        # Khmer presets menu
        if "khmer" in self.preset_menus:
            self.preset_menus["khmer"].configure(values=role_labels)
            self.preset_menus["khmer"].set(self.t("preset_menu_km"))

        # US English presets menu
        if "us_english" in self.preset_menus:
            self.preset_menus["us_english"].configure(values=role_labels)
            self.preset_menus["us_english"].set(self.t("preset_menu_en"))

        # Taiwan Mandarin presets menu
        if "tw_mandarin" in self.preset_menus:
            self.preset_menus["tw_mandarin"].configure(values=role_labels)
            self.preset_menus["tw_mandarin"].set(self.t("preset_menu_zh"))

        # Japanese presets menu
        if "japanese" in self.preset_menus:
            self.preset_menus["japanese"].configure(values=role_labels)
            self.preset_menus["japanese"].set(self.t("preset_menu_ja"))

        # Korean presets menu
        if "korean" in self.preset_menus:
            self.preset_menus["korean"].configure(values=role_labels)
            self.preset_menus["korean"].set(self.t("preset_menu_ko"))

    def check_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.logger.log(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
                self.logger.log(f"CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
            else:
                self.logger.log("⚠️ No CUDA GPU detected. Running in CPU mode (slower).")
        except Exception as e:
            self.logger.log(f"GPU detection error: {e}")
        self.logger.log(f"📢 {self.t('home_notice')}")

    def create_views(self):
        # ==========================================
        # 1. Home View
        # ==========================================
        view_home = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["home"] = view_home
        
        title_row = ctk.CTkFrame(view_home, fg_color="transparent")
        title_row.pack(fill="x", pady=(0, 5))
        
        lbl_title = ctk.CTkLabel(title_row, text=self.t("home_title"), font=self.font_title, anchor="w")
        lbl_title.pack(side="left")
        self.ui_registry["home_title"] = lbl_title
        
        home_scroll = ctk.CTkFrame(view_home, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        home_scroll.pack(fill="both", expand=True, pady=(5, 5))
        
        intro_label = ctk.CTkLabel(
            home_scroll, 
            text=self.t("home_intro"), 
            font=self.font_bold, 
            text_color="gray85",
            anchor="w",
            justify="left"
        )
        intro_label.pack(fill="x", padx=15, pady=(10, 5))
        self.ui_registry["home_intro"] = intro_label
        
        # Notice card
        first_time_notice = ctk.CTkFrame(home_scroll, fg_color="#1F222B", border_width=1, border_color="#2196F3")
        first_time_notice.pack(fill="x", padx=15, pady=4, ipady=2)
        
        lbl_notice = ctk.CTkLabel(
            first_time_notice,
            text=self.t("home_notice"),
            font=self.font_bold,
            text_color="#64B5F6",
            justify="left",
            anchor="w",
            wraplength=820
        )
        lbl_notice.pack(fill="x", padx=15, pady=6)
        self.ui_registry["home_notice"] = lbl_notice
        
        # Highlights card
        highlight_card = ctk.CTkFrame(home_scroll, fg_color="#1F222B", border_width=1, border_color="#2D313E")
        highlight_card.pack(fill="x", padx=15, pady=4, ipady=2)
        
        header_hl = ctk.CTkFrame(highlight_card, fg_color="transparent")
        header_hl.pack(fill="x", padx=15, pady=(5, 2))
        
        icon_path_bulb = os.path.join(self.resource_dir, "assets", "icons", "bulb.png")
        if os.path.exists(icon_path_bulb):
            img_bulb = ctk.CTkImage(light_image=Image.open(icon_path_bulb), size=(20, 20))
            lbl_icon_hl = ctk.CTkLabel(header_hl, image=img_bulb, text="")
            lbl_icon_hl.pack(side="left", padx=(0, 8))
            
        lbl_title_hl = ctk.CTkLabel(header_hl, text=self.t("home_hl_title"), font=self.font_header, text_color="#2196F3")
        lbl_title_hl.pack(side="left")
        self.ui_registry["home_hl_title"] = lbl_title_hl
        
        lbl_content_hl = ctk.CTkLabel(
            highlight_card, 
            text=self.t("home_hl_content"), 
            font=self.font_ui, 
            text_color="gray90", 
            justify="left", 
            anchor="w"
        )
        lbl_content_hl.pack(fill="x", padx=25, pady=(2, 6))
        self.ui_registry["home_hl_content"] = lbl_content_hl
        
        # Hardware card
        hw_card = ctk.CTkFrame(home_scroll, fg_color="#1F222B", border_width=1, border_color="#2D313E")
        hw_card.pack(fill="x", padx=15, pady=4, ipady=2)
        
        header_hw = ctk.CTkFrame(hw_card, fg_color="transparent")
        header_hw.pack(fill="x", padx=15, pady=(5, 2))
        
        icon_path_hammer = os.path.join(self.resource_dir, "assets", "icons", "toolbox.png")
        if os.path.exists(icon_path_hammer):
            img_hammer = ctk.CTkImage(light_image=Image.open(icon_path_hammer), size=(20, 20))
            lbl_icon_hw = ctk.CTkLabel(header_hw, image=img_hammer, text="")
            lbl_icon_hw.pack(side="left", padx=(0, 8))
            
        lbl_title_hw = ctk.CTkLabel(header_hw, text=self.t("home_hw_title"), font=self.font_header, text_color="#4CAF50")
        lbl_title_hw.pack(side="left")
        self.ui_registry["home_hw_title"] = lbl_title_hw
        
        lbl_content_hw = ctk.CTkLabel(
            hw_card, 
            text=self.t("home_hw_content"), 
            font=self.font_ui, 
            text_color="gray90", 
            justify="left", 
            anchor="w",
            wraplength=820
        )
        lbl_content_hw.pack(fill="x", padx=25, pady=(2, 6))
        self.ui_registry["home_hw_content"] = lbl_content_hw
        
        # License card
        license_card = ctk.CTkFrame(home_scroll, fg_color="#1F222B", border_width=1, border_color="#2D313E")
        license_card.pack(fill="x", padx=15, pady=4, ipady=2)
        
        header_lic = ctk.CTkFrame(license_card, fg_color="transparent")
        header_lic.pack(fill="x", padx=15, pady=(5, 2))
        
        icon_path_warning = os.path.join(self.resource_dir, "assets", "icons", "warning.png")
        if os.path.exists(icon_path_warning):
            img_warning = ctk.CTkImage(light_image=Image.open(icon_path_warning), size=(20, 20))
            lbl_icon_lic = ctk.CTkLabel(header_lic, image=img_warning, text="")
            lbl_icon_lic.pack(side="left", padx=(0, 8))
            
        lbl_title_lic = ctk.CTkLabel(header_lic, text=self.t("home_lic_title"), font=self.font_header, text_color="#FF5252")
        lbl_title_lic.pack(side="left")
        self.ui_registry["home_lic_title"] = lbl_title_lic
        
        lbl_content_lic = ctk.CTkLabel(
            license_card, 
            text=self.t("home_lic_content"), 
            font=self.font_ui, 
            text_color="gray90", 
            justify="left", 
            anchor="w",
            wraplength=820
        )
        lbl_content_lic.pack(fill="x", padx=25, pady=(5, 10))
        self.ui_registry["home_lic_content"] = lbl_content_lic
        
        # ==========================================
        # 2. Voice Design View
        # ==========================================
        view_design = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["design"] = view_design
        
        lbl_design_title = ctk.CTkLabel(view_design, text=self.t("design_title"), font=self.font_title, anchor="w")
        lbl_design_title.pack(fill="x", pady=(0, 5))
        self.ui_registry["design_title"] = lbl_design_title
        
        lbl_design_desc = ctk.CTkLabel(view_design, text=self.t("design_desc"), font=self.font_ui, text_color="gray", anchor="w", justify="left")
        lbl_design_desc.pack(fill="x", pady=(0, 8))
        self.ui_registry["design_desc"] = lbl_design_desc
        
        design_input_container = ctk.CTkFrame(view_design, fg_color="transparent")
        design_input_container.pack(fill="x", pady=4)
        design_input_container.grid_columnconfigure(0, weight=1)
        design_input_container.grid_columnconfigure(1, weight=0)
        
        left_side = ctk.CTkFrame(design_input_container, fg_color="transparent")
        left_side.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        lbl_target_design = ctk.CTkLabel(left_side, text=self.t("design_input_lbl"), font=self.font_bold, anchor="w")
        lbl_target_design.pack(fill="x")
        self.ui_registry["design_input_lbl"] = lbl_target_design
        
        self.txt_design = ctk.CTkTextbox(left_side, height=180, font=self.font_ui, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        if self.current_lang == "km":
            initial_design_text = "(A young woman, Cambodian accent, gentle and sweet voice) សួស្តី! នេះគឺជាសំឡេងដែលបង្កើតឡើងដោយការរចនាសំឡេង AI។"
        elif self.current_lang == "zh":
            initial_design_text = "(A young woman, gentle and sweet voice) 哈囉！這是使用語音設計無中生有出來的聲音。"
        else:
            initial_design_text = "(A young woman, gentle and sweet voice) Hello! This is a voice generated from scratch using voice design."
        self.txt_design.insert("0.0", initial_design_text)
        self.txt_design.pack(fill="both", expand=True, pady=(2, 0))
        
        right_side = ctk.CTkFrame(design_input_container, fg_color="#181A1F", border_width=1, border_color="#2A2D35", width=270)
        right_side.grid(row=0, column=1, sticky="nsew")
        right_side.grid_propagate(False)
        
        lbl_presets_h = ctk.CTkLabel(right_side, text=self.t("design_presets_lbl"), font=self.font_bold, text_color="#9C27B0", anchor="w")
        lbl_presets_h.pack(pady=(6, 2), padx=10, fill="x")
        self.ui_registry["design_presets_lbl"] = lbl_presets_h
        
        self.preset_menus = {}
        
        # 1. Khmer Voice Presets (Top for easy access)
        self.preset_menus["khmer"] = ctk.CTkOptionMenu(
            right_side,
            values=[self.t("role_grandfather"), self.t("role_grandmother"), self.t("role_young_male"), self.t("role_young_female"), self.t("role_boy"), self.t("role_girl")],
            command=lambda r: self._apply_voice_preset("khmer", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#E91E63",
            button_hover_color="#C2185B"
        )
        self.preset_menus["khmer"].set(self.t("preset_menu_km"))
        self.preset_menus["khmer"].pack(pady=2, padx=12, fill="x")

        # 2. US English Presets
        self.preset_menus["us_english"] = ctk.CTkOptionMenu(
            right_side,
            values=[self.t("role_grandfather"), self.t("role_grandmother"), self.t("role_young_male"), self.t("role_young_female"), self.t("role_boy"), self.t("role_girl")],
            command=lambda r: self._apply_voice_preset("us_english", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#2196F3",
            button_hover_color="#1976D2"
        )
        self.preset_menus["us_english"].set(self.t("preset_menu_en"))
        self.preset_menus["us_english"].pack(pady=2, padx=12, fill="x")

        # 3. Taiwan Mandarin Presets
        self.preset_menus["tw_mandarin"] = ctk.CTkOptionMenu(
            right_side,
            values=[self.t("role_grandfather"), self.t("role_grandmother"), self.t("role_young_male"), self.t("role_young_female"), self.t("role_boy"), self.t("role_girl")],
            command=lambda r: self._apply_voice_preset("tw_mandarin", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#9C27B0",
            button_hover_color="#7B1FA2"
        )
        self.preset_menus["tw_mandarin"].set(self.t("preset_menu_zh"))
        self.preset_menus["tw_mandarin"].pack(pady=2, padx=12, fill="x")

        # 4. Japanese Presets
        self.preset_menus["japanese"] = ctk.CTkOptionMenu(
            right_side,
            values=[self.t("role_grandfather"), self.t("role_grandmother"), self.t("role_young_male"), self.t("role_young_female"), self.t("role_boy"), self.t("role_girl")],
            command=lambda r: self._apply_voice_preset("japanese", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#FF9800",
            button_hover_color="#F57C00"
        )
        self.preset_menus["japanese"].set(self.t("preset_menu_ja"))
        self.preset_menus["japanese"].pack(pady=2, padx=12, fill="x")

        # 5. Korean Presets
        self.preset_menus["korean"] = ctk.CTkOptionMenu(
            right_side,
            values=[self.t("role_grandfather"), self.t("role_grandmother"), self.t("role_young_male"), self.t("role_young_female"), self.t("role_boy"), self.t("role_girl")],
            command=lambda r: self._apply_voice_preset("korean", r),
            font=self.font_ui,
            dropdown_font=self.font_ui,
            fg_color="#2E3545",
            button_color="#607D8B",
            button_hover_color="#455A64"
        )
        self.preset_menus["korean"].set(self.t("preset_menu_ko"))
        self.preset_menus["korean"].pack(pady=(2, 6), padx=12, fill="x")
        
        self.create_inference_params_ui(view_design, "design")
        self.create_action_buttons_ui(view_design, "design")

        # ==========================================
        # 3. Voice Clone View
        # ==========================================
        view_clone = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["clone"] = view_clone
        
        lbl_clone_title = ctk.CTkLabel(view_clone, text=self.t("clone_title"), font=self.font_title, anchor="w")
        lbl_clone_title.pack(fill="x", pady=(0, 5))
        self.ui_registry["clone_title"] = lbl_clone_title
        
        lbl_clone_desc = ctk.CTkLabel(view_clone, text=self.t("clone_desc"), font=self.font_ui, text_color="gray", anchor="w")
        lbl_clone_desc.pack(fill="x", pady=(0, 8))
        self.ui_registry["clone_desc"] = lbl_clone_desc
        
        row_file = ctk.CTkFrame(view_clone, fg_color="transparent")
        row_file.pack(fill="x", pady=3)
        lbl_clone_ref = ctk.CTkLabel(row_file, text=self.t("clone_ref_lbl"), font=self.font_bold, width=200, anchor="w")
        lbl_clone_ref.pack(side="left")
        self.ui_registry["clone_ref_lbl"] = lbl_clone_ref
        
        self.entry_clone_ref = ctk.CTkEntry(row_file, font=self.font_ui, placeholder_text=self.t("clone_ref_ph"), fg_color="#181A1F")
        self.entry_clone_ref.pack(side="left", fill="x", expand=True, padx=5)
        
        btn_browse_clone = ctk.CTkButton(row_file, text=self.t("btn_browse_short"), font=self.font_ui, width=80, command=self.browse_clone_ref)
        btn_browse_clone.pack(side="left")
        self.ui_registry["btn_browse_short_clone"] = btn_browse_clone
        
        lbl_clone_target = ctk.CTkLabel(view_clone, text=self.t("clone_target_lbl"), font=self.font_bold, anchor="w")
        lbl_clone_target.pack(fill="x", pady=(6, 0))
        self.ui_registry["clone_target_lbl"] = lbl_clone_target
        
        self.txt_clone = ctk.CTkTextbox(view_clone, height=130, font=self.font_ui, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        self.txt_clone.insert("0.0", self.t("clone_default_text"))
        self.txt_clone.pack(fill="x", pady=4)
        
        self.create_inference_params_ui(view_clone, "clone")
        self.create_action_buttons_ui(view_clone, "clone")

        # ==========================================
        # 4. Ultimate Clone View
        # ==========================================
        view_ultimate = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["ultimate"] = view_ultimate
        
        lbl_ult_title = ctk.CTkLabel(view_ultimate, text=self.t("ult_title"), font=self.font_title, anchor="w")
        lbl_ult_title.pack(fill="x", pady=(0, 5))
        self.ui_registry["ult_title"] = lbl_ult_title
        
        lbl_ult_desc = ctk.CTkLabel(view_ultimate, text=self.t("ult_desc"), font=self.font_ui, text_color="gray", anchor="w")
        lbl_ult_desc.pack(fill="x", pady=(0, 8))
        self.ui_registry["ult_desc"] = lbl_ult_desc
        
        row_file2 = ctk.CTkFrame(view_ultimate, fg_color="transparent")
        row_file2.pack(fill="x", pady=3)
        lbl_ult_ref = ctk.CTkLabel(row_file2, text=self.t("ult_ref_lbl"), font=self.font_bold, width=200, anchor="w")
        lbl_ult_ref.pack(side="left")
        self.ui_registry["ult_ref_lbl"] = lbl_ult_ref
        
        self.entry_ult_ref = ctk.CTkEntry(row_file2, font=self.font_ui, placeholder_text=self.t("ult_ref_ph"), fg_color="#181A1F")
        self.entry_ult_ref.pack(side="left", fill="x", expand=True, padx=5)
        
        btn_browse_ult = ctk.CTkButton(row_file2, text=self.t("btn_browse_short"), font=self.font_ui, width=80, command=self.browse_ult_ref)
        btn_browse_ult.pack(side="left")
        self.ui_registry["btn_browse_short_ult"] = btn_browse_ult
        
        row_prompt = ctk.CTkFrame(view_ultimate, fg_color="transparent")
        row_prompt.pack(fill="x", pady=3)
        lbl_ult_prompt = ctk.CTkLabel(row_prompt, text=self.t("ult_prompt_lbl"), font=self.font_bold, width=200, anchor="w")
        lbl_ult_prompt.pack(side="left")
        self.ui_registry["ult_prompt_lbl"] = lbl_ult_prompt
        
        self.entry_ult_prompt = ctk.CTkEntry(row_prompt, font=self.font_ui, placeholder_text=self.t("ult_prompt_ph"), fg_color="#181A1F")
        self.entry_ult_prompt.pack(side="left", fill="x", expand=True, padx=5)
        
        lbl_ult_target = ctk.CTkLabel(view_ultimate, text=self.t("ult_target_lbl"), font=self.font_bold, anchor="w")
        lbl_ult_target.pack(fill="x", pady=(6, 0))
        self.ui_registry["ult_target_lbl"] = lbl_ult_target
        
        self.txt_ultimate = ctk.CTkTextbox(view_ultimate, height=120, font=self.font_ui, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        self.txt_ultimate.insert("0.0", self.t("ult_default_text"))
        self.txt_ultimate.pack(fill="x", pady=4)
        
        self.create_inference_params_ui(view_ultimate, "ultimate")
        self.create_action_buttons_ui(view_ultimate, "ultimate")

        # ==========================================
        # 5. Live Recording Studio View
        # ==========================================
        view_record = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["record"] = view_record
        
        lbl_record_title = ctk.CTkLabel(view_record, text=self.t("record_title"), font=self.font_title, anchor="w")
        lbl_record_title.pack(fill="x", pady=(0, 5))
        self.ui_registry["record_title"] = lbl_record_title
        
        lbl_record_desc = ctk.CTkLabel(view_record, text=self.t("record_desc"), font=self.font_ui, text_color="gray", anchor="w")
        lbl_record_desc.pack(fill="x", pady=(0, 8))
        self.ui_registry["record_desc"] = lbl_record_desc
        
        row_rec_dev = ctk.CTkFrame(view_record, fg_color="transparent")
        row_rec_dev.pack(fill="x", pady=5)
        lbl_rec_dev = ctk.CTkLabel(row_rec_dev, text=self.t("record_dev_lbl"), font=self.font_bold, width=220, anchor="w")
        lbl_rec_dev.pack(side="left")
        self.ui_registry["record_dev_lbl"] = lbl_rec_dev
        
        self.menu_devices = ctk.CTkOptionMenu(
            row_rec_dev,
            variable=self.recording_device_var,
            values=["Default Microphone"],
            font=self.font_ui,
            dropdown_font=self.font_ui,
            width=320
        )
        self.menu_devices.pack(side="left", padx=5)
        
        btn_refresh_devices = ctk.CTkButton(
            row_rec_dev,
            text=self.t("record_btn_refresh"),
            font=self.font_bold,
            width=100,
            command=self.populate_recording_devices
        )
        btn_refresh_devices.pack(side="left", padx=5)
        self.ui_registry["record_btn_refresh"] = btn_refresh_devices
        
        # Reading Prompt Card
        prompt_card = ctk.CTkFrame(view_record, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        prompt_card.pack(fill="x", pady=10, ipady=10)
        
        lbl_reading_hdr = ctk.CTkLabel(prompt_card, text=self.t("record_card_title"), font=self.font_bold, text_color="#E91E63")
        lbl_reading_hdr.pack(pady=(10, 5), padx=15, anchor="w")
        self.ui_registry["record_card_title"] = lbl_reading_hdr
        
        self.lbl_prompt_text = ctk.CTkLabel(
            prompt_card,
            text=f"「{self.record_prompt_text}」",
            font=self.font_header,
            text_color="#FFFFFF",
            justify="center",
            wraplength=820
        )
        self.lbl_prompt_text.pack(pady=15, padx=20, fill="x")
        
        lbl_rec_tip = ctk.CTkLabel(
            prompt_card,
            text=self.t("record_tip"),
            font=self.font_ui,
            text_color="gray"
        )
        lbl_rec_tip.pack(pady=(0, 10))
        self.ui_registry["record_tip"] = lbl_rec_tip
        
        # Controls
        control_frame = ctk.CTkFrame(view_record, fg_color="transparent")
        control_frame.pack(fill="x", pady=10)
        
        self.btn_record = ctk.CTkButton(
            control_frame,
            text=self.t("record_btn_start"),
            font=self.font_bold,
            height=45,
            width=180,
            fg_color="#F44336",
            hover_color="#D32F2F",
            command=self.toggle_recording
        )
        self.btn_record.pack(side="left", padx=(0, 10))
        self.ui_registry["record_btn_start"] = self.btn_record
        
        # Real-time voice waveform canvas
        self.wave_canvas = tk.Canvas(control_frame, width=250, height=45, bg="#111216", highlightthickness=1, highlightbackground="#2A2D35")
        self.wave_canvas.pack(side="left", padx=10)
        self.wave_canvas.create_line(0, 22, 250, 22, fill="#555555", tags="wave", width=1)
        
        self.lbl_record_status = ctk.CTkLabel(
            control_frame,
            text=self.t("record_status_ready"),
            font=self.font_bold,
            text_color="gray85"
        )
        self.lbl_record_status.pack(side="left", padx=10)
        
        # Playback and Apply frame
        self.recorded_actions_frame = ctk.CTkFrame(view_record, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        self.recorded_actions_frame.pack(fill="x", pady=10, ipady=5)
        
        lbl_post_actions = ctk.CTkLabel(self.recorded_actions_frame, text=self.t("record_post_lbl"), font=self.font_bold)
        lbl_post_actions.pack(anchor="w", padx=15, pady=(8, 4))
        self.ui_registry["record_post_lbl"] = lbl_post_actions
        
        actions_row = ctk.CTkFrame(self.recorded_actions_frame, fg_color="transparent")
        actions_row.pack(fill="x", padx=10, pady=5)
        
        self.btn_play_recorded = ctk.CTkButton(
            actions_row,
            text=self.t("record_btn_play"),
            font=self.font_bold,
            height=40,
            width=150,
            fg_color="#4CAF50",
            hover_color="#388E3C",
            state="disabled",
            command=self.toggle_recorded_playback
        )
        self.btn_play_recorded.pack(side="left", padx=5)
        self.ui_registry["record_btn_play"] = self.btn_play_recorded
        
        self.btn_apply_clone = ctk.CTkButton(
            actions_row,
            text=self.t("record_btn_apply_clone"),
            font=self.font_bold,
            height=40,
            width=200,
            fg_color="#2196F3",
            hover_color="#1976D2",
            state="disabled",
            command=self.use_recorded_for_clone
        )
        self.btn_apply_clone.pack(side="left", padx=5)
        self.ui_registry["record_btn_apply_clone"] = self.btn_apply_clone
        
        self.btn_apply_ultimate = ctk.CTkButton(
            actions_row,
            text=self.t("record_btn_apply_ult"),
            font=self.font_bold,
            height=40,
            width=200,
            fg_color="#FF9800",
            hover_color="#F57C00",
            state="disabled",
            command=self.use_recorded_for_ultimate
        )
        self.btn_apply_ultimate.pack(side="left", padx=5)
        self.ui_registry["record_btn_apply_ult"] = self.btn_apply_ultimate
        
        self.populate_recording_devices()

        # ==========================================
        # 6. Settings View
        # ==========================================
        view_settings = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.views["settings"] = view_settings
        
        lbl_settings_h = ctk.CTkLabel(view_settings, text=self.t("settings_title"), font=self.font_title, anchor="w")
        lbl_settings_h.pack(fill="x", pady=(0, 15))
        self.ui_registry["settings_title"] = lbl_settings_h
        
        # Model path
        row_path = ctk.CTkFrame(view_settings, fg_color="transparent")
        row_path.pack(fill="x", pady=8)
        lbl_model_path = ctk.CTkLabel(row_path, text=self.t("settings_model_path"), font=self.font_bold, width=170, anchor="w")
        lbl_model_path.pack(side="left")
        self.ui_registry["settings_model_path"] = lbl_model_path
        
        self.entry_model_dir = ctk.CTkEntry(row_path, textvariable=self.model_dir_var, font=self.font_ui, fg_color="#181A1F")
        self.entry_model_dir.pack(side="left", fill="x", expand=True, padx=5)
        
        btn_browse_model = ctk.CTkButton(row_path, text=self.t("settings_browse"), font=self.font_ui, width=130, command=self.browse_model_dir)
        btn_browse_model.pack(side="left")
        self.ui_registry["settings_browse_model"] = btn_browse_model

        # Output path
        row_out_path = ctk.CTkFrame(view_settings, fg_color="transparent")
        row_out_path.pack(fill="x", pady=8)
        lbl_out_path = ctk.CTkLabel(row_out_path, text=self.t("settings_out_path"), font=self.font_bold, width=170, anchor="w")
        lbl_out_path.pack(side="left")
        self.ui_registry["settings_out_path"] = lbl_out_path
        
        self.entry_output_dir = ctk.CTkEntry(row_out_path, textvariable=self.output_dir_var, font=self.font_ui, fg_color="#181A1F")
        self.entry_output_dir.pack(side="left", fill="x", expand=True, padx=5)
        
        btn_browse_out = ctk.CTkButton(row_out_path, text=self.t("settings_browse"), font=self.font_ui, width=130, command=self.browse_output_dir)
        btn_browse_out.pack(side="left")
        self.ui_registry["settings_browse_out"] = btn_browse_out

        # Device
        row_device = ctk.CTkFrame(view_settings, fg_color="transparent")
        row_device.pack(fill="x", pady=8)
        lbl_dev_setting = ctk.CTkLabel(row_device, text=self.t("settings_device_lbl"), font=self.font_bold, width=170, anchor="w")
        lbl_dev_setting.pack(side="left")
        self.ui_registry["settings_device_lbl"] = lbl_dev_setting
        
        self.menu_device = ctk.CTkOptionMenu(
            row_device,
            variable=self.device_var,
            values=[self.t("device_auto"), self.t("device_cuda"), self.t("device_cpu")],
            font=self.font_ui,
            dropdown_font=self.font_ui,
            width=240
        )
        self.menu_device.pack(side="left", padx=5)
        
        lbl_dev_tip = ctk.CTkLabel(row_device, text=self.t("settings_device_tip"), font=self.font_ui, text_color="gray")
        lbl_dev_tip.pack(side="left", padx=10)
        self.ui_registry["settings_device_tip"] = lbl_dev_tip

        # Separator line
        ctk.CTkFrame(view_settings, height=2, fg_color="#2A2D35").pack(fill="x", pady=15)

        # Download Panel
        lbl_dl_h = ctk.CTkLabel(view_settings, text=self.t("settings_dl_header"), font=self.font_header, anchor="w")
        lbl_dl_h.pack(fill="x", pady=(0, 5))
        self.ui_registry["settings_dl_header"] = lbl_dl_h
        
        lbl_dl_info = ctk.CTkLabel(view_settings, text=self.t("settings_dl_info"), font=self.font_ui, text_color="gray", justify="left", anchor="w")
        lbl_dl_info.pack(fill="x", pady=(0, 10))
        self.ui_registry["settings_dl_info"] = lbl_dl_info

        row_dl = ctk.CTkFrame(view_settings, fg_color="transparent")
        row_dl.pack(fill="x", pady=5)
        
        lbl_dl_src = ctk.CTkLabel(row_dl, text=self.t("settings_source_lbl"), font=self.font_ui)
        lbl_dl_src.pack(side="left", padx=(0, 10))
        self.ui_registry["settings_source_lbl"] = lbl_dl_src
        
        self.menu_download_mirror = ctk.CTkOptionMenu(
            row_dl, 
            variable=self.downloader_source_var, 
            values=[self.t("mirror_hf"), self.t("mirror_hf_mirror"), self.t("mirror_modelscope")], 
            font=self.font_ui, 
            dropdown_font=self.font_ui, 
            width=240
        )
        self.menu_download_mirror.pack(side="left")
        
        self.btn_download = ctk.CTkButton(row_dl, text=self.t("settings_btn_download"), font=self.font_bold, fg_color="#9C27B0", hover_color="#7B1FA2", width=220, command=self.start_model_download)
        self.btn_download.pack(side="left", padx=15)
        self.ui_registry["settings_btn_download"] = self.btn_download

        self.lbl_dl_status = ctk.CTkLabel(view_settings, text="", font=self.font_ui, text_color="gray85", anchor="w", justify="left")
        self.lbl_dl_status.pack(fill="x", pady=(5, 0))

    def create_inference_params_ui(self, parent_frame, func_type):
        param_frame = ctk.CTkFrame(parent_frame, fg_color="#181A1F", border_width=1, border_color="#2A2D35")
        param_frame.pack(fill="x", pady=6, ipady=2)
        
        lbl_h = ctk.CTkLabel(param_frame, text=self.t("params_header"), font=self.font_header)
        lbl_h.pack(anchor="w", padx=15, pady=(6, 2))
        self.ui_registry[f"params_header_{func_type}"] = lbl_h
        
        grid_container = ctk.CTkFrame(param_frame, fg_color="transparent")
        grid_container.pack(fill="x", padx=10, pady=(0, 4))
        grid_container.columnconfigure(0, weight=0)
        grid_container.columnconfigure(1, weight=1)
        
        left_col = ctk.CTkFrame(grid_container, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(5, 15))
        left_col.columnconfigure(2, weight=1)
        
        right_col = ctk.CTkFrame(grid_container, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(15, 5))
        right_col.columnconfigure(0, weight=1)

        # 1. CFG Scale
        lbl_cfg = ctk.CTkLabel(left_col, text=self.t("param_cfg"), font=self.font_bold)
        lbl_cfg.grid(row=0, column=0, padx=(0, 5), pady=3, sticky="e")
        self.ui_registry[f"param_cfg_{func_type}"] = lbl_cfg
        
        cfg_container = ctk.CTkFrame(left_col, fg_color="transparent")
        cfg_container.grid(row=0, column=1, padx=5, pady=3, sticky="w")
        
        lbl_cfg_val = ctk.CTkLabel(cfg_container, text=f"{self.cfg_scale_vars[func_type].get():.1f}", font=self.font_bold, width=35)
        def _update_cfg(v):
            val = round(v * 2) / 2
            lbl_cfg_val.configure(text=f"{val:.1f}")
            self.cfg_scale_vars[func_type].set(val)
        
        ctk.CTkSlider(cfg_container, from_=1.0, to=5.0, number_of_steps=8, variable=self.cfg_scale_vars[func_type], command=_update_cfg, width=150).pack(side="left")
        lbl_cfg_val.pack(side="left", padx=(5, 0))
        
        lbl_cfg_tip = ctk.CTkLabel(left_col, text=self.t("param_cfg_tip"), font=self.font_ui, text_color="gray")
        lbl_cfg_tip.grid(row=0, column=2, padx=10, pady=3, sticky="w")
        self.ui_registry[f"param_cfg_tip_{func_type}"] = lbl_cfg_tip
        
        # 2. Timesteps
        lbl_steps = ctk.CTkLabel(left_col, text=self.t("param_steps"), font=self.font_bold)
        lbl_steps.grid(row=1, column=0, padx=(0, 5), pady=3, sticky="e")
        self.ui_registry[f"param_steps_{func_type}"] = lbl_steps
        
        step_container = ctk.CTkFrame(left_col, fg_color="transparent")
        step_container.grid(row=1, column=1, padx=5, pady=3, sticky="w")
        
        lbl_step_val = ctk.CTkLabel(step_container, text=f"{self.timesteps_vars[func_type].get()}", font=self.font_bold, width=35)
        def _update_step(v):
            val = int(v)
            lbl_step_val.configure(text=f"{val}")
            self.timesteps_vars[func_type].set(val)
            
        ctk.CTkSlider(step_container, from_=5, to=30, number_of_steps=25, variable=self.timesteps_vars[func_type], command=_update_step, width=150).pack(side="left")
        lbl_step_val.pack(side="left", padx=(5, 0))
        
        lbl_steps_tip = ctk.CTkLabel(left_col, text=self.t("param_steps_tip"), font=self.font_ui, text_color="gray")
        lbl_steps_tip.grid(row=1, column=2, padx=10, pady=3, sticky="w")
        self.ui_registry[f"param_steps_tip_{func_type}"] = lbl_steps_tip
        
        # 3. Speed Rate
        lbl_spd = ctk.CTkLabel(left_col, text=self.t("param_speed"), font=self.font_bold)
        lbl_spd.grid(row=2, column=0, padx=(0, 5), pady=3, sticky="e")
        self.ui_registry[f"param_speed_{func_type}"] = lbl_spd
        
        speed_container = ctk.CTkFrame(left_col, fg_color="transparent")
        speed_container.grid(row=2, column=1, padx=5, pady=3, sticky="w")
        
        lbl_speed_val = ctk.CTkLabel(speed_container, text=f"{self.speed_rate_vars[func_type].get():.1f}x", font=self.font_bold, width=35)
        def _update_speed(v):
            val = round(v, 1)
            lbl_speed_val.configure(text=f"{val:.1f}x")
            self.speed_rate_vars[func_type].set(val)
            
        ctk.CTkSlider(speed_container, from_=0.5, to=2.0, number_of_steps=15, variable=self.speed_rate_vars[func_type], command=_update_speed, width=150).pack(side="left")
        lbl_speed_val.pack(side="left", padx=(5, 0))
        
        lbl_spd_tip = ctk.CTkLabel(left_col, text=self.t("param_speed_tip"), font=self.font_ui, text_color="gray")
        lbl_spd_tip.grid(row=2, column=2, padx=10, pady=3, sticky="w")
        self.ui_registry[f"param_speed_tip_{func_type}"] = lbl_spd_tip

        # Right Column Parameters
        chk_norm = ctk.CTkCheckBox(right_col, text=self.t("param_norm"), font=self.font_bold, variable=self.chk_norm_vars[func_type], checkbox_width=16, checkbox_height=16)
        chk_norm.grid(row=0, column=0, padx=(15, 0), pady=(12, 6), sticky="w")
        self.ui_registry[f"param_norm_{func_type}"] = chk_norm
        
        chk_denoise = ctk.CTkCheckBox(right_col, text=self.t("param_denoise"), font=self.font_bold, variable=self.chk_denoise_vars[func_type], checkbox_width=16, checkbox_height=16)
        chk_denoise.grid(row=1, column=0, padx=(15, 0), pady=6, sticky="w")
        self.ui_registry[f"param_denoise_{func_type}"] = chk_denoise
        
        seed_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        seed_frame.grid(row=2, column=0, padx=(15, 0), pady=(6, 12), sticky="w")
        
        chk_seed = ctk.CTkCheckBox(
            seed_frame, 
            text=self.t("param_seed"), 
            font=self.font_bold, 
            variable=self.chk_seed_vars[func_type], 
            checkbox_width=16, 
            checkbox_height=16,
            command=lambda f=func_type: self._toggle_seed_entry(f)
        )
        chk_seed.pack(side="left")
        self.ui_registry[f"param_seed_{func_type}"] = chk_seed
        
        if not hasattr(self, "entry_seeds"):
            self.entry_seeds = {}
            
        entry_seed = ctk.CTkEntry(
            seed_frame, 
            textvariable=self.seed_vars[func_type], 
            font=self.font_mono, 
            width=90, 
            height=24,
            fg_color="#181A1F"
        )
        entry_seed.pack(side="left", padx=(8, 0))
        self.entry_seeds[func_type] = entry_seed
        
        if not self.chk_seed_vars[func_type].get():
            entry_seed.configure(state="disabled")

    def create_action_buttons_ui(self, parent_frame, func_type):
        action_row = ctk.CTkFrame(parent_frame, fg_color="transparent")
        action_row.pack(fill="x", pady=6)
        
        btn_run = ctk.CTkButton(
            action_row, 
            text=self.t("btn_run"), 
            font=self.font_bold, 
            height=40, 
            width=190, 
            fg_color="#E91E63", 
            hover_color="#C2185B",
            command=lambda f=func_type: self.run_inference(f)
        )
        btn_run.pack(side="left", padx=(0, 10))
        self.run_buttons[func_type] = btn_run
        self.ui_registry[f"btn_run_{func_type}"] = btn_run
        
        if func_type == "design":
            btn_batch = ctk.CTkButton(
                action_row, 
                text=self.t("btn_batch"), 
                font=self.font_bold, 
                height=40, 
                width=190, 
                fg_color="#FF9800", 
                hover_color="#F57C00",
                command=self.run_batch_inference
            )
            btn_batch.pack(side="left", padx=(0, 10))
            self.ui_registry["btn_batch"] = btn_batch
        
        play_frame = ctk.CTkFrame(action_row, fg_color="transparent")
        play_frame.pack(side="left", padx=10)
        
        btn_play = ctk.CTkButton(
            play_frame, 
            text=self.t("btn_play"), 
            font=self.font_bold, 
            height=40, 
            width=160, 
            fg_color="#4CAF50", 
            hover_color="#388E3C",
            state="disabled",
            command=self.toggle_playback
        )
        btn_play.pack(side="left", padx=5)
        self.play_buttons[func_type] = btn_play
        self.ui_registry[f"btn_play_{func_type}"] = btn_play

        btn_open = ctk.CTkButton(
            action_row, 
            text=self.t("btn_open_folder"), 
            font=self.font_bold, 
            height=40, 
            width=160, 
            fg_color="#607D8B", 
            hover_color="#455A64",
            command=self.open_output_folder
        )
        btn_open.pack(side="left", padx=10)
        self.ui_registry[f"btn_open_folder_{func_type}"] = btn_open

    # Navigation & View selector
    def select_view(self, name):
        self.current_view_name = name
        for btn_name, btn, color, label_key in self.nav_buttons:
            if btn_name == name:
                btn.configure(fg_color=color, text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color=("gray20", "gray85"))

        for view in self.views.values():
            view.pack_forget()

        if name in self.views:
            self.views[name].pack(fill="both", expand=True)

    # Browsing helpers
    def browse_clone_ref(self):
        f = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3;*.flac;*.m4a")])
        if f:
            self.entry_clone_ref.delete(0, tk.END)
            self.entry_clone_ref.insert(0, f)

    def browse_ult_ref(self):
        f = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3;*.flac;*.m4a")])
        if f:
            self.entry_ult_ref.delete(0, tk.END)
            self.entry_ult_ref.insert(0, f)

    def browse_model_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.model_dir_var.set(d)

    def browse_output_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir_var.set(d)

    # Voice preset handler
    def _apply_voice_preset(self, lang, role_label):
        presets = {
            "khmer": {
                "desc": {
                    "gf": "An elderly man, Cambodian Khmer accent, deep and warm voice",
                    "gm": "An elderly woman, Cambodian Khmer accent, kind and soft voice",
                    "ym": "A young man, Cambodian Khmer accent, clear and energetic voice",
                    "yf": "A young woman, Cambodian Khmer accent, gentle and sweet voice",
                    "b": "A young boy, Cambodian Khmer accent, cute voice",
                    "g": "A young girl, Cambodian Khmer accent, lovely voice"
                },
                "sentence": "សួស្តី! នេះគឺជាសំឡេងដែលបង្កើតឡើងដោយការរចនាសំឡេង AI។"
            },
            "us_english": {
                "desc": {
                    "gf": "An old man, American accent, deep and warm voice",
                    "gm": "An old woman, American accent, kind and gentle voice",
                    "ym": "A young man, American accent, clear and energetic voice",
                    "yf": "A young woman, American accent, gentle and sweet voice",
                    "b": "A young boy, American accent, cute voice",
                    "g": "A young girl, American accent, lovely voice"
                },
                "sentence": "Hello! This is a voice generated from scratch using voice design."
            },
            "tw_mandarin": {
                "desc": {
                    "gf": "An elderly man, Taiwanese Mandarin accent, warm and gentle voice",
                    "gm": "An elderly woman, Taiwanese Mandarin accent, kind and soft voice",
                    "ym": "A young man, Taiwanese Mandarin accent, clear and energetic voice",
                    "yf": "A young woman, Taiwanese Mandarin accent, gentle and sweet voice",
                    "b": "A young boy, Taiwanese Mandarin accent, cute voice",
                    "g": "A young girl, Taiwanese Mandarin accent, lovely and sweet voice"
                },
                "sentence": "哈囉！這是使用語音設計無中生有出來的聲音。"
            },
            "japanese": {
                "desc": {
                    "gf": "An old man, Japanese accent, warm and gentle voice",
                    "gm": "An old woman, Japanese accent, kind and soft voice",
                    "ym": "A young man, Japanese accent, clear and energetic voice",
                    "yf": "A young woman, Japanese accent, gentle and sweet voice",
                    "b": "A young boy, Japanese accent, cute voice",
                    "g": "A young girl, Japanese accent, lovely voice"
                },
                "sentence": "こんにちは！これは音声設計で新しく生成された声です。"
            },
            "korean": {
                "desc": {
                    "gf": "An old man, Korean accent, warm and gentle voice",
                    "gm": "An old woman, Korean accent, kind and soft voice",
                    "ym": "A young man, Korean accent, clear and energetic voice",
                    "yf": "A young woman, Korean accent, gentle and sweet voice",
                    "b": "A young boy, Korean accent, cute voice",
                    "g": "A young girl, Korean accent, lovely voice"
                },
                "sentence": "안녕하세요! 이것은 음성 디자인으로 새로 생성된 목소리입니다."
            }
        }
        
        # Map label to role key
        role_key = "yf"
        r_str = role_label.lower()
        if "grandf" in r_str or "លោកតា" in r_str or "老爺爺" in r_str:
            role_key = "gf"
        elif "grandm" in r_str or "លោកយាយ" in r_str or "老奶奶" in r_str:
            role_key = "gm"
        elif "female" in r_str or "ស្ត្រី" in r_str or "女生" in r_str or "girl" in r_str:
            role_key = "g" if ("young girl" in r_str or "ក្មេងស្រី" in r_str or "小女孩" in r_str) else "yf"
        elif "male" in r_str or "man" in r_str or "បុរស" in r_str or "男生" in r_str or "boy" in r_str:
            role_key = "b" if ("boy" in r_str or "ក្មេងប្រុស" in r_str or "小男孩" in r_str) else "ym"
            
        desc = presets.get(lang, {}).get("desc", {}).get(role_key, "")
        sentence = presets.get(lang, {}).get("sentence", "")
        new_line = f"({desc}) {sentence}"
        
        current_content = self.txt_design.get("0.0", "end-1c")
        if not current_content.strip():
            self.txt_design.delete("0.0", "end")
            self.txt_design.insert("0.0", new_line)
        else:
            if current_content.endswith("\n"):
                self.txt_design.insert("end", new_line)
            else:
                self.txt_design.insert("end", "\n" + new_line)
                
        # Reset dropdown titles
        self._refresh_voice_preset_menus()

    def _toggle_seed_entry(self, func_type):
        if self.chk_seed_vars[func_type].get():
            self.entry_seeds[func_type].configure(state="normal")
        else:
            self.entry_seeds[func_type].configure(state="disabled")

    def set_global_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    # Inference logic
    def run_inference(self, func_type):
        model_dir = self.model_dir_var.get()
        if not os.path.exists(os.path.join(model_dir, "config.json")) and not os.path.exists(os.path.join(model_dir, "pyproject.toml")):
            messagebox.showwarning(self.t("dialog_warning"), self.t("dialog_no_model"))
            return

        self.logger.log(self.t("status_preparing_infer"))
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        
        if func_type == "design":
            text = self.txt_design.get("0.0", "end").strip()
            args = {
                "text": text, 
                "cfg_value": self.cfg_scale_vars["design"].get(), 
                "inference_timesteps": self.timesteps_vars["design"].get(),
                "normalize": self.chk_norm_vars["design"].get(),
                "denoise": self.chk_denoise_vars["design"].get()
            }
        elif func_type == "clone":
            text = self.txt_clone.get("0.0", "end").strip()
            ref_audio = self.entry_clone_ref.get().strip()
            if not ref_audio or not os.path.exists(ref_audio):
                messagebox.showerror(self.t("dialog_error"), self.t("dialog_invalid_ref"))
                self._stop_progress_bar()
                return
            args = {
                "text": text, 
                "reference_wav_path": ref_audio, 
                "cfg_value": self.cfg_scale_vars["clone"].get(), 
                "inference_timesteps": self.timesteps_vars["clone"].get(),
                "normalize": self.chk_norm_vars["clone"].get(),
                "denoise": self.chk_denoise_vars["clone"].get()
            }
        elif func_type == "ultimate":
            text = self.txt_ultimate.get("0.0", "end").strip()
            ref_audio = self.entry_ult_ref.get().strip()
            prompt_text = self.entry_ult_prompt.get().strip()
            if not ref_audio or not os.path.exists(ref_audio):
                messagebox.showerror(self.t("dialog_error"), self.t("dialog_invalid_ref"))
                self._stop_progress_bar()
                return
            if not prompt_text:
                messagebox.showerror(self.t("dialog_error"), self.t("dialog_need_prompt"))
                self._stop_progress_bar()
                return
            args = {
                "text": text, 
                "prompt_wav_path": ref_audio, 
                "prompt_text": prompt_text, 
                "reference_wav_path": ref_audio,
                "cfg_value": self.cfg_scale_vars["ultimate"].get(),
                "inference_timesteps": self.timesteps_vars["ultimate"].get(),
                "normalize": self.chk_norm_vars["ultimate"].get(),
                "denoise": self.chk_denoise_vars["ultimate"].get()
            }
        
        if self.chk_seed_vars[func_type].get():
            try:
                seed_val = int(self.seed_vars[func_type].get().strip())
                args["seed"] = seed_val
            except ValueError:
                messagebox.showerror(self.t("dialog_error"), self.t("dialog_invalid_seed"))
                self._stop_progress_bar()
                return
                
        speed_rate = self.speed_rate_vars[func_type].get()
        threading.Thread(target=self._infer_thread_run, args=(model_dir, args, speed_rate), daemon=True).start()

    def run_batch_inference(self):
        raw_text = self.txt_design.get("0.0", "end").strip()
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        
        if not lines:
            messagebox.showwarning(self.t("dialog_warning"), self.t("dialog_empty_text"))
            return
        
        if len(lines) < 2:
            self.run_inference("design")
            return
        
        model_dir = self.model_dir_var.get()
        if not os.path.exists(os.path.join(model_dir, "config.json")) and not os.path.exists(os.path.join(model_dir, "pyproject.toml")):
            messagebox.showwarning(self.t("dialog_warning"), self.t("dialog_no_model"))
            return
        
        self.logger.log(f"📋 Batch synthesis started: {len(lines)} lines to process.")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        
        cfg = self.cfg_scale_vars["design"].get()
        steps = self.timesteps_vars["design"].get()
        speed_rate = self.speed_rate_vars["design"].get()
        norm = self.chk_norm_vars["design"].get()
        denoise_audio = self.chk_denoise_vars["design"].get()
        
        seed_val = None
        if self.chk_seed_vars["design"].get():
            try:
                seed_val = int(self.seed_vars["design"].get().strip())
            except ValueError:
                messagebox.showwarning(self.t("dialog_warning"), self.t("dialog_invalid_seed"))
                self._stop_progress_bar()
                return
                
        threading.Thread(target=self._batch_infer_thread_run, args=(model_dir, lines, cfg, steps, speed_rate, norm, denoise_audio, seed_val), daemon=True).start()

    def _batch_infer_thread_run(self, model_dir, lines, cfg, steps, speed_rate, norm, denoise_audio, seed_val):
        global VOXCPM_MODEL
        total = len(lines)
        success_count = 0
        fail_count = 0
        generated_files = []
        batch_start = time.time()
        
        try:
            device_choice = self.device_var.get()
            if "CUDA" in device_choice:
                device_val = "cuda"
            elif "CPU" in device_choice:
                device_val = "cpu"
            else:
                device_val = "auto"

            # Check reload
            if VOXCPM_MODEL is not None:
                current_device = str(VOXCPM_MODEL.tts_model.device).lower()
                from voxcpm.model.utils import resolve_runtime_device
                try:
                    target_device = resolve_runtime_device(device_val).lower()
                except Exception:
                    target_device = "cpu"
                if target_device not in current_device and current_device not in target_device:
                    if not (current_device.startswith("cuda") and target_device.startswith("cuda")):
                        self.logger.log(f"🔄 Reloading model for device: {device_choice}...")
                        import gc
                        import torch
                        VOXCPM_MODEL = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

            if VOXCPM_MODEL is None:
                self.logger.log(f"🔄 Initializing VoxCPM2 model on {device_choice}...")
                from voxcpm import VoxCPM
                should_optimize = not getattr(sys, 'frozen', False)
                if device_val == "cpu":
                    should_optimize = False
                elif device_val == "auto":
                    import torch
                    if not torch.cuda.is_available():
                        should_optimize = False
                VOXCPM_MODEL = VoxCPM.from_pretrained(
                    model_dir,
                    load_denoiser=False,
                    optimize=should_optimize,
                    device=device_val
                )
                self.logger.log("✅ Model loaded successfully into memory!")
            
            for i, line in enumerate(lines, 1):
                self.logger.log(f"")
                self.logger.log(f"━━━ [{i}/{total}] ━━━")
                self.logger.log(f"📝 {line[:80]}{'...' if len(line) > 80 else ''}")
                
                if seed_val is not None:
                    self.set_global_seed(seed_val)
                    self.logger.log(f"🔒 Seed ID: {seed_val}")
                else:
                    line_seed = random.randint(0, 10000000)
                    self.set_global_seed(line_seed)
                    self.logger.log(f"🎲 Seed ID: {line_seed}")
                
                try:
                    start_time = time.time()
                    wav = VOXCPM_MODEL.generate(
                        text=line,
                        cfg_value=cfg,
                        inference_timesteps=steps,
                        normalize=norm,
                        denoise=denoise_audio,
                    )
                    
                    if speed_rate != 1.0:
                        self.logger.log(f"⏳ Adjusting speed to {speed_rate}x...")
                        import librosa
                        wav = librosa.effects.time_stretch(wav, rate=speed_rate)
                    
                    os.makedirs(self.output_dir_var.get(), exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"voxcpm_batch{i:02d}_{timestamp}.wav"
                    filepath = os.path.join(self.output_dir_var.get(), filename)
                    sf.write(filepath, wav, VOXCPM_MODEL.tts_model.sample_rate)
                    
                    elapsed = time.time() - start_time
                    self.logger.log(f"✅ Line {i} generated in {elapsed:.2f}s → {filename}")
                    
                    self.last_generated_audio = filepath
                    generated_files.append(filepath)
                    success_count += 1
                    
                except Exception as e:
                    err_msg = str(e)
                    self.logger.log(f"❌ Line {i} failed: {err_msg}")
                    fail_count += 1
            
            if generated_files:
                for btn in self.play_buttons.values():
                    btn.configure(state="normal", text=self.t("btn_play"))
            
            total_elapsed = f"{time.time() - batch_start:.2f}"
            self.logger.log(f"🏁 Batch complete! Success: {success_count} / Failed: {fail_count} / Time: {total_elapsed}s")
            
            summary = self.t("dialog_batch_done", success=success_count, failed=fail_count, elapsed=total_elapsed, out_dir=self.output_dir_var.get())
            self.after(0, lambda: messagebox.showinfo(self.t("dialog_info"), summary))
            
        except Exception as e:
            self._handle_inference_error(e, prefix="Batch synthesis error")
        finally:
            self.after(0, self._stop_progress_bar)

    def _infer_thread_run(self, model_dir, args, speed_rate):
        global VOXCPM_MODEL
        try:
            device_choice = self.device_var.get()
            if "CUDA" in device_choice:
                device_val = "cuda"
            elif "CPU" in device_choice:
                device_val = "cpu"
            else:
                device_val = "auto"

            if VOXCPM_MODEL is not None:
                current_device = str(VOXCPM_MODEL.tts_model.device).lower()
                from voxcpm.model.utils import resolve_runtime_device
                try:
                    target_device = resolve_runtime_device(device_val).lower()
                except Exception:
                    target_device = "cpu"
                if target_device not in current_device and current_device not in target_device:
                    if not (current_device.startswith("cuda") and target_device.startswith("cuda")):
                        self.logger.log(f"🔄 Reloading model for device: {device_choice}...")
                        import gc
                        import torch
                        VOXCPM_MODEL = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

            if VOXCPM_MODEL is None:
                self.logger.log(f"🔄 Initializing VoxCPM2 model on {device_choice}...")
                from voxcpm import VoxCPM
                should_optimize = not getattr(sys, 'frozen', False)
                if device_val == "cpu":
                    should_optimize = False
                elif device_val == "auto":
                    import torch
                    if not torch.cuda.is_available():
                        should_optimize = False
                VOXCPM_MODEL = VoxCPM.from_pretrained(
                    model_dir,
                    load_denoiser=False,
                    optimize=should_optimize,
                    device=device_val
                )
                self.logger.log("✅ Model loaded successfully into memory!")

            self.logger.log("🚀 Starting speech synthesis...")
            seed_val = args.pop("seed", None)
            if seed_val is not None:
                self.set_global_seed(seed_val)
                self.logger.log(f"🔒 Seed ID: {seed_val}")
            else:
                random_seed = random.randint(0, 10000000)
                self.set_global_seed(random_seed)
                self.logger.log(f"🎲 Seed ID: {random_seed}")

            start_time = time.time()
            wav = VOXCPM_MODEL.generate(**args)
            
            if speed_rate != 1.0:
                self.logger.log(f"⏳ Adjusting speed to {speed_rate}x...")
                import librosa
                wav = librosa.effects.time_stretch(wav, rate=speed_rate)
            
            os.makedirs(self.output_dir_var.get(), exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"voxcpm_{timestamp}.wav"
            filepath = os.path.join(self.output_dir_var.get(), filename)
            
            sf.write(filepath, wav, VOXCPM_MODEL.tts_model.sample_rate)
            elapsed = f"{time.time() - start_time:.2f}"
            self.logger.log(f"🎉 Speech generation successful! Time: {elapsed}s")
            self.logger.log(f"Saved: {filepath}")
            
            self.last_generated_audio = filepath
            for btn in self.play_buttons.values():
                btn.configure(state="normal", text=self.t("btn_play"))
            
            msg = self.t("dialog_single_done", elapsed=elapsed, filepath=filepath)
            self.after(0, lambda: messagebox.showinfo(self.t("dialog_info"), msg))
            
        except Exception as e:
            self._handle_inference_error(e, prefix="Synthesis failed")
        finally:
            self.after(0, self._stop_progress_bar)

    def _handle_inference_error(self, e, prefix="Synthesis failed"):
        err_msg = str(e)
        self.logger.log(f"❌ {prefix}: {err_msg}")
        if "1455" in err_msg or (isinstance(e, OSError) and getattr(e, 'winerror', None) == 1455):
            self.logger.log("⚠️ Windows virtual memory (pagefile) is insufficient!")
            self.logger.log("💡 Suggestion: Increase paging file size (16GB-32GB) or set Device to CPU in Settings.")
        import traceback
        self.logger.log(f"Traceback:\n{traceback.format_exc()}")

    def _stop_progress_bar(self):
        try:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(0)
        except Exception:
            pass

    # Playback functions
    def toggle_playback(self):
        global PLAYBACK_THREAD, PLAYBACK_STOP_EVENT, IS_PLAYING
        
        if not self.last_generated_audio or not os.path.exists(self.last_generated_audio):
            return

        if hasattr(self, 'is_playing_recorded') and self.is_playing_recorded:
            self.recorded_play_stop_event.set()
            sd.stop()
            self.btn_play_recorded.configure(text=self.t("record_btn_play"))
            self.is_playing_recorded = False
            time.sleep(0.1)

        if IS_PLAYING:
            PLAYBACK_STOP_EVENT.set()
            sd.stop()
            for btn in self.play_buttons.values():
                btn.configure(text=self.t("btn_play"))
            IS_PLAYING = False
        else:
            PLAYBACK_STOP_EVENT.clear()
            for btn in self.play_buttons.values():
                btn.configure(text=self.t("btn_stop_play"))
            
            def _on_finished():
                for btn in self.play_buttons.values():
                    btn.configure(text=self.t("btn_play"))
                
            PLAYBACK_THREAD = threading.Thread(
                target=SoundPlayer.play_audio, 
                args=(self.last_generated_audio, PLAYBACK_STOP_EVENT, _on_finished),
                daemon=True
            )
            PLAYBACK_THREAD.start()

    def toggle_recorded_playback(self):
        if not self.recorded_filepath or not os.path.exists(self.recorded_filepath):
            return
            
        global IS_PLAYING, PLAYBACK_STOP_EVENT
        if IS_PLAYING:
            PLAYBACK_STOP_EVENT.set()
            sd.stop()
            for btn in self.play_buttons.values():
                btn.configure(text=self.t("btn_play"))
            IS_PLAYING = False
            time.sleep(0.1)
            
        if self.is_playing_recorded:
            self.recorded_play_stop_event.set()
            sd.stop()
            self.btn_play_recorded.configure(text=self.t("record_btn_play"))
            self.is_playing_recorded = False
        else:
            self.recorded_play_stop_event.clear()
            self.btn_play_recorded.configure(text=self.t("btn_stop_play"))
            self.is_playing_recorded = True
            
            def _on_finished():
                self.btn_play_recorded.configure(text=self.t("record_btn_play"))
                self.is_playing_recorded = False
                
            self.recorded_play_thread = threading.Thread(
                target=SoundPlayer.play_audio,
                args=(self.recorded_filepath, self.recorded_play_stop_event, _on_finished),
                daemon=True
            )
            self.recorded_play_thread.start()

    def populate_recording_devices(self):
        try:
            devices = sd.query_devices()
            input_devices = []
            default_device_idx = sd.default.device[0]
            default_device_str = "Default Input Device"
            
            for idx, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    device_name = f"{idx}: {d['name']}"
                    input_devices.append(device_name)
                    if idx == default_device_idx:
                        default_device_str = device_name
            
            if not input_devices:
                self.menu_devices.configure(values=["No Microphone Available"])
                self.recording_device_var.set("No Microphone Available")
                self.btn_record.configure(state="disabled")
            else:
                self.menu_devices.configure(values=input_devices)
                if default_device_str in input_devices:
                    self.recording_device_var.set(default_device_str)
                else:
                    self.recording_device_var.set(input_devices[0])
                self.btn_record.configure(state="normal")
        except Exception as e:
            self.logger.log(f"⚠️ Microphone query error: {e}")
            self.menu_devices.configure(values=["Device Error"])
            self.recording_device_var.set("Device Error")
            self.btn_record.configure(state="disabled")

    def toggle_recording(self):
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            self.btn_record.configure(text=self.t("record_btn_start"), fg_color="#F44336", hover_color="#D32F2F")
            
            if self.recording_stream:
                try:
                    self.recording_stream.stop()
                    self.recording_stream.close()
                except Exception:
                    pass
                self.recording_stream = None
            
            try:
                self.wave_canvas.delete("wave")
                self.wave_canvas.create_line(0, 22, 250, 22, fill="#555555", tags="wave", width=1)
            except Exception:
                pass

            if self.recorded_frames:
                try:
                    audio_data = np.concatenate(self.recorded_frames, axis=0)
                    out_dir = self.output_dir_var.get()
                    os.makedirs(out_dir, exist_ok=True)
                    self.recorded_filepath = os.path.join(out_dir, "user_record_test.wav")
                    sf.write(self.recorded_filepath, audio_data, 44100)
                    
                    duration = len(audio_data) / 44100.0
                    self.lbl_record_status.configure(
                        text=self.t("record_status_done", duration=duration),
                        text_color="#4CAF50"
                    )
                    self.logger.log(f"🎤 Recorded {duration:.1f}s saved: {self.recorded_filepath}")
                    
                    self.btn_play_recorded.configure(state="normal")
                    self.btn_apply_clone.configure(state="normal")
                    self.btn_apply_ultimate.configure(state="normal")
                except Exception as e:
                    self.lbl_record_status.configure(text=f"❌ Error: {e}", text_color="#F44336")
            else:
                self.lbl_record_status.configure(text="⚠️ No audio recorded.", text_color="#FF9800")
        else:
            # Start recording
            device_str = self.recording_device_var.get()
            if "No Microphone" in device_str or "Device Error" in device_str:
                messagebox.showerror(self.t("dialog_error"), "No audio recording device available!")
                return
            
            try:
                device_idx = int(device_str.split(":")[0])
            except Exception:
                device_idx = None
                
            self.recorded_frames = []
            self.vis_buffer = np.zeros(4000)
            self.is_recording = True
            self.btn_record.configure(text=self.t("record_btn_stop"), fg_color="#607D8B", hover_color="#455A64")
            self.lbl_record_status.configure(text=self.t("record_status_recording", elapsed=0.0), text_color="#E91E63")
            
            self.btn_play_recorded.configure(state="disabled")
            self.btn_apply_clone.configure(state="disabled")
            self.btn_apply_ultimate.configure(state="disabled")
            
            def callback(indata, frames, time_info, status):
                if self.is_recording:
                    self.recorded_frames.append(indata.copy())
                    self.vis_buffer = np.roll(self.vis_buffer, -frames)
                    self.vis_buffer[-frames:] = indata[:, 0]
            
            try:
                self.recording_stream = sd.InputStream(
                    samplerate=44100,
                    channels=1,
                    device=device_idx,
                    callback=callback
                )
                self.recording_stream.start()
                start_time = time.time()
                self._update_recording_timer(start_time)
                self.logger.log("🎤 Recording started. Please read the reading text...")
            except Exception as e:
                self.is_recording = False
                self.btn_record.configure(text=self.t("record_btn_start"), fg_color="#F44336", hover_color="#D32F2F")
                self.lbl_record_status.configure(text=f"❌ Failed to open microphone: {e}", text_color="#F44336")

    def _update_recording_timer(self, start_time):
        if self.is_recording:
            elapsed = time.time() - start_time
            self.lbl_record_status.configure(text=self.t("record_status_recording", elapsed=elapsed))
            
            try:
                width = 250
                height = 45
                mid_y = height / 2
                points = []
                step = 4000 // width
                
                for i in range(width):
                    val = self.vis_buffer[i * step]
                    y = mid_y - (val * mid_y * 2.5)
                    y = max(2, min(height - 2, y))
                    points.append((i, y))
                    
                self.wave_canvas.delete("wave")
                for idx in range(len(points) - 1):
                    self.wave_canvas.create_line(
                        points[idx][0], points[idx][1],
                        points[idx+1][0], points[idx+1][1],
                        fill="#E91E63", tags="wave", width=2
                    )
            except Exception:
                pass
            
            if elapsed >= 15.0:
                self.logger.log("⏱️ Max recording time (15s) reached. Stopping automatically.")
                self.toggle_recording()
            else:
                self.after(50, lambda: self._update_recording_timer(start_time))

    def use_recorded_for_clone(self):
        if not self.recorded_filepath or not os.path.exists(self.recorded_filepath):
            messagebox.showwarning(self.t("dialog_warning"), "Please complete recording first!")
            return
        self.entry_clone_ref.delete(0, tk.END)
        self.entry_clone_ref.insert(0, self.recorded_filepath)
        self.select_view("clone")
        self.logger.log("🎤 Live recording applied to Voice Clone.")

    def use_recorded_for_ultimate(self):
        if not self.recorded_filepath or not os.path.exists(self.recorded_filepath):
            messagebox.showwarning(self.t("dialog_warning"), "Please complete recording first!")
            return
        self.entry_ult_ref.delete(0, tk.END)
        self.entry_ult_ref.insert(0, self.recorded_filepath)
        self.entry_ult_prompt.delete(0, tk.END)
        self.entry_ult_prompt.insert(0, self.record_prompt_text)
        self.select_view("ultimate")
        self.logger.log("🎤 Live recording and transcript applied to Ultimate Clone.")

    def open_output_folder(self):
        out_dir = self.output_dir_var.get()
        os.makedirs(out_dir, exist_ok=True)
        os.startfile(out_dir)

    # Downloader thread
    def start_model_download(self):
        if self.is_downloading:
            self.download_cancelled_event.set()
            self.btn_download.configure(text=self.t("status_stopping_dl"), state="disabled")
            self.logger.log("🛑 Download cancellation requested...")
            return

        model_dir = self.model_dir_var.get()
        os.makedirs(model_dir, exist_ok=True)
        
        REQUIRED_MODEL_FILES = {
            "config.json": 4000,
            "special_tokens_map.json": 1500,
            "tokenization_voxcpm2.py": 2500,
            "tokenizer.json": 3500000,
            "tokenizer_config.json": 4500,
            "audiovae.pth": 350000000,
            "model.safetensors": 4000000000
        }
        
        FILE_DISPLAY_SIZES = {
            "config.json": "4.2 KB",
            "special_tokens_map.json": "1.6 KB",
            "tokenization_voxcpm2.py": "2.8 KB",
            "tokenizer.json": "3.5 MB",
            "tokenizer_config.json": "4.9 KB",
            "audiovae.pth": "359.5 MB",
            "model.safetensors": "4.27 GB"
        }
        
        missing_files = []
        for filename, min_size in REQUIRED_MODEL_FILES.items():
            filepath = os.path.join(model_dir, filename)
            if not os.path.exists(filepath) or os.path.getsize(filepath) < min_size:
                missing_files.append(filename)
                
        if not missing_files:
            self.logger.log(f"✅ {self.t('dialog_all_files_exist')}")
            messagebox.showinfo(self.t("dialog_info"), self.t("dialog_all_files_exist"))
            return
            
        self.logger.log(f"🔍 Missing files detected: {', '.join(missing_files)}")
        files_str = "\n".join([f"• {f} ({FILE_DISPLAY_SIZES[f]})" for f in missing_files])
        prompt_msg = self.t("dialog_dl_confirm", files=files_str)
        if not messagebox.askyesno(self.t("dialog_confirm"), prompt_msg):
            self.logger.log("❌ Download cancelled by user.")
            return
            
        self.is_downloading = True
        self.download_cancelled_event.clear()
        
        self.btn_download.configure(text=self.t("settings_btn_stop"), fg_color="#F44336", hover_color="#D32F2F")
        self.lbl_dl_status.configure(text="Connecting to download server...")
        
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0.0)
        
        source = self.downloader_source_var.get()
        threading.Thread(target=self._download_thread_run, args=(source, model_dir, missing_files), daemon=True).start()

    def _download_thread_run(self, source, model_dir, files_to_download):
        if "ModelScope" in source:
            base_url = "https://modelscope.cn/api/v1/models/OpenBMB/VoxCPM2/repo?Revision=master&FilePath="
            self.logger.log("🔄 Download mirror: ModelScope")
        elif "Mirror" in source or "鏡像" in source or "ម៉ាស៊ីនឆ្លុះ" in source:
            base_url = "https://hf-mirror.com/openbmb/VoxCPM2/resolve/main/"
            self.logger.log("🔄 Download mirror: Hugging Face Mirror")
        else:
            base_url = "https://huggingface.co/openbmb/VoxCPM2/resolve/main/"
            self.logger.log("🔄 Download mirror: Hugging Face Official")
            
        success_all = True
        cancelled = False
        
        try:
            for idx, filename in enumerate(files_to_download, 1):
                if self.download_cancelled_event.is_set():
                    cancelled = True
                    break
                    
                url = base_url + filename
                filepath = os.path.join(model_dir, filename)
                temp_filepath = filepath + ".downloading"
                
                self.logger.log(f"📥 Downloading [{idx}/{len(files_to_download)}]: {filename}...")
                self._safe_update_ui(f"Downloading [{idx}/{len(files_to_download)}]: {filename}...", 0.0)
                
                success = self._download_single_file(url, temp_filepath, filename, idx, len(files_to_download))
                
                if self.download_cancelled_event.is_set():
                    cancelled = True
                    if os.path.exists(temp_filepath):
                        try: os.remove(temp_filepath)
                        except Exception: pass
                    break
                    
                if not success:
                    success_all = False
                    self.logger.log(f"❌ {filename} download failed.")
                    if os.path.exists(temp_filepath):
                        try: os.remove(temp_filepath)
                        except Exception: pass
                    break
                else:
                    if os.path.exists(filepath):
                        try: os.remove(filepath)
                        except Exception: pass
                    os.rename(temp_filepath, filepath)
                    self.logger.log(f"✅ {filename} download completed!")
                    
            if cancelled:
                self.logger.log("🛑 Download cancelled.")
                self._safe_update_ui(self.t("status_dl_stopped"), 0.0)
                self.after(0, lambda: messagebox.showwarning(self.t("dialog_warning"), self.t("status_dl_stopped")))
            elif not success_all:
                self.logger.log("❌ Download failed. Check connection or try another mirror.")
                self._safe_update_ui(self.t("status_dl_failed"), 0.0)
                self.after(0, lambda: messagebox.showerror(self.t("dialog_error"), self.t("status_dl_failed")))
            else:
                self.logger.log("🎉 All model files downloaded and verified!")
                self._safe_update_ui(self.t("status_dl_complete"), 1.0)
                self.after(0, lambda: messagebox.showinfo(self.t("dialog_info"), self.t("dialog_dl_complete", dir=model_dir)))
                
        except Exception as e:
            self.logger.log(f"❌ Download error: {e}")
            self._safe_update_ui("Error", 0.0)
            self.after(0, lambda: messagebox.showerror(self.t("dialog_error"), str(e)))
        finally:
            self.is_downloading = False
            self.btn_download.configure(
                text=self.t("settings_btn_download"), 
                fg_color="#9C27B0", 
                hover_color="#7B1FA2", 
                state="normal"
            )
            self.after(0, self._stop_progress_bar)

    def _download_single_file(self, url, temp_filepath, filename, idx, total_count):
        import urllib.request
        import urllib.error
        import socket
        
        max_retries = 5
        retry_delay = 3
        timeout = 25
        
        for attempt in range(1, max_retries + 1):
            if self.download_cancelled_event.is_set():
                return False
                
            bytes_so_far = 0
            if os.path.exists(temp_filepath):
                bytes_so_far = os.path.getsize(temp_filepath)
                
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            if bytes_so_far > 0:
                req.add_header('Range', f'bytes={bytes_so_far}-')
                
            os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)
            
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status_code = response.getcode()
                    if status_code == 206:
                        file_mode = 'ab'
                        content_range = response.info().get('Content-Range', '')
                        total_size = 0
                        if '/' in content_range:
                            try:
                                total_size = int(content_range.split('/')[-1])
                            except ValueError:
                                pass
                        if not total_size:
                            total_size = bytes_so_far + int(response.info().get('Content-Length', 0))
                    else:
                        bytes_so_far = 0
                        file_mode = 'wb'
                        total_size = int(response.info().get('Content-Length', 0))
                        
                    block_size = 256 * 1024
                    start_time = time.time()
                    last_ui_update = 0
                    session_bytes = 0
                    
                    with open(temp_filepath, file_mode) as f:
                        while True:
                            if self.download_cancelled_event.is_set():
                                return False
                            chunk = response.read(block_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            bytes_so_far += len(chunk)
                            session_bytes += len(chunk)
                            
                            current_time = time.time()
                            if current_time - last_ui_update > 0.15:
                                last_ui_update = current_time
                                elapsed = current_time - start_time
                                speed = (session_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                                
                                if total_size > 0:
                                    percent = float(bytes_so_far) / total_size * 100
                                    remaining = total_size - bytes_so_far
                                    speed_bytes = session_bytes / elapsed if elapsed > 0 else 0
                                    rem_sec = remaining / speed_bytes if speed_bytes > 0 else 0
                                    rem_m, rem_s = divmod(int(rem_sec), 60)
                                    status_text = f"[{idx}/{total_count}] {filename}: {percent:.1f}% | {bytes_so_far / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB | {speed:.2f} MB/s | ETA: {rem_m:02d}:{rem_s:02d}"
                                    progress_val = bytes_so_far / total_size
                                else:
                                    status_text = f"[{idx}/{total_count}] {filename}: {bytes_so_far / (1024*1024):.1f}MB | {speed:.2f} MB/s"
                                    progress_val = 0.5
                                self._safe_update_ui(status_text, progress_val)
                    return True
            except urllib.error.HTTPError as e:
                if e.code == 416:
                    if os.path.exists(temp_filepath):
                        try: os.remove(temp_filepath)
                        except Exception: pass
            except Exception as e:
                pass
                
            if attempt < max_retries:
                for _ in range(retry_delay * 10):
                    if self.download_cancelled_event.is_set():
                        return False
                    time.sleep(0.1)
        return False

    def _safe_update_ui(self, text, val):
        self.after(0, lambda: self._update_ui_elements(text, val))
        
    def _update_ui_elements(self, text, val):
        try:
            self.lbl_dl_status.configure(text=text)
            self.progress_bar.set(val)
        except Exception:
            pass

if __name__ == "__main__":
    app = App()
    app.mainloop()
