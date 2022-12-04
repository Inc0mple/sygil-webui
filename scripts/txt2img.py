# This file is part of sygil-webui (https://github.com/Sygil-Dev/sygil-webui/).

# Copyright 2022 Sygil-Dev team.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# base webui import and utils.
from sd_utils import *
from typing import Union

# streamlit imports
# from streamlit import StopException
#from streamlit.elements import image as STImage
import streamlit.components.v1 as components
# from streamlit.runtime.media_file_manager import media_file_manager
from streamlit.elements.image import image_to_url

# other imports
import uuid
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# streamlit components
from custom_components import sygil_suggestions

# Temp imports
# from bokeh.models.widgets import Button
# from bokeh.models import CustomJS
# from streamlit_bokeh_events import streamlit_bokeh_events
import platform
import speech_recognition as sr
import sys
# import vosk
import json
# from vosk import SetLogLevel
# SetLogLevel(-1)
import os
import numpy as np

from transformers import pipeline
p = platform.system()

with hc.HyLoader('Loading Stable Diffusion Model...', hc.Loaders.standard_loaders, index=[0]):
    load_models(use_LDSR=False, LDSR_model='model',
                use_GFPGAN=False, GFPGAN_model='model',
                use_RealESRGAN=True, RealESRGAN_model="RealESRGAN_x4plus",
                CustomModel_available=False, custom_model="Stable Diffusion v1.5")
with hc.HyLoader('Initialising Speech-To-Text Model...', hc.Loaders.standard_loaders, index=[0]):
# with st.spinner():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)
    prompt_extend_pipe = pipeline('text-generation', model='daspartho/prompt-extend', device=0)

    

# def invertExtendPrompt():
#     st.session_state.extendPrompt = not st.session_state.extendPrompt
    

# end of imports
# ---------------------------------------------------------------------------------------------------------------

sygil_suggestions.init()

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

#
# Dev mode (server)
# _component_func = components.declare_component(
#         "sd-gallery",
#         url="http://localhost:3001",
#     )

# Init Vuejs component
_component_func = components.declare_component(
    "sd-gallery", "./frontend/dists/sd-gallery/dist")


def sdGallery(images=[], key=None):
    component_value = _component_func(
        images=imgsToGallery(images), key=key, default="")
    return component_value


def imgsToGallery(images):
    urls = []
    for i in images:
        # random string for id
        random_id = str(uuid.uuid4())
        url = image_to_url(
            image=i,
            image_id=random_id,
            width=i.width,
            clamp=False,
            channels="RGB",
            output_format="PNG"
        )
        # image_io = BytesIO()
        # i.save(image_io, 'PNG')
        # width, height = i.size
        # image_id = "%s" % (str(images.index(i)))
        # (data, mimetype) = STImage._normalize_to_bytes(image_io.getvalue(), width, 'auto')
        # this_file = media_file_manager.add(data, mimetype, image_id)
        # img_str = this_file.url
        urls.append(url)

    return urls


class plugin_info():
    plugname = "txt2img"
    description = "Text to Image"
    isTab = True
    displayPriority = 1


@logger.catch(reraise=True)
def stable_horde(outpath, prompt, seed, sampler_name, save_grid, batch_size,
                 n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, GFPGAN_model,
                 use_RealESRGAN, realesrgan_model_name, use_LDSR,
                 LDSR_model_name, ddim_eta, normalize_prompt_weights,
                 save_individual_images, sort_samples, write_info_files,
                 jpg_sample, variant_amount, variant_seed, api_key,
                 nsfw=False, censor_nsfw=True):

    log = []

    log.append("Generating image with Stable Horde.")

    st.session_state["progress_bar_text"].code(
        '\n'.join(str(log)), language='')

    # start time after garbage collection (or before?)
    start_time = time.time()

    # We will use this date here later for the folder name, need to start_time if not need
    run_start_dt = datetime.datetime.now()

    mem_mon = MemUsageMonitor('MemMon')
    mem_mon.start()

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    params = {
        "sampler_name": "k_euler",
        "toggles": [1, 4],
        "cfg_scale": cfg_scale,
        "seed": str(seed),
        "width": width,
        "height": height,
        "seed_variation": variant_seed if variant_seed else 1,
        "steps": int(steps),
        "n": int(n_iter)
        # You can put extra params here if you wish
    }

    final_submit_dict = {
        "prompt": prompt,
        "params": params,
        "nsfw": nsfw,
        "censor_nsfw": censor_nsfw,
        "trusted_workers": True,
        "workers": []
    }
    log.append(final_submit_dict)

    headers = {"apikey": api_key}
    logger.debug(final_submit_dict)
    st.session_state["progress_bar_text"].code(
        '\n'.join(str(log)), language='')

    horde_url = "https://stablehorde.net"

    submit_req = requests.post(
        f'{horde_url}/api/v2/generate/async', json=final_submit_dict, headers=headers)
    if submit_req.ok:
        submit_results = submit_req.json()
        logger.debug(submit_results)

        log.append(submit_results)
        st.session_state["progress_bar_text"].code(
            '\n'.join(str(log)), language='')

        req_id = submit_results['id']
        is_done = False
        while not is_done:
            chk_req = requests.get(
                f'{horde_url}/api/v2/generate/check/{req_id}')
            if not chk_req.ok:
                logger.error(chk_req.text)
                return
            chk_results = chk_req.json()
            logger.info(chk_results)
            is_done = chk_results['done']
            time.sleep(1)
        retrieve_req = requests.get(
            f'{horde_url}/api/v2/generate/status/{req_id}')
        if not retrieve_req.ok:
            logger.error(retrieve_req.text)
            return
        results_json = retrieve_req.json()
        # logger.debug(results_json)
        results = results_json['generations']

        output_images = []
        comments = []
        prompt_matrix_parts = []

        if not st.session_state['defaults'].general.no_verify_input:
            try:
                check_prompt_length(prompt, comments)
            except:
                import traceback
                logger.info("Error verifying input:", file=sys.stderr)
                logger.info(traceback.format_exc(), file=sys.stderr)

        all_prompts = batch_size * n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

        for iter in range(len(results)):
            b64img = results[iter]["img"]
            base64_bytes = b64img.encode('utf-8')
            img_bytes = base64.b64decode(base64_bytes)
            img = Image.open(BytesIO(img_bytes))

            sanitized_prompt = slugify(prompt)

            prompts = all_prompts[iter * batch_size:(iter + 1) * batch_size]
            #captions = prompt_matrix_parts[n * batch_size:(n + 1) * batch_size]
            seeds = all_seeds[iter * batch_size:(iter + 1) * batch_size]

            if sort_samples:
                full_path = os.path.join(
                    os.getcwd(), sample_path, sanitized_prompt)

                sanitized_prompt = sanitized_prompt[:200-len(full_path)]
                sample_path_i = os.path.join(sample_path, sanitized_prompt)

                #print(f"output folder length: {len(os.path.join(os.getcwd(), sample_path_i))}")
                #print(os.path.join(os.getcwd(), sample_path_i))

                os.makedirs(sample_path_i, exist_ok=True)
                base_count = get_next_sequence_number(sample_path_i)
                filename = f"{base_count:05}-{steps}_{sampler_name}_{seeds[iter]}"
            else:
                full_path = os.path.join(os.getcwd(), sample_path)
                sample_path_i = sample_path
                base_count = get_next_sequence_number(sample_path_i)
                filename = f"{base_count:05}-{steps}_{sampler_name}_{seed}_{sanitized_prompt}"[
                    :200-len(full_path)]  # same as before

            save_sample(img, sample_path_i, filename, jpg_sample, prompts, seeds, width, height, steps, cfg_scale,
                        normalize_prompt_weights, use_GFPGAN, write_info_files, prompt_matrix, init_img=None,
                        denoising_strength=0.75, resize_mode=None, uses_loopback=False, uses_random_seed_loopback=False,
                        save_grid=save_grid,
                        sort_samples=sampler_name, sampler_name=sampler_name, ddim_eta=ddim_eta, n_iter=n_iter,
                        batch_size=batch_size, i=iter, save_individual_images=save_individual_images,
                        model_name="Stable Diffusion v1.5")

            output_images.append(img)

            # update image on the UI so we can see the progress
            if "preview_image" in st.session_state:
                st.session_state["preview_image"].image(img)

            if "progress_bar_text" in st.session_state:
                st.session_state["progress_bar_text"].empty()

            # if len(results) > 1:
                #final_filename = f"{iter}_{filename}"
            # img.save(final_filename)
            #logger.info(f"Saved {final_filename}")
    else:
        if "progress_bar_text" in st.session_state:
            st.session_state["progress_bar_text"].error(submit_req.text)

        logger.error(submit_req.text)

    mem_max_used, mem_total = mem_mon.read_and_stop()
    time_diff = time.time()-start_time

    info = f"""
            {prompt}
            Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', GFPGAN' if use_GFPGAN else ''}{', '+realesrgan_model_name if use_RealESRGAN else ''}
            {', Prompt Matrix Mode.' if prompt_matrix else ''}""".strip()

    stats = f'''
            Took { round(time_diff, 2) }s total ({ round(time_diff/(len(all_prompts)),2) }s per image)
            Peak memory usage: { -(mem_max_used // -1_048_576) } MiB / { -(mem_total // -1_048_576) } MiB / { round(mem_max_used/mem_total*100, 3) }%'''

    for comment in comments:
        info += "\n\n" + comment

    # mem_mon.stop()
    #del mem_mon
    torch_gc()

    return output_images, seed, info, stats


#
@logger.catch(reraise=True)
def txt2img(prompt: str, ddim_steps: int, sampler_name: str, n_iter: int, batch_size: int, cfg_scale: float, seed: Union[int, str, None],
            height: int, width: int, separate_prompts: bool = False, normalize_prompt_weights: bool = True,
            save_individual_images: bool = False, save_grid: bool = False, group_by_prompt: bool = True,
            save_as_jpg: bool = False, use_GFPGAN: bool = True, GFPGAN_model: str = 'GFPGANv1.3', use_RealESRGAN: bool = False,
            RealESRGAN_model: str = "RealESRGAN_x4plus", use_LDSR: bool = True, LDSR_model: str = "model",
            fp=None, variant_amount: float = 0.0,
            variant_seed: int = None, ddim_eta: float = 0.0, write_info_files: bool = False,
            use_stable_horde: bool = False, stable_horde_key: str = ''):

    outpath = st.session_state['defaults'].general.outdir_txt2img

    seed = seed_to_int(seed)

    if not use_stable_horde:

        if sampler_name == 'PLMS':
            sampler = PLMSSampler(server_state["model"])
        elif sampler_name == 'DDIM':
            sampler = DDIMSampler(server_state["model"])
        elif sampler_name == 'k_dpm_2_a':
            sampler = KDiffusionSampler(
                server_state["model"], 'dpm_2_ancestral')
        elif sampler_name == 'k_dpm_2':
            sampler = KDiffusionSampler(server_state["model"], 'dpm_2')
        elif sampler_name == 'k_euler_a':
            sampler = KDiffusionSampler(
                server_state["model"], 'euler_ancestral')
        elif sampler_name == 'k_euler':
            sampler = KDiffusionSampler(server_state["model"], 'euler')
        elif sampler_name == 'k_heun':
            sampler = KDiffusionSampler(server_state["model"], 'heun')
        elif sampler_name == 'k_lms':
            sampler = KDiffusionSampler(server_state["model"], 'lms')
        else:
            raise Exception("Unknown sampler: " + sampler_name)

        def init():
            pass

        def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
            samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale,
                                             unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x,
                                             img_callback=generation_callback if not server_state[
                                                 "bridge"] else None,
                                             log_every_t=int(st.session_state.update_preview_frequency if not server_state["bridge"] else 100))

            return samples_ddim

    if use_stable_horde:
        output_images, seed, info, stats = stable_horde(
            prompt=prompt,
            seed=seed,
            outpath=outpath,
            sampler_name=sampler_name,
            save_grid=save_grid,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=ddim_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            prompt_matrix=separate_prompts,
            use_GFPGAN=use_GFPGAN,
            GFPGAN_model=GFPGAN_model,
            use_RealESRGAN=use_RealESRGAN,
            realesrgan_model_name=RealESRGAN_model,
            use_LDSR=use_LDSR,
            LDSR_model_name=LDSR_model,
            ddim_eta=ddim_eta,
            normalize_prompt_weights=normalize_prompt_weights,
            save_individual_images=save_individual_images,
            sort_samples=group_by_prompt,
            write_info_files=write_info_files,
            jpg_sample=save_as_jpg,
            variant_amount=variant_amount,
            variant_seed=variant_seed,
            api_key=stable_horde_key
        )
    else:

        # try:
        output_images, seed, info, stats = process_images(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            prompt=prompt,
            seed=seed,
            sampler_name=sampler_name,
            save_grid=save_grid,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=ddim_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            prompt_matrix=separate_prompts,
            use_GFPGAN=use_GFPGAN,
            GFPGAN_model=GFPGAN_model,
            use_RealESRGAN=use_RealESRGAN,
            realesrgan_model_name=RealESRGAN_model,
            use_LDSR=use_LDSR,
            LDSR_model_name=LDSR_model,
            ddim_eta=ddim_eta,
            normalize_prompt_weights=normalize_prompt_weights,
            save_individual_images=save_individual_images,
            sort_samples=group_by_prompt,
            write_info_files=write_info_files,
            jpg_sample=save_as_jpg,
            variant_amount=variant_amount,
            variant_seed=variant_seed,
        )

        del sampler

    return output_images, seed, info, stats

    # except RuntimeError as e:
    #err = e
    #err_msg = f'CRASHED:<br><textarea rows="5" style="color:white;background: black;width: -webkit-fill-available;font-family: monospace;font-size: small;font-weight: bold;">{str(e)}</textarea><br><br>Please wait while the program restarts.'
    #stats = err_msg
    # return [], seed, 'err', stats

#


@logger.catch(reraise=True)
def layout():
    if "myPrompt" not in st.session_state:
        st.session_state.myPrompt = ""
    
    if "use_RealESRGAN" not in st.session_state:
        st.session_state.use_RealESRGAN = True
    if "RealESRGAN_model" not in st.session_state:
        st.session_state["RealESRGAN_model"] = "RealESRGAN_x4plus"
    if "custom_model" not in st.session_state:
        st.session_state["custom_model"] = "Stable Diffusion v1.5"
    
    
    print("Reloading layout")
    
    # speechNeedsProcessing = False
    top_col1, top_col2 = st.columns([2, 1])
    
    
    with top_col1:
        
        
        # preview_tab, gallery_tab = st.tabs(["Preview", "Gallery"])
        st.session_state["preview_image"] = st.empty()

        st.session_state["progress_bar_text"] = st.empty()
        st.session_state["progress_bar_text"].info(
            "Nothing but crickets here, try generating something first.")

        st.session_state["progress_bar"] = st.empty()

        

        # with preview_tab:
        #     st.session_state["preview_image"] = st.empty()

        #     st.session_state["progress_bar_text"] = st.empty()
        #     st.session_state["progress_bar_text"].info(
        #         "Nothing but crickets here, try generating something first.")

        #     st.session_state["progress_bar"] = st.empty()

        #     message = st.empty()

        # with gallery_tab:
        #     st.session_state["gallery"] = st.empty()
        #     st.session_state["gallery"].info(
        #         "Nothing but crickets here, try generating something first.")
    
    with top_col2:

        # with st.expander("Output Settings"):
        # message = st.empty()
        val = st_audiorec()
        st.write(
            'Audio data received in the Python backend will appear below this message ...')
        # print(val)
        # print("Val above")
        inferredText = ""
        # st.session_state.successfulRecord = False
        # if isinstance(val, dict):  
        if val and (st.session_state.audioVal != val):
            st.session_state.audioVal = val
            # retrieve audio data
            print(st.session_state.speechNeedsProcessing)
            with st.spinner('analysing speech...'):
                ind, val = zip(*val['arr'].items())
                ind = np.array(ind, dtype=int)  # convert to np array
                val = np.array(val)             # convert to np array
                sorted_ints = val[ind]
                stream = BytesIO(
                    b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
                try:
                    print("initating rec")
                    rec = sr.Recognizer()
                    with sr.AudioFile(stream) as source:
                        audio = rec.record(source)
                    # audio = rec.record(wav_bytes)
                    print("running recognize_vosk")
                    inferredTextJson = rec.recognize_vosk(audio)
                    inferredText = json.loads(inferredTextJson)['text']
                    print(inferredText)
                    st.session_state.successfulRecord = True
                    # val = None
                except Exception as e:
                    print(f"Sorry, couldn't hear due to exception: {e}")
                    # cmd = input()
            if st.session_state.successfulRecord:
                # if st.session_state.extendPrompt:
                #     extended_prompt = prompt_extend_pipe(
                #         inferredText+',', num_return_sequences=1)[0]["generated_text"]
                # else:
                #     extended_prompt = inferredText
                st.success("Inferred speech: " + inferredText)
                st.session_state.myPrompt = inferredText
                st.session_state.speechNeedsProcessing = True
                st.session_state.successfulRecord = False
                # val = None
            else:
                st.error("Nothing here!")
        # while not inferredText:
        #     st.warning('Awaiting recording')
        #     st.stop()

        with st.form("txt2img-inputs"):

            st.session_state["generation_mode"] = "txt2img"
            extendPrompt = st.checkbox("Extend Prompt with auto-generated tags", value=True)


            #prompt = st.text_area("Input Text","")
            placeholder = "A corgi wearing a top hat as an oil painting."
            prompt = st.text_area(
                "Input Text", st.session_state.myPrompt, placeholder=placeholder, height=54)
            # sygil_suggestions.suggestion_area(placeholder)

            width = st.session_state['defaults'].txt2img.width.value
            height = st.session_state['defaults'].txt2img.height.value
            cfg_scale = 11# st.session_state['defaults'].txt2img.cfg_scale.value

            seed = st.session_state['defaults'].txt2img.seed

            st.session_state["batch_count"] = st.session_state['defaults'].txt2img.batch_count.value
            # st.session_state.defaults.txt2img.batch_size.value
            st.session_state["batch_size"] = 1

            st.session_state["update_preview"] = st.session_state["defaults"].general.update_preview
            st.session_state["update_preview_frequency"] = st.session_state[
                'defaults'].txt2img.update_preview_frequency
            custom_models_available()

            st.session_state.sampling_steps = st.session_state.defaults.txt2img.sampling_steps.value

            sampler_name_list = ["k_lms", "k_euler", "k_euler_a",
                                 "k_dpm_2", "k_dpm_2_a",  "k_heun", "PLMS", "DDIM"]
            sampler_name = "k_euler"

            use_stable_horde = False
            stable_horde_key = ''

            separate_prompts = st.session_state['defaults'].txt2img.separate_prompts

            normalize_prompt_weights = st.session_state['defaults'].txt2img.normalize_prompt_weights

            save_individual_images = False

            save_grid = False
            group_by_prompt = st.session_state['defaults'].txt2img.group_by_prompt

            write_info_files = st.session_state['defaults'].txt2img.write_info_files

            save_as_jpg = False

            # check if GFPGAN, RealESRGAN and LDSR are available.
            # if "GFPGAN_available" not in st.session_state:
            GFPGAN_available()

            # if "RealESRGAN_available" not in st.session_state:
            RealESRGAN_available()

            # if "LDSR_available" not in st.session_state:
            LDSR_available()

            if st.session_state["GFPGAN_available"] or st.session_state["RealESRGAN_available"] or st.session_state["LDSR_available"]:
                st.session_state["use_GFPGAN"] = st.session_state['defaults'].txt2img.use_GFPGAN
                st.session_state["GFPGAN_model"] = st.session_state["GFPGAN_models"]

            st.session_state["use_LDSR"] = False
            st.session_state["LDSR_model"] = "model"

            variant_amount = st.session_state['defaults'].txt2img.variant_amount.value
            variant_seed = st.session_state['defaults'].txt2img.seed

            generate_button = st.form_submit_button("Generate")
            
            success = False
            if generate_button or (st.session_state.speechNeedsProcessing and inferredText):
                val = None
                inferredText = ""
                if extendPrompt:
                    prompt = prompt_extend_pipe(
                        prompt+',', num_return_sequences=1)[0]["generated_text"]
                st.session_state.speechNeedsProcessing = False
                prompt += "### (((NSFW))),(((explicit))),(((bad anatomy))),(((watermark))),(((artist name))),(((ugly)))"
                # with col2:
                # if not use_stable_horde:
                #     with hc.HyLoader('Loading Models...', hc.Loaders.standard_loaders, index=[0]):
                #         load_models(use_LDSR=st.session_state["use_LDSR"], LDSR_model=st.session_state["LDSR_model"],
                #                     use_GFPGAN=st.session_state["use_GFPGAN"], GFPGAN_model=st.session_state["GFPGAN_model"],
                #                     use_RealESRGAN=st.session_state[
                #                         "use_RealESRGAN"], RealESRGAN_model=st.session_state["RealESRGAN_model"],
                #                     CustomModel_available=server_state["CustomModel_available"], custom_model=st.session_state["custom_model"])

                output_images, seeds, info, stats = txt2img(prompt, st.session_state.sampling_steps, sampler_name, st.session_state["batch_count"], st.session_state["batch_size"],
                                                            cfg_scale, seed, 512, 512, separate_prompts, normalize_prompt_weights, save_individual_images,
                                                            save_grid, group_by_prompt, save_as_jpg, st.session_state[
                                                                "use_GFPGAN"], st.session_state['GFPGAN_model'],
                                                            use_RealESRGAN=st.session_state[
                                                                "use_RealESRGAN"], RealESRGAN_model=st.session_state["RealESRGAN_model"],
                                                            use_LDSR=st.session_state[
                                                                "use_LDSR"], LDSR_model=st.session_state["LDSR_model"],
                                                            variant_amount=variant_amount, variant_seed=variant_seed, write_info_files=write_info_files,
                                                            use_stable_horde=use_stable_horde, stable_horde_key=stable_horde_key)
                success = True
                # message.success('Render Complete: ' + info +
                #                 '; Stats: ' + stats, icon="✅")
        if success:
            st.success('Render Complete: ' + info +
                                '; Stats: ' + stats, icon="✅")
            success = False
                # with gallery_tab:
                #     logger.info(seeds)
                #     sdGallery(output_images)
            
    


