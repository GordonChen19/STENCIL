from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from tqdm import tqdm


class BaseModel:
    def __init__(self, model_id="sd-legacy/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.pipe = None

    def load_model(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.pipe = self.pipe.to("cuda:0")
        self.pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()

    def fine_tune(self, emb, latent, latent_mask, iter=100, bsz=4):

        #Only finetune the decoder

        self.pipe.unet.requires_grad_(True)
        emb.requires_grad_(True)

        for name, param in self.pipe.unet.named_parameters():
            if "up_blocks" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


        params = [{"params": [p for n, p in self.pipe.unet.named_parameters() if "up_blocks" in n], "lr": 2e-5},
                {"params":[emb], "lr": 1e-5}]


        optimizer = torch.optim.Adam(params)

        self.pipe.unet.train()

        pbar = tqdm(range(iter))

        for epoch, step in enumerate(pbar):
            with torch.no_grad():
                batch_indices = torch.randperm(latent.shape[0])[:bsz]
                latent_batch = latent[batch_indices]
                latent_mask_batch = latent_mask[batch_indices]
                emb_batch = emb[batch_indices]


                noise = torch.randn_like(latent_batch)
                timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps,(latent_batch.shape[0],), device = self.pipe.device).long()
                noisy_latents = self.pipe.scheduler.add_noise(latent_batch, noise, timesteps)

            pred_noise = self.pipe.unet(noisy_latents, timesteps, emb_batch).sample

            total_loss = ((pred_noise - noise) * latent_mask_batch)**2

            loss = total_loss.mean()

            pbar.set_description(f"Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.pipe.unet.eval()
        emb.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        return emb







from diffusers.models.attention_processor import Attention

class ControlledAttnProcessor:

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # if attn.save_last_attn_slice:
        #     attn.last_attn_slice = attention_probs
        #     attn.save_last_attn_slice = False

        # if attn.use_last_attn_slice:
        #     attention_probs = attn.last_attn_slice
        #     # for source_idx, mapped_idx in enumerate(pipe.token_mapping):
        #     #     attention_probs[:,:,mapped_idx] = attn.last_attn_slice[:, :, source_idx]
        #     attn.use_last_attn_slice = False

        if attn.save_attn_map:

            attention_map = attention_probs[:,:,pipe.target_token_idx]
            query_length = attention_map.shape[1]
            spatial_resolution = int(query_length**0.5)
            attention_map = attention_map.view(attention_probs.shape[0],1,spatial_resolution, spatial_resolution)

            attention_map = attention_map.mean(dim=1).unsqueeze(1)
            attention_map = attention_map.mean(dim=0).unsqueeze(0)

            target_resolution = 16
            attention_map_rescaled = F.interpolate(
                attention_map, size=(target_resolution, target_resolution), mode="bilinear", align_corners=False
            )

            if pipe.attention_maps is None:
                pipe.attention_maps = attention_map_rescaled
            else:
                pipe.attention_maps =  pipe.attention_maps + attention_map_rescaled


        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
for name, module in pipe.unet.named_modules():
    module_name = type(module).__name__
    if module_name == "Attention" and "attn2" in name:
        module.set_processor(ControlledAttnProcessor())



def save_cross_attention_map(save=True):
    pipe.attention_maps = None
    for name, module in pipe.unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and ('attn2' in name):
            module.save_attn_map = save
        else:
            module.save_attn_map = False

save_cross_attention_map()