from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Values are pulled from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_research_model: str = "qwen3.5:4b"
    ollama_num_ctx: int = 16384
    ollama_num_predict: int = 2048

    # Groq (Creative Strategy Agent)
    groq_api_key: str = ""
    groq_primary_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    groq_max_tokens: int = 2048
    groq_temperature: float = 0.7

    # Groq (Prompt Generation Agent)
    prompt_gen_primary_model: str = "openai/gpt-oss-120b"
    prompt_gen_fallback_model: str = "openai/gpt-oss-20b"
    prompt_gen_max_tokens: int = 2048
    prompt_gen_temperature: float = 0.6

    tavily_api_key: str = ""

    # Scraper
    scrape_timeout_ms: int = 30000
    nav_attempt_timeout_ms: int = 10000
    scrape_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    max_visible_text_chars: int = 20000
    max_review_text_chars: int = 8000
    min_review_text_chars: int = 200
    review_wait_timeout_ms: int = 4000

    # Agent
    max_extraction_retries: int = 2
    max_creative_retries: int = 2
    max_prompt_gen_retries: int = 2

    # ComfyUI - image
    comfyui_server: str = "http://127.0.0.1:8188"
    comfyui_checkpoint: str = "juggernautXL_ragnarok.safetensors"
    comfyui_steps: int = 20
    comfyui_cfg: float = 7.0
    comfyui_sampler: str = "dpmpp_2m_sde"
    comfyui_scheduler: str = "karras"
    comfyui_batch_size: int = 5
    comfyui_generation_timeout_seconds: float = 180.0
    total_images_per_product: int = 5
    image_output_dir: str = "outputs/generated_images"
    max_image_gen_retries: int = 2

    # ---------------------------------------------------------------
    # ComfyUI - video (Agent 6: NimVideo/cogvideox-2b-img2vid, run
    # through a custom node graph, not core ComfyUI nodes - see
    # Video_generation.md Challenge 1 for why). Node IDs below are
    # pinned to the specific exported API-format workflow JSON that
    # was hand-tested and confirmed working - unlike the image
    # workflow's node IDs (which are just "1" through "7" by
    # convention, since build_image_workflow() constructs that graph
    # itself), these come from a workflow file this project doesn't
    # generate and can't assume the shape of. If you re-export the
    # workflow JSON from ComfyUI later (e.g. after upgrading the
    # custom node package), re-verify these IDs still match - nothing
    # here will error loudly if the workflow's internal node numbering
    # ever shifts, it'll just silently patch the wrong node.
    # ---------------------------------------------------------------
    comfyui_video_workflow_json: str = "workflows/cogvideox-2b-img2vid-workflow-API.json"
    comfyui_video_image_load_node_id: str = "36"
    comfyui_video_positive_prompt_node_id: str = "30"
    comfyui_video_negative_prompt_node_id: str = "31"
    comfyui_video_sampler_node_id: str = "63"
    comfyui_video_save_node_id: str = "44"

    # Frame count IS a real runtime parameter on this checkpoint's ComfyUI
    # sampler node, despite Prompt_generation.md Challenge 6 assuming
    # duration was fixed by the checkpoint itself - that assumption was
    # wrong (or at least incomplete), confirmed once this node's actual
    # inputs were inspected. Fixed at 8fps regardless. CogVideoX's temporal
    # VAE requires frame counts of the form 4n+1, so an exact 5.00s isn't
    # reachable - 41 frames (5.125s) is the nearest valid value below the
    # 49 (6.125s) used in the original hand-test; 37 (4.625s) is the next
    # one down if you'd rather undershoot than overshoot.
    comfyui_video_num_frames: int = 41
    comfyui_video_steps: int = 20
    comfyui_video_cfg: float = 6.0
    comfyui_video_scheduler: str = "CogVideoXDDIM"
    comfyui_video_denoise_strength: float = 1.0
    comfyui_video_default_negative_prompt: str = (
        "low quality, blurry, watermark, bad anatomy, distorted, "
        "flickering, artifacts"
    )
    # UPDATED after a real measurement, not the earlier guess: one
    # sampling step took ~186s on this RTX 3050 at 49 frames/20 steps/
    # 720x480, i.e. ~3700s (~62min) for a full generation. 1800s
    # guaranteed every attempt would time out before finishing - raised
    # to a real ceiling above the observed per-run cost, with headroom.
    # If you'd rather trade video length/quality for speed instead of
    # accepting ~60min/video, lower comfyui_video_num_frames and/or
    # comfyui_video_steps below instead of raising this further.
    comfyui_video_generation_timeout_seconds: float = 4500.0

    # One video per theme, not a total split across themes the way
    # images use distribute_total() - see Video_generation.md Challenge 2
    # for why oversampling/batching doesn't make sense at video's cost
    # profile the way it didn't for images either (Image_generation.md
    # Challenge 2), just more so here.
    # Was 1. At ~60min per attempt, a single retry now costs another hour
    # for a theme that already failed once - worth deciding deliberately
    # rather than inheriting image's retry count. 0 means a failed theme
    # is skipped immediately rather than tried twice; raise back to 1 only
    # if you have evidence failures are transient (infra blips) rather
    # than this checkpoint genuinely being too slow/tight for this card.
    max_video_gen_retries: int = 0
    video_output_dir: str = "outputs/generated_videos"

    # ---------------------------------------------------------------
    # Image Selection (Critic) Agent - Agent 5, runs between Agent 4 and
    # Agent 6 (Video Generation). Originally used Groq's qwen/qwen3.6-27b
    # (hosted vision preview model), but that hit two separate real
    # problems in practice: an OTPM 429 (1000 output-tokens/minute cap on
    # this org's tier) and, after fixing that, a deterministic
    # json_validate_failed from Groq's own strict server-side JSON-schema
    # validation - roughly 2/3 of calls failed all retries and fell back
    # to an unjudged first candidate, defeating the point of the agent.
    #
    # Switched to running locally via Ollama's qwen3.5:4b, whose vision
    # capability was confirmed working (`ollama show qwen3.5:4b` lists
    # "vision"; a real, accurate description came back for an actual test
    # image) before committing to the switch. This also sidesteps the
    # json_validate_failed failure category by construction - Ollama's
    # path uses this project's own tolerant JSON parsing (see llm.py's
    # structured_chat_vision), not a provider's strict server-side
    # validator, and has no per-minute output-token quota to hit at all.
    #
    # Tradeoff worth watching: qwen3.5:4b is far smaller than the 27B
    # model it replaced, so image-judgment quality may be less nuanced -
    # and it now runs on the same RTX 3050 ComfyUI uses for image/video
    # generation, so there's VRAM to share (though not contend for at the
    # same instant, since this agent runs after Agent 4 finishes and
    # before Agent 6 starts).
    # ---------------------------------------------------------------
    image_selection_model: str = "qwen3.5:4b"
    # Feeds num_predict for the Ollama call (see structured_chat_vision) -
    # no longer constrained by Groq's OTPM cap, so this has real headroom
    # now; kept at a moderate value since the actual output (a handful of
    # scores plus short rationale) doesn't need much more than this.
    image_selection_max_tokens: int = 1024
    image_selection_temperature: float = 0.2
    # Below this score, the pick is still made (Agent 5 needs one image per
    # theme regardless) but flagged "selected_below_threshold" rather than
    # "selected" - see image_selection_agent/schema.py.
    image_selection_score_threshold: int = 55
    max_image_selection_retries: int = 2
    # No longer a hard API ceiling (Groq's 5-images-per-request cap doesn't
    # apply locally) - kept as a soft cap for latency/focus: a small local
    # model judging fewer images at once is faster and likely more
    # reliable than one judging many at once, even though it could
    # technically be sent more given qwen3.5:4b's 262K context.
    image_selection_max_images_per_call: int = 5


settings = Settings()