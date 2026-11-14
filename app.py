import os
import gradio as gr
from huggingface_hub import InferenceClient

# Load token and private login credentials securely from Space Secrets
HF_TOKEN = os.environ.get("HF_TOKEN")
APP_USER = os.environ.get("APP_USER", "author")
APP_PASS = os.environ.get("APP_PASS")

client = InferenceClient(token=HF_TOKEN)

SYSTEM_PROMPT = (
    "You are a master Nobel-caliber book writer and werewolf fiction co-author. "
    "Write atmospheric, emotionally gripping, literary prose with rich world-building, "
    "strictly maintaining pack lore and narrative continuity."
)

def write_chapter(prompt, history, temp):
    if not prompt or not prompt.strip():
        return "Please enter a scene prompt or chapter instruction."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat_completion(
            model="inclusionAI/Ling-3.1-flash",
            messages=messages,
            max_tokens=2048,
            temperature=temp,
        )
        return response.choices.message.content
    except Exception as e:
        return f"⚠️ Notice: {str(e)}"

def generate_scene_art(scene_prompt):
    if not scene_prompt or not scene_prompt.strip():
        return None
    image = client.text_to_image(
        prompt=f"Nobel literary book cover illustration, werewolf lore, cinematic natural lighting, clean painterly style, {scene_prompt}",
        model="black-forest-labs/FLUX.1-schnell",
    )
    return image

def generate_book_trailer(video_prompt):
    if not video_prompt or not video_prompt.strip():
        return None
    video_bytes = client.text_to_video(
        prompt=f"Cinematic literary novel book trailer shot, luminous atmosphere, {video_prompt}",
        model="Wan-AI/Wan2.1-T2V-14B",
    )
    output_path = "trailer_clip.mp4"
    with open(output_path, "wb") as f:
        f.write(video_bytes)
    return output_path

with gr.Blocks(title="Knowledge AI Studio", analytics_enabled=False) as demo:
    gr.Markdown("# 📖 Knowledge AI Studio")
    gr.Markdown("Private co-writing and visual studio powered by **Ling 3.1 Flash**, **FLUX.1**, and **Wan 2.1**.")

    with gr.Tab("✍️ Chapter & Lore Writer"):
        temp_slider = gr.Slider(0.5, 1.2, value=0.9, label="Creativity (Temperature)")
        gr.ChatInterface(
            fn=write_chapter,
            additional_inputs=[temp_slider],
            examples=[
                ["Write an opening chapter where an ancient alpha discovers a forgotten treaty under a silver full moon.", 0.9],
                ["Create a rich cultural history and lineage for three noble werewolf clans.", 0.85],
            ],
        )

    with gr.Tab("🎨 Book Cover & Character Art"):
        art_input = gr.Textbox(
            label="Describe your character or scene",
            placeholder="Silver-furred werewolf standing on a sunlit mountain ridge overlooking an ancient valley..."
        )
        art_btn = gr.Button("Generate Artwork", variant="primary")
        art_output = gr.Image(label="Generated Book Art")
        art_btn.click(fn=generate_scene_art, inputs=art_input, outputs=art_output)

    with gr.Tab("🎬 Book Trailer Video"):
        vid_input = gr.Textbox(
            label="Describe the trailer motion scene",
            placeholder="Slow sweeping camera through a misty northern pine forest at dawn..."
        )
        vid_btn = gr.Button("Generate Video Clip", variant="primary")
        vid_output = gr.Video(label="Generated Trailer Clip")
        vid_btn.click(fn=generate_book_trailer, inputs=vid_input, outputs=vid_output)

if __name__ == "__main__":
    if APP_PASS:
        demo.launch(
            theme=gr.themes.Soft(),
            auth=(APP_USER, APP_PASS),
            auth_message="🔒 Authorized Family Access Only",
            pwa=True,
        )
    else:
        demo.launch(theme=gr.themes.Soft(), pwa=True)
