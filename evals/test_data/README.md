# Eval test data

The image / audio / video eval cases reference files in this directory:

- `professional_image.jpg` — a clean product photo
- `blurry_image.jpg` — a deliberately low-quality image
- `id_document.jpg` — anything with visible PII (IDs, addresses, etc.)
- `disturbing_image.jpg` — graphic / NSFW content to flag
- `professional_audio.mp3`, `noisy_audio.mp3`, `rude_audio.mp3`
- `professional_video.mp4`, `low_quality_video.mp4`, `disturbing_video.mp4`

Drop your own assets here. Cases for missing files are reported as errors
rather than crashing the whole run, so you can fill them in over time.
