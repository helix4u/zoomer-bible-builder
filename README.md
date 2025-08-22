bible yanked from https://github.com/aruljohn/Bible-kjv

probably needs like 'pip install requests', idk.

example usage:

python zoomer_bible_builder.py --repo-dir .\Bible-kjv --model "gemma-3-4b-it" --start-index 0 --end-index 10 --no-stream

added tts by chapter for use with kokoro fastapi

python tts_by_chapter.py zoomer_bible.txt --outdir D:\AITools\zoomer-bible-builder\bible_audio --voice "af_sky+af+af_nicole" --speed 1.75 --format mp3 --skip-existing
