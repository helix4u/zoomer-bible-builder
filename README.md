bible yanked from https://github.com/aruljohn/Bible-kjv

bom snagged from https://github.com/johngthecreator/Book_of_Mormon_Scriptures

probably needs like 'pip install requests', idk.

example usage:

python zoomer_bible_builder.py --repo-dir .\Bible-kjv --model "gemma-3-4b-it" --no-stream --ctx-pairs 10

added tts by chapter for use with kokoro fastapi

python tts_by_chapter.py zoomer_bible.txt --outdir D:\AITools\zoomer-bible-builder\bible_audio --voice "af_sky+af+af_nicole" --speed 1.75 --format mp3 --skip-existing

python tts_by_chapter.py zoomer_bom.txt --outdir D:\AITools\zoomer-bible-builder\bom_audio --voice "af_sky+af+af_nicole" --speed 1.75 --format mp3 --skip-existing

zoomer bom is tougher to get to behave. need to fix the thing getting wonky with repetition. 
python zoomer_bom_builder.py --system-prompt system_zoomer_prompt.txt --bom-json .\Book_of_Mormon_Scriptures\book-of-mormon.json --model "qwen3-4b-instruct-2507" --no-stream --ctx-pairs 20


