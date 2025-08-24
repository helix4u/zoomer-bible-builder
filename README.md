# Zoomer Bible Builder

**Because the world doesn't have enough weird AI shit yet. This is a python project to slam the Bible (and the Book of Mormon, if you want) through a local LLM, so you get a version that vibes more with meme kids and ADHD brains.**

## What Is This?

- Grabs the King James Bible ([source](https://github.com/aruljohn/Bible-kjv)), yanks the Book of Mormon ([source](https://github.com/johngthecreator/Book_of_Mormon_Scriptures)), and uses a local LLM to translate that holy text into whatever the hell Zoomer-speak is.
- Toss in some Text-to-Speech for fun audio output, perfect for TikTok or torturing your religious relatives.

## Requirements

- Python. Obviously.
- Probably `pip install requests`, but who knows, you'll figure it out.
- Whatever LLM you like, just point the damn thing at it.

## Example Sorcery

```bash
# Zap the Bible into Zoomer-lingo
python zoomer_bible_builder.py --repo-dir .\Bible-kjv --model "gemma-3-4b-it" --no-stream --ctx-pairs 10
```

**Text-to-Speech? Say less:**
```bash
# Bible audio, go brrr
python tts_by_chapter.py zoomer_bible.txt --outdir D:\AITools\zoomer-bible-builder\bible_audio --voice "af_sky+af+af_nicole" --speed 1.75 --format mp3 --skip-existing

# Book of Mormon gets the treatment too
python tts_by_chapter.py zoomer_bom.txt --outdir D:\AITools\zoomer-bible-builder\bom_audio --voice "af_sky+af+af_nicole" --speed 1.75 --format mp3 --skip-existing
```

If the BOM script starts looping like a Sunday school dropout, yeah, that's on the to-fix pile.

**Book of Mormon builder—watch it bug out sometimes, I'm not your therapist:**
```bash
python zoomer_bom_builder.py --system-prompt system_zoomer_prompt.txt --bom-json .\Book_of_Mormon_Scriptures\book-of-mormon.json --model "qwen3-4b-instruct-2507" --no-stream --ctx-pairs 20
```

Pro tip: You can now plug in an API key if you hate freedom and want to use hosted models.

## Credits

- King James Bible ripped from [aruljohn/Bible-kjv](https://github.com/aruljohn/Bible-kjv)
- Book of Mormon yanked from [johngthecreator/Book_of_Mormon_Scriptures](https://github.com/johngthecreator/Book_of_Mormon_Scriptures)

## License

Do what you want, just don't be annoying. You're on your own.
