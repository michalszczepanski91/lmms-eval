# TextVQA

Task `textvqa_val`, tokenizer `Qwen/Qwen2.5-VL-3B-Instruct`.

| | |
|---|---|
| Questions (distinct images) | 5,000 (3,166) |
| Image size, MP, avg (min–max) | 0.77 (0.26–15.93), avg 952×819 px |
| Image tokens, avg (min–max) | 976.5 (333–2,028) |
| Question tokens (prompt text), avg (min–max) | 18.3 (13–49) |
| Reference answer words, avg (min–max) | 1.55 (1–12) |
| Reference answer tokens, avg (min–max) | 2.92 (1–22) |
| Reference answers of one word / with a digit | 72% / 24% |
| Scoring | VQA accuracy: min(1, human answers matching the model's / 3), averaged over the 10 leave-one-out subsets |

## Examples

### doc 2

![textvqa_val doc 2](textvqa_val_2.jpg)

| | |
|---|---|
| Image | 1024×1024 px → resized 1036×1036 px → 1369 image tokens |
| Prompt (17 tokens) | What kind of beer is this?<br>Answer the question using a single word or phrase. |
| Reference | "ale" ×7, "stone" ×1, "self righteous" ×1, "sublimely self-righteous ale" ×1 |
| Reference tokens | `ale` |

### doc 549

![textvqa_val doc 549](textvqa_val_549.jpg)

| | |
|---|---|
| Image | 1024×684 px → resized 1036×672 px → 888 image tokens |
| Prompt (18 tokens) | What brands of beer are there here?<br>Answer the question using a single word or phrase. |
| Reference | "caracu, antartica, itaipava, evassa, paulistania, petra" ×2, "caracu, antarctica, itaipava" ×1, "caracu antarciaca itaipava devassa paulistania petra" ×1, "caracu, antarctica, itaipava, devassa, paulistania, petra" ×1, "caracu, antartica, itaipava, devassa paulistania, and petra" ×1, "petra , paulistania , devassa, itaipava " ×1, "caracu, antarctica, itaipava, devassa, paulistiania, petra" ×1, "unanswerable" ×1, "caracu, antarctica, itaipava, evassa, paulistania, petra" ×1 |
| Reference tokens | `car` `ac` `u` `,` `␣ant` `art` `ica` `,` `␣it` `a` `ip` `ava` `,` `␣ev` `assa` `,` `␣paul` `istan` `ia` `,` `␣pet` `ra` |
