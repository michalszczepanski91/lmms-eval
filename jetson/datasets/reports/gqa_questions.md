# GQA: questions 0–49

Task `gqa`, tokenizer `Qwen/Qwen2.5-VL-3B-Instruct`, answers from `jetson/results/thor/Qwen2.5-VL-3B-Instruct/gqa/hf-bf16/20260925-062406`.

- **Image tokens:** the image is resized to multiples of 28 px within 256–2,048 tokens; each 28×28 px block is one token (W × H = tokens across × down).
- **Prompt tokens:** the prompt text only; the chat template adds 21 tokens. The model reads prompt + image tokens.
- **Answer tokens:** the model's answer tokenized; **generated tokens:** as logged by the run (end token included; empty for Edge-LLM).
- **Reference:** up to 3 of the most common reference answers.

| Doc | Prompt text | Prompt tokens | Image size (px) | Resized (px) | Image tokens W × H | Image tokens | Model answer | Answer tokens | Generated tokens | Reference | Score |
|---:|---|---:|---|---|---|---:|---|---:|---:|---|---:|
| 0 | Is it overcast?<br>Answer the question using a single word or phrase. | 15 | 500×280 | 616×336 | 22 × 12 | 264 | no | 1 | 2 | no | 1 |
| 1 | Who is wearing the dress?<br>Answer the question using a single word or phrase. | 16 | 500×375 | 532×392 | 19 × 14 | 266 | woman | 1 | 2 | women | 0 |
| 2 | Does the utensil on top of the table look clean and black?<br>Answer the question using a single word or phrase. | 24 | 640×426 | 644×420 | 23 × 15 | 345 | no | 1 | 2 | no | 1 |
| 3 | Is the surfer that looks wet wearing a wetsuit?<br>Answer the question using a single word or phrase. | 23 | 640×480 | 644×476 | 23 × 17 | 391 | yes | 1 | 2 | yes | 1 |
| 4 | How tall is the chair in the bottom of the photo?<br>Answer the question using a single word or phrase. | 22 | 640×480 | 644×476 | 23 × 17 | 391 | 4 feet | 2 | 3 | short | 0 |
| 5 | What kind of device is on top of the desk?<br>Answer the question using a single word or phrase. | 21 | 500×375 | 532×392 | 19 × 14 | 266 | computer | 1 | 2 | keyboard | 0 |
| 6 | What is the airplane flying above?<br>Answer the question using a single word or phrase. | 17 | 640×424 | 644×420 | 23 × 15 | 345 | ocean | 2 | 3 | ocean | 1 |
| 7 | What color are the pants?<br>Answer the question using a single word or phrase. | 16 | 640×480 | 644×476 | 23 × 17 | 391 | red | 1 | 2 | red | 1 |
| 8 | Is the ground blue or brown?<br>Answer the question using a single word or phrase. | 17 | 640×396 | 644×392 | 23 × 14 | 322 | brown | 1 | 2 | brown | 1 |
| 9 | What is around the open window?<br>Answer the question using a single word or phrase. | 17 | 640×232 | 756×280 | 27 × 10 | 270 | curtains | 3 | 4 | drapes | 0 |
| 10 | What's around the window?<br>Answer the question using a single word or phrase. | 16 | 640×232 | 756×280 | 27 × 10 | 270 | curtains | 3 | 4 | drapes | 0 |
| 11 | Who is standing at the table?<br>Answer the question using a single word or phrase. | 17 | 427×640 | 420×644 | 15 × 23 | 345 | people | 1 | 2 | woman | 0 |
| 12 | Are there drapes to the right of the bed?<br>Answer the question using a single word or phrase. | 21 | 640×232 | 756×280 | 27 × 10 | 270 | yes | 1 | 2 | yes | 1 |
| 13 | What is hanging above the chalkboard?<br>Answer the question using a single word or phrase. | 18 | 427×640 | 420×644 | 15 × 23 | 345 | picture | 1 | 2 | picture | 1 |
| 14 | Is the cake on a platter?<br>Answer the question using a single word or phrase. | 18 | 640×428 | 644×420 | 23 × 15 | 345 | no | 1 | 2 | no | 1 |
| 15 | Is the person to the right of the cup wearing jeans?<br>Answer the question using a single word or phrase. | 22 | 375×500 | 392×532 | 14 × 19 | 266 | yes | 1 | 2 | yes | 1 |
| 16 | What device is sitting next to the mouse pad?<br>Answer the question using a single word or phrase. | 20 | 500×375 | 532×392 | 19 × 14 | 266 | keyboard | 1 | 2 | keyboard | 1 |
| 17 | Does the sweater look open and blue?<br>Answer the question using a single word or phrase. | 18 | 640×425 | 644×420 | 23 × 15 | 345 | yes | 1 | 2 | yes | 1 |
| 18 | Is the jacket long sleeved and black?<br>Answer the question using a single word or phrase. | 19 | 640×480 | 644×476 | 23 × 17 | 391 | yes | 1 | 2 | yes | 1 |
| 19 | Are there beds next to the small outlet?<br>Answer the question using a single word or phrase. | 19 | 612×612 | 616×616 | 22 × 22 | 484 | yes | 1 | 2 | yes | 1 |
| 20 | On which side of the picture is the leather bag?<br>Answer the question using a single word or phrase. | 21 | 640×640 | 644×644 | 23 × 23 | 529 | right | 1 | 2 | right | 1 |
| 21 | Is the blue pillow square and large?<br>Answer the question using a single word or phrase. | 18 | 640×427 | 644×420 | 23 × 15 | 345 | yes | 1 | 2 | no | 0 |
| 22 | Which color is the cake?<br>Answer the question using a single word or phrase. | 16 | 640×480 | 644×476 | 23 × 17 | 391 | white | 1 | 2 | white | 1 |
| 23 | What is the name of the cooking utensil that is hang from the hook?<br>Answer the question using a single word or phrase. | 26 | 427×640 | 420×644 | 15 × 23 | 345 | pan | 1 | 2 | pan | 1 |
| 24 | Where is the skinny person standing?<br>Answer the question using a single word or phrase. | 17 | 427×640 | 420×644 | 15 × 23 | 345 | behind table | 3 | 4 | table | 0 |
| 25 | Are the plates on top of an ottoman?<br>Answer the question using a single word or phrase. | 20 | 640×480 | 644×476 | 23 × 17 | 391 | no | 1 | 2 | no | 1 |
| 26 | Is the freezer near the wall small or large?<br>Answer the question using a single word or phrase. | 20 | 640×480 | 644×476 | 23 × 17 | 391 | small | 1 | 2 | large | 0 |
| 27 | What type of food is to the left of the baby that is sitting atop the woman?<br>Answer the question using a single word or phrase. | 28 | 640×480 | 644×476 | 23 × 17 | 391 | dessert | 2 | 3 | marshmallow | 0 |
| 28 | Is the fence made of cement or aluminum?<br>Answer the question using a single word or phrase. | 19 | 494×640 | 504×644 | 18 × 23 | 414 | aluminum | 2 | 3 | aluminum | 1 |
| 29 | Which side are the white houses on?<br>Answer the question using a single word or phrase. | 18 | 640×480 | 644×476 | 23 × 17 | 391 | left | 1 | 2 | left | 1 |
| 30 | Are both the phone and the coffee cup the same color?<br>Answer the question using a single word or phrase. | 22 | 640×425 | 644×420 | 23 × 15 | 345 | no | 1 | 2 | yes | 0 |
| 31 | Are there either any small refrigerators or microwaves in the picture?<br>Answer the question using a single word or phrase. | 25 | 640×480 | 644×476 | 23 × 17 | 391 | no | 1 | 2 | no | 1 |
| 32 | How does that car look like, orange or maybe white?<br>Answer the question using a single word or phrase. | 22 | 640×427 | 644×420 | 23 × 15 | 345 | white | 1 | 2 | white | 1 |
| 33 | What color is the book?<br>Answer the question using a single word or phrase. | 16 | 640×468 | 644×476 | 23 × 17 | 391 | white | 1 | 2 | white | 1 |
| 34 | What color is the dirt?<br>Answer the question using a single word or phrase. | 16 | 640×480 | 644×476 | 23 × 17 | 391 | brown | 1 | 2 | red | 0 |
| 35 | Is the bag made of leather lying on top of a sofa?<br>Answer the question using a single word or phrase. | 23 | 640×468 | 644×476 | 23 × 17 | 391 | no | 1 | 2 | no | 1 |
| 36 | What are the drapes around of?<br>Answer the question using a single word or phrase. | 18 | 640×232 | 756×280 | 27 × 10 | 270 | window | 1 | 2 | window | 1 |
| 37 | On which side is the picture?<br>Answer the question using a single word or phrase. | 17 | 434×640 | 448×644 | 16 × 23 | 368 | left | 1 | 2 | left | 1 |
| 38 | What material is the crosswalk in front of the stores?<br>Answer the question using a single word or phrase. | 22 | 640×427 | 644×420 | 23 × 15 | 345 | asphalt | 2 | 3 | concrete | 0 |
| 39 | How large are the sprinkles that are sprinkled on the cupcakes?<br>Answer the question using a single word or phrase. | 24 | 500×375 | 532×392 | 19 × 14 | 266 | small | 1 | 2 | small | 1 |
| 40 | What type of material is the crosswalk near the street lamp made of?<br>Answer the question using a single word or phrase. | 25 | 640×427 | 644×420 | 23 × 15 | 345 | pavement | 3 | 4 | concrete | 0 |
| 41 | Which kind of clothing is pink?<br>Answer the question using a single word or phrase. | 17 | 640×427 | 644×420 | 23 × 15 | 345 | tank top | 2 | 3 | tank top | 1 |
| 42 | How is the clothing item that is pink called?<br>Answer the question using a single word or phrase. | 20 | 640×427 | 644×420 | 23 × 15 | 345 | tank top | 2 | 3 | tank top | 1 |
| 43 | Which kind of clothing is not pink?<br>Answer the question using a single word or phrase. | 18 | 640×427 | 644×420 | 23 × 15 | 345 | shorts | 2 | 3 | hat | 0 |
| 44 | Is this helicopter on or off?<br>Answer the question using a single word or phrase. | 17 | 640×480 | 644×476 | 23 × 17 | 391 | on | 1 | 2 | off | 0 |
| 45 | Do you see any cats?<br>Answer the question using a single word or phrase. | 16 | 640×491 | 644×504 | 23 × 18 | 414 | yes | 1 | 2 | yes | 1 |
| 46 | Is that shoe behind a dog?<br>Answer the question using a single word or phrase. | 17 | 640×491 | 644×504 | 23 × 18 | 414 | no | 1 | 2 | no | 1 |
| 47 | What kind of clothing is sleeveless?<br>Answer the question using a single word or phrase. | 18 | 640×427 | 644×420 | 23 × 15 | 345 | tank top | 2 | 3 | tank top | 1 |
| 48 | Is the field soft and snowy?<br>Answer the question using a single word or phrase. | 17 | 640×427 | 644×420 | 23 × 15 | 345 | no | 1 | 2 | no | 1 |
| 49 | What fruits are beneath the microwave?<br>Answer the question using a single word or phrase. | 17 | 480×640 | 476×644 | 17 × 23 | 391 | bananas | 2 | 3 | bananas | 1 |
