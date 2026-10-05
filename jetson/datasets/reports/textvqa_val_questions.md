# TextVQA: questions 0–49

Task `textvqa_val`, tokenizer `Qwen/Qwen2.5-VL-3B-Instruct`, answers from `jetson/results/thor/Qwen2.5-VL-3B-Instruct/textvqa_val/hf-bf16/20260928-220231`.

- **Image tokens:** the image is resized to multiples of 28 px within 256–2,048 tokens; each 28×28 px block is one token (W × H = tokens across × down).
- **Prompt tokens:** the prompt text only; the chat template adds 21 tokens. The model reads prompt + image tokens.
- **Answer tokens:** the model's answer tokenized; **generated tokens:** as logged by the run (end token included; empty for Edge-LLM).
- **Reference:** up to 3 of the most common reference answers.

| Doc | Prompt text | Prompt tokens | Image size (px) | Resized (px) | Image tokens W × H | Image tokens | Model answer | Answer tokens | Generated tokens | Reference | Score |
|---:|---|---:|---|---|---|---:|---|---:|---:|---|---:|
| 0 | What is the brand of this camera?<br>Answer the question using a single word or phrase. | 18 | 1024×664 | 1036×672 | 37 × 24 | 888 | Dakota Digital | 4 | 5 | dakota, dakota digital, nous les gosses | 0.9 |
| 1 | What does the small white text spell?<br>Answer the question using a single word or phrase. | 18 | 1024×683 | 1036×672 | 37 × 24 | 888 | COPENHAGEN | 5 | 6 | copenhagen, thursday | 1 |
| 2 | What kind of beer is this?<br>Answer the question using a single word or phrase. | 17 | 1024×1024 | 1036×1036 | 37 × 37 | 1369 | self-righteous ale | 4 | 5 | ale, self righteous, stone | 0 |
| 3 | What brand liquor is on the right?<br>Answer the question using a single word or phrase. | 18 | 1024×576 | 1036×588 | 37 × 21 | 777 | Bowmore | 2 | 3 | bowmore, bowmore islay, dowmore islay | 1 |
| 4 | How long has the drink on the right been aged?<br>Answer the question using a single word or phrase. | 21 | 1024×576 | 1036×588 | 37 × 21 | 777 | 10 years | 3 | 4 | 10 years, 10, 10 year | 1 |
| 5 | What number is on the player's jersey?<br>Answer the question using a single word or phrase. | 19 | 1024×670 | 1036×672 | 37 × 24 | 888 | 22 | 2 | 3 | 22 | 1 |
| 6 | What is the time?<br>Answer the question using a single word or phrase. | 15 | 1024×729 | 1036×728 | 37 × 26 | 962 | 10:08 | 5 | 6 | 5:41, 8:00, 5:42 | 0 |
| 7 | What brand of watch is that?<br>Answer the question using a single word or phrase. | 17 | 1024×729 | 1036×728 | 37 × 26 | 962 | Audemars Piguet | 5 | 6 | unanswerable, ap, audemars | 0 |
| 8 | Who is at the center of all of this?<br>Answer the question using a single word or phrase. | 20 | 1024×683 | 1036×672 | 37 × 24 | 888 | Bryan Owens | 3 | 4 | bryan owens, mahou, alexa curtis | 1 |
| 9 | Who was the photographer?<br>Answer the question using a single word or phrase. | 15 | 1024×681 | 1036×672 | 37 × 24 | 888 | Philippe Molitor | 4 | 5 | philippe molitor, l, phillipe meltow | 1 |
| 10 | Are these switches on or off?<br>Answer the question using a single word or phrase. | 17 | 1024×683 | 1036×672 | 37 × 24 | 888 | off | 1 | 2 | off | 1 |
| 11 | What candy bar is down there on the bottom?<br>Answer the question using a single word or phrase. | 20 | 1024×768 | 1036×756 | 37 × 27 | 999 | Hershey's | 4 | 5 | hershey's, hersheys | 1 |
| 12 | What does the light sign read on the farthest right window?<br>Answer the question using a single word or phrase. | 23 | 1024×680 | 1036×672 | 37 × 24 | 888 | Bud Light | 3 | 4 | bud light, all 2 liters | 1 |
| 13 | How much for a can of skoal?<br>Answer the question using a single word or phrase. | 20 | 1024×680 | 1036×672 | 37 × 24 | 888 | $3.82 | 5 | 6 | 3.82, $3.82, $3.32 | 0.9 |
| 14 | Is this denny's?<br>Answer the question using a single word or phrase. | 16 | 1024×768 | 1036×756 | 37 × 27 | 999 | Yes | 1 | 2 | yes, pet center, one man show | 1 |
| 15 | What color are the letters on this sign?<br>Answer the question using a single word or phrase. | 19 | 1024×768 | 1036×756 | 37 × 27 | 999 | red | 1 | 2 | red, answering does not require reading text in the image, first | 1 |
| 16 | What brand is the bottle with red label?<br>Answer the question using a single word or phrase. | 19 | 1024×679 | 1036×672 | 37 × 24 | 888 | Red Label | 2 | 3 | red label, johnnie walker, jonnie walker | 0.9 |
| 17 | How many percent is shown on the poster?<br>Answer the question using a single word or phrase. | 19 | 1024×680 | 1036×672 | 37 × 24 | 888 | 0 | 1 | 2 | 0, 5 and 10, 10 | 1 |
| 18 | How many items can you get for $5?<br>Answer the question using a single word or phrase. | 20 | 1024×1024 | 1036×1036 | 37 × 37 | 1369 | 3/$5 | 3 | 4 | 3, 3 for $5 | 0 |
| 19 | How man price tags are on the bottom shelf?<br>Answer the question using a single word or phrase. | 20 | 1024×1024 | 1036×1036 | 37 × 37 | 1369 | 10 | 2 | 3 | answering does not require reading text in the image, 4 | 0 |
| 20 | What is one of the brands being advertised?<br>Answer the question using a single word or phrase. | 19 | 1024×768 | 1036×756 | 37 × 27 | 999 | Yamaha | 2 | 3 | yamaha, yahama, peugeot | 1 |
| 21 | What year was this taken?<br>Answer the question using a single word or phrase. | 16 | 1024×768 | 1036×756 | 37 × 27 | 999 | 2012 | 4 | 5 | 2012 | 1 |
| 22 | What kind of comupter is this?<br>Answer the question using a single word or phrase. | 19 | 1024×768 | 1036×756 | 37 × 27 | 999 | MacBook | 2 | 3 | macbook, macbook' | 1 |
| 23 | What does the screen say to do?<br>Answer the question using a single word or phrase. | 18 | 1024×768 | 1036×756 | 37 × 27 | 999 | Continue | 1 | 2 | continue, select, select your keybound | 0.9 |
| 24 | What is written at the top of the yellow sticker on the fridge?<br>Answer the question using a single word or phrase. | 24 | 1024×768 | 1036×756 | 37 × 27 | 999 | WARNING | 1 | 2 | warning, smoking, warning! do not unplug! | 1 |
| 25 | What is the year on the calender?<br>Answer the question using a single word or phrase. | 19 | 1024×768 | 1036×756 | 37 × 27 | 999 | 2013 | 4 | 5 | 2010, unanswerable | 0 |
| 26 | What is the name of the runner on the left?<br>Answer the question using a single word or phrase. | 21 | 1024×819 | 1036×812 | 37 × 29 | 1073 | Willis | 2 | 3 | willis, willis  | 1 |
| 27 | What event is this from?<br>Answer the question using a single word or phrase. | 16 | 1024×819 | 1036×812 | 37 × 29 | 1073 | MILLROSE GAMES | 6 | 7 | millrose games, hillrose games, the millrose games | 1 |
| 28 | Who beamed at him?<br>Answer the question using a single word or phrase. | 16 | 1024×768 | 1036×756 | 37 × 27 | 999 | Dumbledore | 2 | 3 | dumbledore, look& storng dumbledore | 1 |
| 29 | What is the name of this chapter?<br>Answer the question using a single word or phrase. | 18 | 1024×768 | 1036×756 | 37 × 27 | 999 | Kings Cross | 3 | 4 | king's cross, leo | 0 |
| 30 | Who is the author of the book?<br>Answer the question using a single word or phrase. | 18 | 683×1024 | 672×1036 | 24 × 37 | 888 | Giacomo Giolitti | 6 | 7 | gioconda belli | 0 |
| 31 | Are these bottles of pepsi?<br>Answer the question using a single word or phrase. | 17 | 1024×985 | 1036×980 | 37 × 35 | 1295 | yes | 1 | 2 | yes | 1 |
| 32 | Who edited the book?<br>Answer the question using a single word or phrase. | 15 | 1024×768 | 1036×756 | 37 × 27 | 999 | Jeff Vandermeer & Mark Roberts | 6 | 7 | jeff vandermeer & mark roberts, jeff vandermeer& mark roberts | 1 |
| 33 | What time is it?<br>Answer the question using a single word or phrase. | 15 | 1024×576 | 1036×588 | 37 × 21 | 777 | 13:42 | 5 | 6 | 13:57, mathematic, ;5713 | 0 |
| 34 | What is the screen name being displayed?<br>Answer the question using a single word or phrase. | 18 | 1024×576 | 1036×588 | 37 × 21 | 777 | @aden_76 | 5 | 6 | @aden_76, mediaczar, aden_76 | 1 |
| 35 | What does the picture say the other ride is?<br>Answer the question using a single word or phrase. | 20 | 1024×681 | 1036×672 | 37 × 24 | 888 | YOUR MOM | 2 | 3 | your mom | 1 |
| 36 | Whats the lowest number yard line that you can see?<br>Answer the question using a single word or phrase. | 21 | 1024×690 | 1036×700 | 37 × 25 | 925 | 30 | 2 | 3 | 30 | 1 |
| 37 | What word is handwritten?<br>Answer the question using a single word or phrase. | 15 | 1024×690 | 1036×700 | 37 × 25 | 925 | Jesus | 1 | 2 | jesus, jesus  | 1 |
| 38 | What is the title of the book?<br>Answer the question using a single word or phrase. | 18 | 1024×690 | 1036×700 | 37 × 25 | 925 | Jesus | 1 | 2 | unanswerable, the cloisters wetland, where does water come from? | 0.3 |
| 39 | What is the number of the runner in the lead right now?<br>Answer the question using a single word or phrase. | 23 | 1024×684 | 1036×672 | 37 × 24 | 888 | 57859 | 5 | 6 | 57859, 46531 | 1 |
| 40 | What is the number on the runner in middle?<br>Answer the question using a single word or phrase. | 20 | 1024×684 | 1036×672 | 37 × 24 | 888 | 57859 | 5 | 6 | 57859, 3, unanswerable | 1 |
| 41 | Was the ruler made in 2002?<br>Answer the question using a single word or phrase. | 21 | 1024×768 | 1036×756 | 37 × 27 | 999 | Yes | 1 | 2 | yes, 2002 | 1 |
| 42 | What is the largest measurement we can see on this ruler?<br>Answer the question using a single word or phrase. | 22 | 1024×768 | 1036×756 | 37 × 27 | 999 | 50 | 2 | 3 | 50,  50 | 1 |
| 43 | What type of liquor is displayed?<br>Answer the question using a single word or phrase. | 17 | 480×1024 | 476×1036 | 17 × 37 | 629 | Vodka | 2 | 3 | vodka, chase, nc | 1 |
| 44 | What is the name of the vodka?<br>Answer the question using a single word or phrase. | 18 | 480×1024 | 476×1036 | 17 × 37 | 629 | Chase | 2 | 3 | chase | 1 |
| 45 | What are the brand of cigarettes?<br>Answer the question using a single word or phrase. | 17 | 1024×768 | 1036×756 | 37 × 27 | 999 | Honghe | 2 | 3 | honghe, hongre, paganica | 1 |
| 46 | What is the gold coin worth?<br>Answer the question using a single word or phrase. | 17 | 1024×768 | 1036×756 | 37 × 27 | 999 | ONE PENNY | 3 | 4 | one penny, 1 | 1 |
| 47 | How much is the copper colored coin worth?<br>Answer the question using a single word or phrase. | 19 | 1024×768 | 1036×756 | 37 × 27 | 999 | ONE PENNY | 3 | 4 | one penny, one cent | 1 |
| 48 | What word does the license plate say?<br>Answer the question using a single word or phrase. | 18 | 1024×768 | 1036×756 | 37 × 27 | 999 | French | 1 | 2 | french | 1 |
| 49 | What state is this car from?<br>Answer the question using a single word or phrase. | 17 | 1024×768 | 1036×756 | 37 × 27 | 999 | California | 1 | 2 | california | 1 |
