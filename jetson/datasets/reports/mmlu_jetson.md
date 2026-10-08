# mmlu_jetson

Tokenizer and chat template: `Qwen/Qwen2.5-3B-Instruct`.

| | |
|---|---|
| Questions | 14,042 |
| Prompt tokens (chat template + few-shot context), avg (min–max) | 738 (282–3,138) |
| Question tokens (the question alone), avg (min–max) | 59 (2–951) |
| Reference answer words / tokens, avg (min–max) | 1.0 (1–1) / 1.0 (1–1) |
| Scoring | the first A-D letter of the answer; accuracy weighted by subject size |

## Example (first question)

```
The following are multiple choice questions (with answers) about miscellaneous. Answer the last question with the letter of the correct option only.
How many axles does a standard automobile have?
A. one
B. two
C. four
D. eight
Answer: B

What place is named in the title of the 1979 live album by rock legends Cheap Trick?
A. Budapest
B. Budokan
C. Bhutan
D. Britain
Answer: B

Who is the shortest man to ever win an NBA slam dunk competition?
A. Anthony 'Spud' Webb
B. Michael 'Air' Jordan
C. Tyrone 'Muggsy' Bogues
D. Julius 'Dr J' Erving
Answer: A

What is produced during photosynthesis?
A. hydrogen
B. nylon
C. oxygen
D. light
Answer: C

Which of these songs was a Top 10 hit for the rock band The Police?
A. 'Radio Ga-Ga'
B. 'Ob-la-di Ob-la-da'
C. 'De Do Do Do De Da Da Da'
D. 'In-a-Gadda-Da-Vida'
Answer: C

An important source of information on the credit rating of retail businesses is
A. the Retail Merchants Association
B. the local chamber of commerce
C. Dun & Bradstreet, Inc.
D. the United States Retail Credit Association
Answer:
```

Reference: C
