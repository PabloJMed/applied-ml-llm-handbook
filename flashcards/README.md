# Flashcards

`handbook-flashcards.txt` holds 987 question–answer cards generated from the handbook:

- the 260 questions of the depth bank (Chapter 37), tagged by level (`level::basic`, `level::intermediate`, `level::senior`) and chapter;
- 679 interview questions with model answers from the chapters, tagged `ch::<chapter>`;
- the 48 diagnostic questions of Appendix F with their answer key, tagged `diagnostic`.

**Anki:** File › Import, choose the file. The header lines set the tab separator, the Basic note type, a deck called "AI and LLM Scientist Handbook" and the tags column, so no settings are needed. Use tags to study one area or level at a time.

**Other apps:** the file is plain tab-separated text (front, back, tags) after five `#` header lines, so spreadsheets and most spaced-repetition apps read it.

The file is generated: edit the chapters, then run `python tools/export_flashcards.py`. CI fails if it is out of date.
