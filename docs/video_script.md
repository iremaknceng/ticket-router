# 2-minute demo video script (only the first 2:00 are watched)

Record the screen with voice (OBS / Loom / Zoom). App running live: `python app.py` locally,
or `python app.py --share` in Colab.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | Title slide | "A home-appliance brand gets thousands of Turkish and English support messages a day. My app routes each one to the right team instantly, and sends unclear ones to a human." |
| 0:15–0:35 | App: type `bulaşık makinem su boşaltmıyor, model DW-6534` | "The message is normalized, the model code is extracted, and the INT8 distilled BERT routes it to Technical Service with X % confidence. Below, all models side by side with their latency." |
| 0:35–0:50 | Type `buzdolabim garantisi bitti mi acaba??` | "Typos and missing Turkish letters are fixed by edit distance - here 'buzdolabim'. It goes to Warranty." |
| 0:50–1:05 | Type `who won the champions league last night` (difficult / out-of-scope), then `Merhaba!` | "An out-of-scope question goes to a human agent. A greeting gets a rule-based ELIZA reply." |
| 1:05–1:15 | Type your failure example from the Limitations slide | "Here it fails: … because …" |
| 1:15–1:45 | Architecture slide, then Results slide | "Preprocessing with regex and edit distance, four models, the INT8 student runs the app. On the same test set: accuracy …, OOS recall …, size from … MB to … MB, latency from … to … ms." |
| 1:45–2:00 | Findings slide | "The quantized student keeps … of the teacher's accuracy at … the latency. Next step: real Turkish tickets." |

Upload to YouTube (Unlisted) or Google Drive ("anyone with the link"); open the link in a
private browser window before submitting.
