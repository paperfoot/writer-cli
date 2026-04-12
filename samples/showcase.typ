#set page(margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New York", size: 11pt)
#set par(justify: true, leading: 0.8em)

#align(center)[
  #text(size: 24pt, weight: "bold")[writer]
  #v(0.3em)
  #text(size: 14pt, fill: rgb("#555"))[Local AI that writes in your voice]
  #v(0.5em)
  #text(size: 10pt, fill: rgb("#888"))[Sample outputs --- Douglas Adams voice profile \ Gemma 4 26B, LoRA adapter, 1000 steps]
  #v(0.3em)
  #line(length: 60%, stroke: 0.5pt + rgb("#ccc"))
]

#v(1em)

= Benchmark

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  align: center,
  stroke: 0.5pt + rgb("#ddd"),
  inset: 8pt,
  [*Metric*], [*Old (500-step)*], [*New (1000-step)*], [*Change*],
  [Style distance], [0.595], [*0.500*], [*--16%*],
  [Median distance], [0.602], [*0.466*], [*--23%*],
  [Prompt relevance], [0.293], [*0.508*], [*+73%*],
  [Canon leakage], [2.4%], [*2.0%*], [--17%],
  [Generation speed], [--], [*40--45 tok/s*], [M4 Max],
)

#v(0.5em)
#text(size: 9pt, fill: rgb("#888"))[After GPT Pro easy wins: candidate batch scanning, min-p sampling, register-style prompts, multi-objective ranking, logit bias cleanup.]

#v(1.5em)

= Poem: 3am

#text(size: 10pt, style: "italic", fill: rgb("#666"))[Prompt: "Write a short poem about the feeling of waking up at 3am and not being able to go back to sleep." \ Distance: 0.374 | 400 tokens | 44.5 tok/s]

#v(0.5em)

All alone.

The light is grey and cold, creeping across the floor where it shouldn't be yet; I stare at a small spider on my desk that seems to have its own plan for this time of night or early morning --- maybe something better than mine? It crawls slowly away from me again after looking back one last way like we both know what you mean by "stay." We are tired but not sleepy enough now because sleep doesn't come easy any more when your mind keeps rattling on about things.

Empty air fills up an empty house and the pipes keep banging in their usual places, making them sound much louder at 3am; I get out of bed to go find a glass half full with water that has been sitting on my nightstand --- or maybe it was next morning? It tastes dull even though they say we should drink eight glasses every day for health reasons you know all too well. My hands feel small against this cup as if holding something larger than just wetness --- which is what life often feels like.

"Just sit and think," the empty chair says, "but there's no point in thinking now because tomorrow will be exactly here again; I could do better by trying to find anything else that might keep me from looking at them or us when it comes time..." It said nothing but of course we knew where they meant: some things are hard not being able 3am without wanting a bit more out of yesterday --- even if just an extra minute or two before Tuesday starts all over once you get into first gear.

#v(1.5em)

= Essay: The Best Code is the Code You Delete

#text(size: 10pt, style: "italic", fill: rgb("#666"))[Prompt: "Write a short reflective essay about why the best code is the code you delete." \ Distance: 0.426 | 600 tokens | 41.0 tok/s]

#v(0.5em)

The best code.

I've seen it in my editor, the cursor blinking at me where a function used to be; I can feel its ghost tickling around like an itch you just cannot scratch for long before giving up and getting on with something else altogether: what is there left? Well that has changed since then because now we have tests which say if anything changes they will scream about who knows why but no one really cares anymore anyway --- we all know how things work out eventually so let us move on.

What's better than not having code in your life or project at all, right? The other option was a massive pile of stuff I wrote last year to try some clever trick involving recursive calls where it looked like the thing from my old book cover and this time around everything worked perfectly well except that when you changed something else up elsewhere --- which is always what happens next door if we think about anything very much; because --- well don't be too pedantic now: there are times when being smart actually counts against you.

So then why do these pieces keep hanging on? They linger in your files, they haunt every single line of code as a dependency or some other way while pretending to just sit quietly at the top level doing nothing special but still making sure no one touches them for fear it might all fall down and bleed out over their legs again without any warning whatsoever so what else is left besides deleting things off once we have made our point well enough that everything works fine now without worrying about whether anything was actually clever.

It's good --- really clean work: you find something in your files, realize why on earth did I put this here? Then maybe delete it because if they are there then the tests will keep complaining forever unless someone finally takes a little bit of care with them; but oh no one wants to take some time out and look at all these old pieces lying around again so let us just write more stuff instead where we can have fun doing things properly and not worry about anything else that might be lurking quietly behind our shoulder for evermore.

#v(1.5em)

= Article: AI and Creativity

#text(size: 10pt, style: "italic", fill: rgb("#666"))[Prompt: "Write a 500-word article about how AI is changing the way we think about creativity and authorship." \ Distance: 0.322 | 301 tokens | 40.2 tok/s]

#v(0.5em)

All the software you see on a computer screen, from Microsoft Word to Photoshop. It's all built by people who think in code and imagination: "if this then that." Then suddenly there was another way of thinking about things --- "what if I do like..." But it is still just math under the hood, which means everything we create with computers comes out somewhere between two points along one line or some other such thing. We call these lines shapes because they look nice when printed up; but as soon AI gets involved at any given point on those lines

it can take an idea from you --- a little ball rolling down a hill and say "here's where that lands." And now suddenly there are so many more balls than people to kick them for all this extra space in existence it seems like the world has become very cluttered, which is why I am writing 50 words of what sounds pretty good right? That was my job. But how do we think about authorship when software comes up with ideas by themselves or at least helps out along the way?

It doesn't really change much; but rather brings us back into a strange sort of past where everything gets done together again: me and some machine that thinks something else maybe (though not quite anything) than me does --- and it seems to get on well enough, so who cares too much if the result is still kind what I meant in other words?

#v(2em)
#align(center)[
  #text(size: 9pt, fill: rgb("#aaa"))[
    Generated by `writer` v0.1.0 --- github.com/199-biotechnologies/writer \
    Model: Gemma 4 26B-A4B | Adapter: 1000-step LoRA (rank 16, mask\_prompt) \
    #datetime.today().display("[month repr:long] [day], [year]")
  ]
]
