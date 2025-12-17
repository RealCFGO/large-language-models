# LLM Tokenisering Visualizer

[Also available in English](README.md)

Dette repository indeholder en lille, fokuseret Streamlit-applikation designet til at undersøge, hvordan store sprogmodeller (LLMs) tokeniserer tekst og foretager next-token forudsigelser. Projektet er beregnet som et teknisk og uddannelsesmæssigt prototypeværktøj snarere end et fuldt produktionssystem, med fokus på gennemsigtighed og inspektion af kernemekanismer.

## Omfang og Designvalg

Applikationen bruger en **fast model**, GPT-2 (small), for at sikre reproducerbarhed og undgå forvirring fra sammenligning af forskellige modelarkitekturer. GPT-2 bruges udelukkende til next-token forudsigelse, da det er en kausal sprogmodel og derfor egnet til sekventiel generering.

To tokeniseringsstrategier understøttes for sammenligning:

- **Byte Pair Encoding (BPE)** via `GPT2TokenizerFast`  
- **WordPiece** via `BertTokenizerFast`

WordPiece bruges kun til analytisk og visuel sammenligning. Alle sandsynlighedsberegninger og forudsigelser udføres med GPT-2’s BPE-tokenizer for at bevare modelkompatibilitet.

## Kernefunktionalitet

Applikationen indeholder to tæt sammenkoblede komponenter:

### Tokeniseringsinspektion

Inputtekst tokeniseres og vises direkte i interfacet. Hver token vises inline med visuel differentiering baseret på dens rolle, inklusiv ordstart, subword-fortsættelser, tegnsætning og specialtegn. Token-indekser og token-IDs er tilgængelige i tabelform, sammen med et kort, der viser, hvor mange tokens hvert ord er delt i. Dette gør forskelle i tokeniseringsgranularitet umiddelbart observerbare, især for længere eller morfologisk komplekse ord.

### Next-Token Forudsigelse

For en given inputsekvens beregner modellen logits for næste token-position. Disse logits omdannes til sandsynligheder via softmax, og de top-N mest sandsynlige tokens vises. Fordelingen vises både numerisk og grafisk, hvilket tydeligt viser, at generering er probabilistisk snarere end deterministisk.

To sampling-relaterede parametre kan justeres:

- **Temperature**, som skalerer logits og styrer, hvor koncentreret eller diffus sandsynlighedsfordelingen er.  
- **Repetition penalty**, som reducerer sandsynligheden for tokens, der allerede forekommer i input.

Justering af disse parametre opdaterer forudsigelserne i realtid, hvilket giver direkte indsigt i deres effekt på modeladfærd.

## Teknisk Struktur

Kodebasen er struktureret omkring klar separation af funktioner:

- Model- og tokenizer-loading er isoleret og cached for at undgå redundant beregning.  
- Tokeniseringslogik, token-klassifikation og visualisering er indkapslet i dedikerede funktioner.  
- Sandsynlighedsberegning og sampling-logik holdes adskilt fra UI-laget.  
- Visualisering håndteres via Plotly, mens Streamlit kun bruges som et letvægts-interface.

## Tiltenkt Brug

Projektet er beregnet til analytiske og uddannelsesmæssige formål, især i sammenhænge, hvor forståelse af parsing, subword-tokenisering og next-token forudsigelse er vigtigere end rå genereringskvalitet. Det egner sig til demonstrationer, kursusarbejde og eksplorativ analyse af LLM-adfærd på token-niveau.
