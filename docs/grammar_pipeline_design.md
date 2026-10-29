# Grammar-Aware Text-to-Sign Pipeline Design

## Pipeline
English text -> spaCy UD parse -> SL gloss rules -> HamNoSys/SiGML -> animation params

## Components
1. Linguistic model: SL grammar rules (topic-comment, classifiers, spatial agreement)
2. Pipeline stages: UD parse -> gloss rules -> HamNoSys -> SiGML -> animation params
3. Resources: SignBank, HamNoSys, SiGML
4. ISL-specific: SOV, spatial indexing, non-manual markers
5. Modular grammar plugins per language (ISLGrammar, ASLGrammar, BSLGrammar)
6. Output: structured animation sequence (timing, holds, transitions, facial expr)
7. Fallback chain: validated animation -> procedural from gloss -> fingerspell -> unavailable
8. Editor: user corrects gloss sequence before animation
