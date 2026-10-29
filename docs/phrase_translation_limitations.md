# phrase_translation.py Limitations (Workstream 6)

Current state: `phrase_translation.py` does NOT exist in the repository. The existing pipeline is `app.py`/`train.py`, a customer-churn scikit-learn model (LogisticRegression + ColumnTransformer). There is zero sign-language infrastructure.

Limitations of a hypothetical phrase-lookup approach (current state):
- No grammatical analysis: phrase lookup ignores English syntax, SL grammar rules (topic-comment, classifiers, spatial agreement), and non-manual markers.
- No UD parsing: no dependency parsing (spaCy/Stanza) to determine S/O/V relationships before SL gloss mapping.
- No SOV transformation: English SVO is not converted to ISL SOV; spatial indexing and classifier verbs are missing.
- No HamNoSys/SiGML output: no structured animation sequence (timing, holds, transitions, facial expressions).
- No fallback chain: validated animation → procedural gloss → fingerspelling → unavailable is not implemented.
- No per-language grammar plugins: ISL, ASL, BSL rules are not modular.
- No gloss-correction editor: user cannot edit gloss sequence before animation.
- No HamNoSys string or SiGML XML generation for avatar animation.

Next step: Design grammar-aware pipeline and implement `grammar_translator.py`.
