"""
Unicode Devanagari normalization for OCR output
"""

import re
import unicodedata

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False
    print("Warning: indic-transliteration not available")


class UnicodeNormalizer:
    """Normalize and clean Sanskrit/Devanagari text"""

    def __init__(self):
        self.indic_available = INDIC_TRANSLITERATION_AVAILABLE

    def normalize(self, text, input_format='devanagari'):
        """
        Complete normalization pipeline

        Args:
            text: Input text (may be romanized or Devanagari)
            input_format: 'devanagari', 'iast', 'itrans', 'auto'

        Returns:
            Normalized Unicode Devanagari text
        """
        if not text or not text.strip():
            return ""

        # Auto-detect format if needed
        if input_format == 'auto':
            input_format = self._detect_format(text)

        # Convert to Devanagari if romanized
        if input_format != 'devanagari':
            text = self.romanized_to_devanagari(text, input_format)

        # Clean and normalize
        text = self.clean_text(text)
        text = self.fix_unicode(text)
        text = self.normalize_characters(text)

        return text

    def _detect_format(self, text):
        """Auto-detect if text is Devanagari or romanized"""
        # Check if text contains Devanagari characters
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        ascii_chars = sum(1 for c in text if ord(c) < 128)

        if devanagari_chars > ascii_chars:
            return 'devanagari'
        else:
            # Assume IAST for romanized Sanskrit
            return 'iast'

    def romanized_to_devanagari(self, text, scheme='iast'):
        """
        Convert romanized Sanskrit to Devanagari

        Args:
            text: Romanized text
            scheme: 'iast', 'itrans', 'velthuis', 'wx', etc.

        Returns:
            Devanagari text
        """
        if not self.indic_available:
            print("Warning: indic-transliteration not available, returning original")
            return text

        # Map scheme names
        scheme_map = {
            'iast': sanscript.IAST,
            'itrans': sanscript.ITRANS,
            'velthuis': sanscript.VELTHUIS,
            'wx': sanscript.WX,
            'harvard-kyoto': sanscript.HK,
            'slp1': sanscript.SLP1,
        }

        input_scheme = scheme_map.get(scheme.lower(), sanscript.IAST)

        try:
            devanagari = transliterate(text, input_scheme, sanscript.DEVANAGARI)
            return devanagari
        except Exception as e:
            print(f"Transliteration error: {e}")
            return text

    def devanagari_to_romanized(self, text, scheme='iast'):
        """
        Convert Devanagari to romanized Sanskrit

        Args:
            text: Devanagari text
            scheme: Output romanization scheme

        Returns:
            Romanized text
        """
        if not self.indic_available:
            print("Warning: indic-transliteration not available")
            return text

        scheme_map = {
            'iast': sanscript.IAST,
            'itrans': sanscript.ITRANS,
            'velthuis': sanscript.VELTHUIS,
            'wx': sanscript.WX,
            'harvard-kyoto': sanscript.HK,
            'slp1': sanscript.SLP1,
        }

        output_scheme = scheme_map.get(scheme.lower(), sanscript.IAST)

        try:
            romanized = transliterate(text, sanscript.DEVANAGARI, output_scheme)
            return romanized
        except Exception as e:
            print(f"Transliteration error: {e}")
            return text

    def clean_text(self, text):
        """Remove OCR artifacts and clean text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common OCR errors
        text = text.replace('|', '')  # Often misread as pipe
        text = text.replace('_', '')
        text = text.replace('~', '')

        # Remove non-Devanagari, non-punctuation characters (except spaces)
        # Devanagari range: U+0900 to U+097F
        # Keep common punctuation: । ॥ ॰
        cleaned = []
        for char in text:
            code = ord(char)
            if (0x0900 <= code <= 0x097F or  # Devanagari
                char in ' \n\t।॥॰' or  # Whitespace and Devanagari punctuation
                char in '.,;:!?-()[]{}'):  # Common punctuation
                cleaned.append(char)

        text = ''.join(cleaned)

        # Remove repeated punctuation
        text = re.sub(r'([।॥])\1+', r'\1', text)

        return text.strip()

    def fix_unicode(self, text):
        """Fix common Unicode normalization issues"""
        # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
        text = unicodedata.normalize('NFC', text)

        # Also try NFKC (Compatibility Decomposition, followed by Canonical Composition)
        # This handles various equivalent representations
        text = unicodedata.normalize('NFKC', text)

        return text

    def normalize_characters(self, text):
        """Normalize specific Devanagari characters"""
        # Normalize anusvara and chandrabindu
        # ं (U+0902) and ँ (U+0901)

        # Normalize visarga variations
        # ः (U+0903)

        # Fix common ligature issues
        replacements = {
            # Common OCR confusions
            'ो': 'ो',  # Normalize vowel sign O
            'ौ': 'ौ',  # Normalize vowel sign AU
            'ं': 'ं',   # Normalize anusvara
            'ः': 'ः',   # Normalize visarga
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Fix broken conjuncts (half-letters)
        # This is complex and may require more sophisticated handling

        return text

    def fix_word_spacing(self, text):
        """Fix word spacing issues from OCR"""
        # Remove space before combining marks (matras, etc.)
        # Devanagari combining marks: U+0901 to U+0903, U+093C, U+093E to U+094F, U+0951 to U+0957
        combining_marks = '[\u0901-\u0903\u093c\u093e-\u094f\u0951-\u0957]'
        text = re.sub(f' ({combining_marks})', r'\1', text)

        # Fix spaces around Devanagari punctuation
        text = re.sub(r' ([।॥])', r'\1', text)
        text = re.sub(r'([।॥]) ', r'\1 ', text)

        return text

    def split_sentences(self, text):
        """Split text into sentences using Devanagari punctuation"""
        # Split on danda (।) and double danda (॥)
        sentences = re.split(r'[।॥]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def tokenize_words(self, text):
        """Simple word tokenization"""
        # Split on whitespace
        words = text.split()
        return [w.strip() for w in words if w.strip()]


class SanskritTextProcessor:
    """High-level Sanskrit text processing"""

    def __init__(self):
        self.normalizer = UnicodeNormalizer()

    def process_ocr_output(self, ocr_text, input_format='auto'):
        """
        Process raw OCR output

        Args:
            ocr_text: Raw text from OCR
            input_format: Expected format of input

        Returns:
            Dictionary with processed text and metadata
        """
        # Normalize
        normalized = self.normalizer.normalize(ocr_text, input_format)

        # Split into sentences
        sentences = self.normalizer.split_sentences(normalized)

        # Tokenize
        words = self.normalizer.tokenize_words(normalized)

        # Get romanized version
        romanized = self.normalizer.devanagari_to_romanized(normalized, 'iast')

        return {
            'original': ocr_text,
            'normalized': normalized,
            'sentences': sentences,
            'words': words,
            'word_count': len(words),
            'romanized': romanized,
        }


# Example transliteration mappings for manual conversion (if library not available)
IAST_TO_DEVANAGARI = {
    'a': 'अ', 'ā': 'आ', 'i': 'इ', 'ī': 'ई', 'u': 'उ', 'ū': 'ऊ',
    'ṛ': 'ऋ', 'ṝ': 'ॠ', 'ḷ': 'ऌ', 'ḹ': 'ॡ',
    'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
    'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ṅ': 'ङ',
    'c': 'च', 'ch': 'छ', 'j': 'ज', 'jh': 'झ', 'ñ': 'ञ',
    'ṭ': 'ट', 'ṭh': 'ठ', 'ḍ': 'ड', 'ḍh': 'ढ', 'ṇ': 'ण',
    't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
    'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म',
    'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व',
    'ś': 'श', 'ṣ': 'ष', 's': 'स', 'h': 'ह',
    'ṃ': 'ं', 'ḥ': 'ः', '।': '।', '॥': '॥',
}


if __name__ == "__main__":
    print("Testing Unicode Normalizer...")

    normalizer = UnicodeNormalizer()
    processor = SanskritTextProcessor()

    # Test cases
    test_cases = [
        ("rāmaḥ vanam gacchati", "iast"),
        ("राम: वनं गच्छति", "devanagari"),
    ]

    for text, fmt in test_cases:
        print(f"\nInput ({fmt}): {text}")
        result = processor.process_ocr_output(text, fmt)
        print(f"Normalized: {result['normalized']}")
        print(f"Romanized: {result['romanized']}")
        print(f"Words: {result['words']}")

    print("\nNormalization module ready!")

