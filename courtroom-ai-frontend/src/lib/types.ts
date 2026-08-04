export type LegalCategory = 'consumer' | 'labour' | 'rent' | 'rti' | 'criminal' | 'cyber' | 'general';

export type Language =
  | 'en'
  | 'hi'
  | 'gu'
  | 'mr'
  | 'ta'
  | 'te'
  | 'kn'
  | 'bn'
  | 'ml'
  | 'pa'
  | 'ur';

export interface LanguageInfo {
  code: Language;
  name: string;
  native: string;
}

export interface LegalSection {
  id: string;
  document_id: string;
  section_number: string;
  title: string;
  content: string;
  plain_language: string;
  keywords: string[];
  category: LegalCategory;
  next_steps?: string;
  jurisdiction?: string;
  document_name?: string;
  document_short_name?: string;
}

export interface RetrievalResult {
  section: LegalSection;
  score: number;
  matchedTerms: string[];
}
