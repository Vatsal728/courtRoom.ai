import type { LegalCategory, Language, LanguageInfo } from './types';

export const LANGUAGES: readonly LanguageInfo[] = [
  { code: 'en', name: 'English', native: 'English' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
  { code: 'gu', name: 'Gujarati', native: 'ગુજરાતી' },
  { code: 'mr', name: 'Marathi', native: 'मराठी' },
  { code: 'ta', name: 'Tamil', native: 'தமிழ்' },
  { code: 'te', name: 'Telugu', native: 'తెలుగు' },
  { code: 'kn', name: 'Kannada', native: 'ಕನ್ನಡ' },
  { code: 'bn', name: 'Bengali', native: 'বাংলা' },
  { code: 'ml', name: 'Malayalam', native: 'മലയാളം' },
  { code: 'pa', name: 'Punjabi', native: 'ਪੰਜਾਬੀ' },
  { code: 'ur', name: 'Urdu', native: 'اردو' },
] as const;

export function getLanguageInfo(code: Language): LanguageInfo {
  return LANGUAGES.find(l => l.code === code) || { code, name: code, native: code };
}

export function searchLanguages(query: string): LanguageInfo[] {
  const q = query.toLowerCase().trim();
  if (!q) return [...LANGUAGES];
  return LANGUAGES.filter(
    l =>
      l.name.toLowerCase().includes(q) ||
      l.native.toLowerCase().includes(q) ||
      l.code.toLowerCase().includes(q)
  );
}

export function detectLanguage(text: string): Language {
  // Simple heuristic detection
  const lowerText = text.toLowerCase();
  if (lowerText.match(/[\u0a80-\u0aff]/)) return 'gu'; // Gujarati range
  if (lowerText.match(/[\u0900-\u097f]/)) return 'hi'; // Hindi range
  return 'en';
}

export function detectCategory(text: string): LegalCategory {
  const lowerText = text.toLowerCase();
  if (lowerText.includes('consumer') || lowerText.includes('bought') || lowerText.includes('refund') || lowerText.includes('product')) {
    return 'consumer';
  }
  if (lowerText.includes('wage') || lowerText.includes('salary') || lowerText.includes('employee') || lowerText.includes('work')) {
    return 'labour';
  }
  if (lowerText.includes('rent') || lowerText.includes('tenant') || lowerText.includes('landlord') || lowerText.includes('flat')) {
    return 'rent';
  }
  if (lowerText.includes('rti') || lowerText.includes('information') || lowerText.includes('public authority')) {
    return 'rti';
  }
  if (lowerText.includes('fir') || lowerText.includes('police') || lowerText.includes('crime') || lowerText.includes('stolen')) {
    return 'criminal';
  }
  if (lowerText.includes('cyber') || lowerText.includes('hacked') || lowerText.includes('scam') || lowerText.includes('online fraud')) {
    return 'cyber';
  }
  return 'general';
}
