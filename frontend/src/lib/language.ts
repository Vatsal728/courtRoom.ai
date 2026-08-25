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

// Native-language domain badges (RTI stays as-is in every language).
const DOMAIN_LABELS: Record<Language, Record<LegalCategory, string>> = {
  en: { criminal: 'CRIMINAL', consumer: 'CONSUMER', labour: 'LABOUR', rent: 'RENT', rti: 'RTI', cyber: 'CYBER', civil: 'CIVIL', general: 'GENERAL' },
  hi: { criminal: 'अपराध', consumer: 'उपभोक्ता', labour: 'श्रम', rent: 'किराया', rti: 'RTI', cyber: 'साइबर', civil: 'दीवानी', general: 'सामान्य' },
  gu: { criminal: 'ગુનાખોરી', consumer: 'ઉપભોક્તા', labour: 'શ્રમ', rent: 'ભાડું', rti: 'RTI', cyber: 'સાયબર', civil: 'સિવિલ', general: 'સામાન્ય' },
  mr: { criminal: 'गुन्हा', consumer: 'ग्राहक', labour: 'श्रम', rent: 'भाडे', rti: 'RTI', cyber: 'सायबर', civil: 'दिवाणी', general: 'सामान्य' },
  ta: { criminal: 'குற்றம்', consumer: 'நுகர்வோர்', labour: 'தொழிலாளர்', rent: 'வாடகை', rti: 'RTI', cyber: 'சைபர்', civil: 'சிவில்', general: 'பொது' },
  te: { criminal: 'నేరం', consumer: 'వినియోగదారుడు', labour: 'కార్మిక', rent: 'అద్దె', rti: 'RTI', cyber: 'సైబర్', civil: 'సివిల్', general: 'సాధారణ' },
  kn: { criminal: 'ಅಪರಾಧ', consumer: 'ಗ್ರಾಹಕ', labour: 'ಕಾರ್ಮಿಕ', rent: 'ಬಾಡಿಗೆ', rti: 'RTI', cyber: 'ಸೈಬರ್', civil: 'ಸಿವಿಲ್', general: 'ಸಾಮಾನ್ಯ' },
  bn: { criminal: 'অপরাধ', consumer: 'ভোক্তা', labour: 'শ্রম', rent: 'ভাড়া', rti: 'RTI', cyber: 'সাইবার', civil: 'দেওয়ানি', general: 'সাধারণ' },
  ml: { criminal: 'കുറ്റം', consumer: 'ഉപഭോക്താവ്', labour: 'തൊഴിലാളി', rent: 'വാടക', rti: 'RTI', cyber: 'സൈബർ', civil: 'സിവിൽ', general: 'പൊതു' },
  pa: { criminal: 'ਅਪਰਾਧ', consumer: 'ਖਪਤਕਾਰ', labour: 'ਮਜ਼ਦੂਰੀ', rent: 'ਕਿਰਾਇਆ', rti: 'RTI', cyber: 'ਸਾਈਬਰ', civil: 'ਸਿਵਲ', general: 'ਆਮ' },
  ur: { criminal: 'جرم', consumer: 'صارف', labour: 'مزدوری', rent: 'کرایہ', rti: 'RTI', cyber: 'سائبر', civil: 'دیوانی', general: 'عام' },
};

// Short UI labels used inside chat (badges, source cards, actions).
const UI_LABELS: Record<Language, Record<string, string>> = {
  en: { sources: 'Sources', view: 'View', hide: 'Hide', read_full: 'Read full text', read_less: 'Read less', edit: 'Edit', copy: 'Copy', save: 'Save', saved: 'Saved', is_this_illegal: 'Is this illegal?' },
  hi: { sources: 'स्रोत', view: 'देखें', hide: 'छिपाएँ', read_full: 'पूरा पाठ पढ़ें', read_less: 'कम पढ़ें', edit: 'संपादित करें', copy: 'कॉपी', save: 'सहेजें', saved: 'सहेजा गया', is_this_illegal: 'क्या यह गैरकानूनी है?' },
  gu: { sources: 'સ્ત્રોતો', view: 'જુઓ', hide: 'છુપાવો', read_full: 'સંપૂર્ણ લખાણ વાંચો', read_less: 'ઓછું વાંચો', edit: 'સંપાદિત કરો', copy: 'કૉપિ', save: 'સાચવો', saved: 'સાચવેલ', is_this_illegal: 'શું આ ગેરકાયદેસર છે?' },
  mr: { sources: 'स्रोत', view: 'पहा', hide: 'लपवा', read_full: 'संपूर्ण मजकूर वाचा', read_less: 'कमी वाचा', edit: 'संपादित करा', copy: 'कॉपी', save: 'जतन करा', saved: 'जतन केले', is_this_illegal: 'हे बेकायदेशीर आहे का?' },
  ta: { sources: 'மூலங்கள்', view: 'காண்க', hide: 'மறை', read_full: 'முழு உரையைப் படிக்கவும்', read_less: 'குறைவாக படி', edit: 'திருத்து', copy: 'நகலெடு', save: 'சேமி', saved: 'சேமிக்கப்பட்டது', is_this_illegal: 'இது சட்டவிரோதமானதா?' },
  te: { sources: 'మూలాలు', view: 'చూడండి', hide: 'దాచండి', read_full: 'పూర్తి వచనం చదవండి', read_less: 'తక్కువ చదవండి', edit: 'సవరించు', copy: 'కాపీ', save: 'సేవ్', saved: 'సేవ్ చేయబడింది', is_this_illegal: 'ఇది చట్టవిరుద్ధమా?' },
  kn: { sources: 'ಮೂಲಗಳು', view: 'ನೋಡಿ', hide: 'ಮರೆಮಾಡಿ', read_full: 'ಪೂರ್ಣ ಪಠ್ಯ ಓದಿ', read_less: 'ಕಡಿಮೆ ಓದಿ', edit: 'ಸಂಪಾದಿಸಿ', copy: 'ನಕಲಿಸಿ', save: 'ಉಳಿಸಿ', saved: 'ಉಳಿಸಲಾಗಿದೆ', is_this_illegal: 'ಇದು ಕಾನೂನುಬಾಹಿರವೇ?' },
  bn: { sources: 'সূত্র', view: 'দেখুন', hide: 'লুকান', read_full: 'সম্পূর্ণ পাঠ্য পড়ুন', read_less: 'কম পড়ুন', edit: 'সম্পাদনা', copy: 'কপি', save: 'সংরক্ষণ', saved: 'সংরক্ষিত', is_this_illegal: 'এটা কি অবৈধ?' },
  ml: { sources: 'ഉറവിടങ്ങൾ', view: 'കാണുക', hide: 'മറയ്ക്കുക', read_full: 'മുഴുവൻ വാചകം വായിക്കുക', read_less: 'കുറച്ച് വായിക്കുക', edit: 'തിരുത്തുക', copy: 'പകർത്തുക', save: 'സംരക്ഷിക്കുക', saved: 'സംരക്ഷിച്ചു', is_this_illegal: 'ഇത് നിയമവിരുദ്ധമാണോ?' },
  pa: { sources: 'ਸਰੋਤ', view: 'ਵੇਖੋ', hide: 'ਛੁਪਾਓ', read_full: 'ਪੂਰਾ ਪਾਠ ਪੜ੍ਹੋ', read_less: 'ਘੱਟ ਪੜ੍ਹੋ', edit: 'ਸੰਪਾਦਿਤ ਕਰੋ', copy: 'ਕਾਪੀ', save: 'ਸੰਭਾਲੋ', saved: 'ਸੰਭਾਲਿਆ', is_this_illegal: 'ਕੀ ਇਹ ਗੈਰ-ਕਾਨੂੰਨੀ ਹੈ?' },
  ur: { sources: 'ذرائع', view: 'دیکھیں', hide: 'چھپائیں', read_full: 'مکمل متن پڑھیں', read_less: 'کم پڑھیں', edit: 'ترمیم', copy: 'کاپی', save: 'محفوظ کریں', saved: 'محفوظ', is_this_illegal: 'کیا یہ غیر قانونی ہے؟' },
};

// Backend response_type values -> display category (backend emits e.g.
// "civil_only" / "criminal_and_civil" / "cyber_defamation").
const DOMAIN_ALIASES: Record<string, LegalCategory> = {
  civil_only: 'civil',
  criminal_only: 'criminal',
  criminal_and_civil: 'criminal',
  both_criminal_civil: 'criminal',
  cyber_defamation: 'cyber',
  labor: 'labour',
};

export function localizeDomain(lang: Language, domain?: string): string {
  const key = String(domain || '').toLowerCase();
  const cat = DOMAIN_ALIASES[key] ?? (key as LegalCategory);
  return DOMAIN_LABELS[lang]?.[cat] ?? String(domain || '').toUpperCase();
}

export function uiLabel(lang: Language, key: string): string {
  return UI_LABELS[lang]?.[key] ?? UI_LABELS.en[key] ?? key;
}
